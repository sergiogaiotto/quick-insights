"""
Quick Insights — Deep Agent Service

Architecture based on deepagents/examples/text-to-sql-agent:
- Uses OpenAI via langchain-openai
- SQLDatabaseToolkit from langchain-community for SQL tools
- LangGraph for agent orchestration with tool calling
- Progressive disclosure: AGENTS.md (always loaded) + skills/ (on-demand)
- Planning via structured system prompt with guardrails
"""

from typing import Annotated, TypedDict
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.core.database import (
    engine,
    get_sync_connection,
    get_table_schema_text,
    execute_readonly_sql,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sql_query: str
    query_result: dict
    analysis_type_id: int | None


# ---------------------------------------------------------------------------
# Skills loader (progressive disclosure)
# ---------------------------------------------------------------------------

def _load_agents_md() -> str:
    """Load the always-present AGENTS.md identity file."""
    path = settings.agents_md
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _load_skill(skill_name: str) -> str:
    """Load a specific skill's SKILL.md content."""
    path = settings.skills_dir / skill_name / "SKILL.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _get_skills_summary() -> str:
    """Get a summary of available skills for the agent context."""
    skills_dir = settings.skills_dir
    if not skills_dir.exists():
        return ""

    summaries = []
    for skill_dir in sorted(skills_dir.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text(encoding="utf-8")
            # Extract name and description from frontmatter
            lines = content.split("\n")
            name = skill_dir.name
            description = ""
            for line in lines:
                if line.startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                    break
            summaries.append(f"- **{name}**: {description}")

    if summaries:
        return "## Available Skills\n" + "\n".join(summaries)
    return ""


# ---------------------------------------------------------------------------
# Analysis type config
# ---------------------------------------------------------------------------

def _get_analysis_config(analysis_type_id: int | None) -> dict:
    """Load system prompt and guardrails for an analysis type."""
    default = {
        "system_prompt": (
            "Você é um analista de dados especialista. Responda em português do Brasil. "
            "Gere SQL ANSI compatível com SQLite. Explique os resultados de forma clara."
        ),
        "guardrails_input": "",
        "guardrails_output": "",
    }
    if not analysis_type_id:
        return default

    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT system_prompt, guardrails_input, guardrails_output "
            "FROM analysis_types WHERE id = ?",
            (analysis_type_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "system_prompt": row[0] or default["system_prompt"],
                "guardrails_input": row[1] or "",
                "guardrails_output": row[2] or "",
            }
        return default
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Build Deep Agent graph
# ---------------------------------------------------------------------------

def build_agent():
    """
    Build and compile the LangGraph Deep Agent.

    Uses the same pattern as deepagents/examples/text-to-sql-agent:
    - ChatOpenAI as the LLM
    - SQLDatabaseToolkit for database interaction tools
    - Progressive disclosure via AGENTS.md + skills
    - LangGraph StateGraph for orchestration
    """

    # --- LLM ---
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    # --- SQL Database (via SQLAlchemy, like deepagents example) ---
    db = SQLDatabase(engine=engine, sample_rows_in_table_info=3)

    # --- SQL Toolkit (same as deepagents example) ---
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()

    # --- Bind tools to LLM ---
    llm_with_tools = llm.bind_tools(sql_tools)

    # --- Load progressive disclosure content ---
    agents_md = _load_agents_md()
    skills_summary = _get_skills_summary()
    query_skill = _load_skill("query-writing")
    schema_skill = _load_skill("schema-exploration")

    # --- Agent node ---
    def agent_node(state: AgentState):
        config = _get_analysis_config(state.get("analysis_type_id"))
        schema_text = get_table_schema_text()

        system_content = f"""{config['system_prompt']}

## Agent Identity
{agents_md}

{skills_summary}

## Skill: Query Writing
{query_skill}

## Skill: Schema Exploration
{schema_skill}

## Current Database Schema
{schema_text}

## Guardrails de Entrada
{config['guardrails_input']}

## Guardrails de Saída
{config['guardrails_output']}
"""
        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # --- Routing ---
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # --- Build graph ---
    tool_node = ToolNode(sql_tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def reset_agent():
    """Reset agent (e.g. after schema changes from Excel upload)."""
    global _agent
    _agent = None


# ---------------------------------------------------------------------------
# SQL LIMIT helper
# ---------------------------------------------------------------------------

import re as _re


def _apply_limit(sql: str, limit: int | None) -> str:
    """Enforce a LIMIT on a SQL query.

    - limit=0 or limit=None → remove existing LIMIT (return all rows)
    - limit>0 → replace or append LIMIT
    """
    pattern = _re.compile(r'\s*\bLIMIT\s+\d+\b', _re.IGNORECASE)

    if limit is None or limit <= 0:
        # Remove any existing LIMIT clause
        return pattern.sub('', sql)

    if pattern.search(sql):
        return pattern.sub(f' LIMIT {limit}', sql)

    stripped = sql.rstrip().rstrip(';').rstrip()
    return f"{stripped} LIMIT {limit}"


# ---------------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------------

async def run_query(
    question: str,
    analysis_type_id: int | None = None,
    context: str | None = None,
    result_limit: int | None = 20,
) -> dict:
    """Run a natural language query through the Deep Agent."""
    agent = get_agent()

    messages = []
    if context:
        messages.append(HumanMessage(content=f"Contexto anterior: {context}"))
        messages.append(AIMessage(content="Entendido, vou considerar o contexto anterior."))
    messages.append(HumanMessage(content=question))

    result = agent.invoke({
        "messages": messages,
        "sql_query": "",
        "query_result": {},
        "analysis_type_id": analysis_type_id,
    })

    # Extract results from message history
    final_messages = result["messages"]
    ai_response = ""
    sql_generated = ""

    for msg in final_messages:
        # Capture SQL from tool calls (last one wins)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] in ("sql_db_query", "execute_query", "query_database"):
                    sql_generated = tc["args"].get("query", tc["args"].get("sql", ""))

        # Capture final AI text response (last non-tool AI message)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                if msg.type == "ai" and msg.content.strip():
                    ai_response = msg.content

    # Re-execute the captured SQL to get structured {columns, rows, row_count}
    # SQLDatabaseToolkit returns string results, not structured data,
    # so we re-run through our own executor that returns the proper format.
    data = {}
    if sql_generated:
        sql_to_run = _apply_limit(sql_generated, result_limit)
        data = execute_readonly_sql(sql_to_run)
        # If execution failed (e.g. query was already consumed), data has "error" key
        if "error" in data:
            data = {}

    # Save to history
    conn = get_sync_connection()
    try:
        conn.execute(
            "INSERT INTO query_history (question, sql_generated, result_summary, analysis_type_id) "
            "VALUES (?, ?, ?, ?)",
            (question, sql_generated, ai_response[:500] if ai_response else "", analysis_type_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "question": question,
        "sql_generated": sql_generated,
        "explanation": ai_response,
        "data": data,
    }