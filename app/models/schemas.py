from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Pergunta em linguagem natural")
    analysis_type_id: Optional[int] = Field(None, description="ID do tipo de análise")
    conversation_context: Optional[str] = Field(None, description="Contexto da conversa anterior")
    result_limit: Optional[int] = Field(20, description="Limite de registros retornados (0 = sem limite)")


class QueryResponse(BaseModel):
    question: str
    sql_generated: str
    explanation: str
    data: dict
    insights: Optional[str] = None


class AnalysisTypeCreate(BaseModel):
    name: str = Field(..., min_length=2)
    system_prompt: str = ""
    guardrails_input: str = ""
    guardrails_output: str = ""


class AnalysisTypeUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    guardrails_input: Optional[str] = None
    guardrails_output: Optional[str] = None


class EmailRequest(BaseModel):
    to_email: str
    subject: str
    body_html: str
    excel_data: Optional[dict] = None


class ApiKeyCreate(BaseModel):
    label: str = Field(..., min_length=2)


class ApiQueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    analysis_type_id: Optional[int] = None


class GallerySaveRequest(BaseModel):
    title: str = Field(..., min_length=2)
    description: str = ""
    query_data: dict = Field(..., description="Dados da consulta {columns, rows}")
    chart_config: Optional[dict] = Field(None, description="Config do gráfico")
    page_html: str = Field("", description="HTML completo da página PyGWalker")
    local_storage: Optional[dict] = Field(None, description="Snapshot do localStorage do PyGWalker")


class PredictionRequest(BaseModel):
    query_data: dict = Field(..., description="Dados da consulta {columns, rows}")
    target: str = Field("", description="Coluna alvo (Y) — vazio para clustering")
    features: list[str] = Field(..., min_length=1, description="Colunas features (X)")
    model_type: str = Field(..., pattern="^(linear|logistic|clustering)$", description="Tipo de modelo")