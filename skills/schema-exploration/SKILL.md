---
name: schema-exploration
description: For discovering and understanding database structure, tables, columns, relationships, and data content
---

# Schema Exploration Skill

## When to Use This Skill

Use this skill when you need to:
- Understand the database structure
- Find which tables contain certain types of data
- Discover column names and data types
- Map relationships between tables
- Answer questions like "Quais tabelas existem?" or "Quais colunas tem a tabela X?"

## Workflow

### 1. List All Tables
Use `list_tables` tool to see all available tables in the database.
Returns table names, column info, and row counts.

### 2. Get Schema for Specific Tables
Use `get_schema` tool with table names to examine:
- **Column names** — What fields are available
- **Data types** — INTEGER, REAL, TEXT, etc.
- **Sample data** — Example rows to understand content
- **Row count** — Total records

### 3. Map Relationships
Identify how tables connect:
- Look for columns with similar names across tables
- Common patterns: id columns, name matches
- Document parent-child relationships

### 4. Answer the Question
Provide clear information in Portuguese about:
- Available tables and their purpose
- Column names and what they contain
- How tables relate to each other
- Sample data to illustrate content

## Response Format

**For "listar tabelas" questions:**
- Show all table names with brief descriptions
- Include row counts
- Group related tables when possible

**For "descrever tabela" questions:**
- List all columns with data types
- Explain what each column likely contains
- Show sample data for context
- Note potential relationships to other tables

**For "como consultar X" questions:**
- Identify required tables
- Map the JOIN path
- Explain the relationship chain
- Suggest the query-writing skill for execution

## Tips

- Column names from Excel uploads are sanitized (lowercase, underscores)
- Table names come from Excel sheet names (also sanitized)
- When unsure which table to use, list all tables first
- Always respond in Portuguese (Brazil)
