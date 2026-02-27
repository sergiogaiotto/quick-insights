---
name: query-writing
description: For writing and executing SQL queries — from simple single-table queries to complex multi-table JOINs and aggregations on SQLite databases
---

# Query Writing Skill

## When to Use This Skill

Use this skill when you need to answer a question by writing and executing a SQL query.

## Workflow for Simple Queries

For straightforward questions about a single table:

1. **Identify the table** — Which table has the data?
2. **Get the schema** — Use `get_schema` to see columns
3. **Write the query** — SELECT relevant columns with WHERE/LIMIT/ORDER BY
4. **Validate** — Use `query_checker` to verify syntax
5. **Execute** — Run with `execute_query`
6. **Format answer** — Present results clearly in Portuguese

## Workflow for Complex Queries

For questions requiring multiple tables:

### 1. Plan Your Approach
Break down the task:
- Identify all tables needed
- Map relationships (foreign keys or common columns)
- Plan JOIN structure
- Determine aggregations

### 2. Examine Schemas
Use `get_schema` for EACH table to find join columns and needed fields.

### 3. Construct Query
- SELECT — Columns and aggregates
- FROM/JOIN — Connect tables on matching columns
- WHERE — Filters before aggregation
- GROUP BY — All non-aggregate columns
- HAVING — Filters after aggregation
- ORDER BY — Sort meaningfully
- LIMIT — Default 20 rows

### 4. Validate and Execute
Use `query_checker` first, then `execute_query`.

## SQLite-Specific Notes

- Use double quotes for identifiers: `"table_name"."column_name"`
- SQLite uses `||` for string concatenation
- No native FULL OUTER JOIN — use UNION of LEFT and RIGHT
- Date functions: `date()`, `datetime()`, `strftime()`
- Use `CAST(x AS REAL)` for decimal division
- `GROUP_CONCAT()` for string aggregation
- `COALESCE()` for null handling
- Use `ROUND(value, 2)` for decimal formatting
- `printf('%,.2f', value)` for formatted numbers

## Quality Guidelines

- Query only relevant columns (not SELECT *)
- Always apply LIMIT (20 default)
- Use table aliases for clarity
- For complex queries: plan before executing
- Never use DML statements (INSERT, UPDATE, DELETE, DROP)
- Always respond in Portuguese (Brazil)
- Include insights about the data patterns
