# Quick Insights v2.1.0 — Changelog

## Novos arquivos/modificados

### Backend
- `app/core/database.py` — Tabelas `datamarts`, `datamart_tables`, `user_datamarts` + CRUD completo
- `app/core/security.py` — Tipo `root` com acesso total, `is_root()`, `is_admin()` helpers
- `app/models/schemas.py` — Modelos DataMart, ChartRequest, datamart_ids em QueryRequest/UserCreate/UserUpdate
- `app/api/routes.py` — 12 novos endpoints (DataMart CRUD, User export/import, Skill export/import, Chart typed/options)
- `app/services/agent_service.py` — Filtro de tabelas por DataMart via `accessible_tables`
- `app/services/viz_service.py` — Submenu de gráficos (8 tipos), `generate_typed_chart_html()`, `get_chart_options_for_data()`
- `app/main.py` — Versão 2.1.0
- `app/templates/login.html` — Primeiro acesso cria Root

### Novos endpoints
| Método | Rota | Descrição |
|--------|------|-----------|
| GET | /api/datamarts | Listar DataMarts |
| POST | /api/datamarts | Criar DataMart |
| PUT | /api/datamarts/{id} | Atualizar DataMart |
| DELETE | /api/datamarts/{id} | Excluir DataMart |
| POST | /api/datamarts/{id}/tables | Associar tabela ao DataMart |
| DELETE | /api/datamarts/{id}/tables/{name} | Remover tabela do DataMart |
| GET | /api/datamarts/user | DataMarts do usuário logado |
| GET | /api/users/export | Exportar usuários para Excel |
| POST | /api/users/import | Importar usuários via Excel |
| GET | /api/skills/export/excel | Exportar skills para Excel |
| POST | /api/skills/import | Importar skills via Excel |
| POST | /api/chart/typed | Gráfico com tipo selecionado |
| POST | /api/chart/options | Opções de gráfico para os dados |

## Funcionalidades

### 1. DataMart
- Agrupamento de tabelas em DataMarts
- Combobox/textbox no upload para criar ou selecionar DataMart
- Atribuição múltipla de DataMarts por usuário (checkbox no cadastro)
- Seletor de DataMart na consulta (filtra tabelas acessíveis)
- Root tem acesso a todos os DataMarts

### 2. Tipo Root
- Acesso total: todas tabelas, DataMarts, skills, funcionalidades
- Primeiro login cria conta Root automaticamente
- Hierarquia: Root > Admin > User

### 3. Export/Import Usuários
- Botão "Exportar Excel" na aba Usuários
- Botão "Importar Excel" — senha padrão "minhasenha01", tipo admin, DataMart default

### 4. Export/Import Skills
- Botão "Exportar Excel" na aba Skills
- Botão "Importar Excel" — DataMart default

### 5. Submenu de Gráficos
- Botão "Gráfico" agora abre submenu com 8 tipos:
  Auto (LLM), Barras, Linhas, Dispersão, Área, Pizza, Rosca, Radar, Polar
- Análise de adequação por tipo de dado
