# Quick Insights

**Transforme perguntas em respostas.** Consulte seus dados usando linguagem natural com um Deep Agent que converte suas perguntas em SQL automaticamente.

Desenvolvido por [Sergio Gaiotto](https://www.falagaiotto.com.br) — Especialista, pesquisador e educador em dados e inteligência artificial aplicada.

---

## Sobre

Quick Insights é uma aplicação open-source que permite a qualquer pessoa consultar bancos de dados usando português natural, sem precisar escrever SQL. O sistema utiliza um **Deep Agent** com planejamento, exploração de schema e escrita de queries via skills progressivas, powered by **OpenAI** e **LangChain Deep Agents**.

Faz parte de uma iniciativa aplicada da [Fala Gaiotto](https://www.falagaiotto.com.br) para democratizar o acesso a dados e inteligência artificial.

---

## Funcionalidades

| Recurso                           | Descrição                                                                       |
|-----------------------------------|---------------------------------------------------------------------------------|
| **Deep Agent com planejamento**   | Agente com skills progressivas, planning e exploração inteligente de schema     |
| **Consulta em linguagem natural** | Pergunte em português e receba respostas com SQL gerado automaticamente         |
| **Upload de Excel**               | Importe arquivos `.xlsx` — cada aba vira uma tabela no banco (cria ou atualiza) |
| **Tabelas existentes**            | Trabalhe diretamente com tabelas já cadastradas, sem necessidade de upload      |
| **System Prompts customizados**   | Configure tipos de análise com guardrails de entrada e saída                    |
| **Exportação Excel**              | Exporte os resultados de qualquer consulta para `.xlsx`                         |
| **Envio por email**               | Envie resultados via Outlook com introdução padrão e anexo Excel                |
| **API externa**                   | Integração REST com autenticação por chave SHA256+salt                          |
| **Contexto de conversa**          | Faça perguntas de acompanhamento mantendo o contexto anterior                   |
| **Identidade visual**             | Dark theme Fala Gaiotto, responsivo, tipografia moderna                         |

---

## Arquitetura

Projeto baseado no padrão **Deep Agents** com progressive disclosure:

```
quick-insights/
├── run.py                         # Ponto de entrada (uvicorn)
├── requirements.txt               # Dependências Python
├── .env                           # Variáveis de ambiente
├── AGENTS.md                      # Identidade e instruções do agente (always loaded)
│
├── skills/                        # Workflows especializados (loaded on-demand)
│   ├── query-writing/
│   │   └── SKILL.md              # Como escrever e executar SQL queries
│   └── schema-exploration/
│       └── SKILL.md              # Como descobrir estrutura do banco
│
├── app/
│   ├── main.py                    # FastAPI app + startup
│   │
│   ├── api/
│   │   └── routes.py              # Endpoints REST
│   │
│   ├── core/
│   │   ├── config.py              # Settings via pydantic-settings
│   │   ├── database.py            # SQLAlchemy + SQLite
│   │   └── security.py            # API keys com SHA256 + salt
│   │
│   ├── models/
│   │   └── schemas.py             # Pydantic: request/response models
│   │
│   ├── services/
│   │   ├── agent_service.py       # Deep Agent: planning + skills + tools
│   │   ├── email_service.py       # Envio Outlook/Exchange + anexo Excel
│   │   └── excel_service.py       # Import multi-aba Excel → SQLite
│   │
│   ├── templates/
│   │   └── default.html           # Frontend SPA
│   │
│   └── static/                    # CSS/JS
│
├── data/                          # Banco SQLite (auto-criado)
└── uploads/                       # Arquivos Excel enviados
```

---

## Stack Tecnológica

| Camada         | Tecnologia                                         |
|----------------|----------------------------------------------------|
| Backend        | Python 3.11 + FastAPI + Uvicorn                    |
| Deep Agent     | deepagents + LangGraph + LangChain + OpenAI        |
| SQL Toolkit    | langchain-community SQLDatabaseToolkit + SQLAlchemy|
| Banco de dados | SQLite                                             |
| Frontend       | HTML + Tailwind CSS + JavaScript                   |
| Email          | exchangelib (Outlook / Exchange)                   |
| Segurança      | SHA256 + salt (API keys)                           |

---

## Setup

### 1. Clone e crie o ambiente virtual

```bash
git clone <repo-url> quick-insights
cd quick-insights
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure o `.env`

```env
OPENAI_API_KEY=sk-sua-chave-aqui
OPENAI_MODEL=gpt-4.1
```

### 4. Execute

```bash
python run.py
```

Acesse **http://localhost:8000**

---

## Variáveis de Ambiente

| Variável            | Obrig | Descrição                                              |
|---------------------|-------|--------------------------------------------------------|
| `OPENAI_API_KEY`    | Sim   | Chave da API OpenAI                                    |
| `OPENAI_MODEL`      | Não   | Modelo (padrão: `gpt-4.1`)                             |
| `API_SALT`          | Não   | Salt para hash SHA256 das API keys                     |
| `API_SECRET_KEY`    | Não   | Chave secreta da aplicação                             |
| `EMAIL_ADDRESS`     | Não   | Email Outlook para envio de resultados                 |
| `EMAIL_PASSWORD`    | Não   | Senha ou app password do email                         |
| `EMAIL_SERVER`      | Não   | Servidor Exchange (padrão: `outlook.office365.com`)    |
| `HOST`              | Não   | Host do servidor (padrão: `0.0.0.0`)                   |
| `PORT`              | Não   | Porta do servidor (padrão: `8000`)                     |

---

## Deep Agent: Como Funciona

### Progressive Disclosure

O agente usa o padrão Deep Agents de carregamento progressivo:

1. **AGENTS.md** (sempre carregado) — identidade, regras gerais, safety
2. **skills/** (sob demanda) — workflows especializados:
   - `query-writing/` — como escrever e executar SQL
   - `schema-exploration/` — como descobrir estrutura do banco

### Tools Disponíveis

| Tool              | Descrição                                    |
|-------------------|----------------------------------------------|
| `list_tables`     | Listar tabelas com schema e contagem         |
| `get_schema`      | Schema detalhado de uma tabela específica    |
| `execute_query`   | Executar SQL read-only no banco              |
| `query_checker`   | Validar SQL antes de executar                |

### Planning

Para consultas complexas, o agente usa planejamento automático:
1. Identifica tabelas necessárias
2. Mapeia relacionamentos
3. Planeja estrutura de JOINs
4. Executa e analisa resultados

---

## Licença

Apache 2.0

---

<p align="center">
  <strong>QUICK</strong><span style="color:#ff6347">INSIGHTS</span><br>
  <a href="https://www.falagaiotto.com.br">falagaiotto.com.br</a>
</p>
