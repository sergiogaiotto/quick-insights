# QUICK**INSIGHTS**

**Transforme perguntas em respostas.** Plataforma de análise de dados com linguagem natural, visualização interativa, modelagem preditiva e Deep Agent autônomo que converte português em SQL.

Desenvolvido por [Sergio Gaiotto](https://www.falagaiotto.com.br) — Especialista, pesquisador e educador em dados e inteligência artificial aplicada.

---

## Visão Geral

Quick Insights é uma aplicação open-source que elimina a barreira técnica entre pessoas e seus dados. O sistema combina um **Deep Agent** com planejamento autônomo, exploração de schema e escrita de queries via skills progressivas, um motor de **análise estatística descritiva e preditiva** com modelos de machine learning, e **visualização interativa** com PyGWalker e Chart.js — tudo acessível via interface web em português brasileiro.

Faz parte de uma iniciativa aplicada da [Fala Gaiotto](https://www.falagaiotto.com.br) para democratizar o acesso a dados e inteligência artificial.

---

## Funcionalidades

### Consulta em Linguagem Natural

O núcleo da aplicação. O usuário digita uma pergunta em português e o Deep Agent autonomamente explora o banco de dados, identifica tabelas e colunas relevantes, gera SQL otimizado, executa e retorna resultados formatados com insights. Suporta contexto conversacional — perguntas de acompanhamento mantêm a referência da conversa anterior.

O limite de registros retornados é configurável via dropdown na interface (20, 50, 100, 500, 1000 ou Todos). A lógica de LIMIT é aplicada após a geração do SQL pelo agente, garantindo que o usuário controla o volume independente do que o LLM decide.

### Upload de Excel

Arquivos `.xlsx` são importados diretamente pela interface. Cada aba da planilha é convertida em uma tabela SQLite — se a tabela já existe, é recriada com os novos dados. Não há limite de abas ou colunas. Após o upload, as tabelas ficam imediatamente disponíveis para consulta.

### System Prompts e Tipos de Análise

O sistema permite criar, editar e excluir tipos de análise customizados. Cada tipo define um system prompt próprio com guardrails de entrada e saída, controlando o comportamento do agente. Isso permite que uma mesma instalação atenda diferentes contextos — análise financeira, marketing, operações — cada um com instruções e restrições específicas.

### Visualização Interativa (Explorar)

Após qualquer consulta, o botão "Explorar" abre o **PyGWalker** — um ambiente drag-and-drop para criação de visualizações. O usuário arrasta colunas para eixos, aplica filtros, muda tipos de gráfico, tudo sem código. O estado completo (HTML + localStorage) é preservado para salvar na galeria.

### Gráficos Inteligentes (Chart.js + LLM)

O botão "Gráfico" envia os dados para o LLM, que analisa colunas, tipos e distribuição para recomendar automaticamente a melhor visualização. O resultado é um gráfico Chart.js renderizado com configuração otimizada — tipo de gráfico, eixos, cores, legendas — sem intervenção manual.

### Galeria de Análises

Visualizações podem ser salvas na galeria com título e descrição. Cada item recebe um token único de compartilhamento, permitindo acesso via URL pública sem autenticação. A galeria preserva o estado completo do PyGWalker, incluindo filtros e configurações visuais aplicadas.

### Análise Avançada — Estatística Descritiva

O módulo de análise avançada gera automaticamente um dashboard estatístico completo a partir dos dados da consulta. A aba descritiva inclui:

**Tendência Central** — média, mediana, moda para cada coluna numérica.

**Dispersão** — desvio padrão, variância, amplitude (max - min), coeficiente de variação.

**Posição** — quartis (Q1, Q2, Q3), amplitude interquartil (IQR), percentis 10 e 90.

**Histogramas** — distribuição de frequência para cada coluna numérica com bins automáticos via Chart.js.

**Matriz de Correlação** — correlação de Pearson entre até 12 colunas numéricas, renderizada como heatmap interativo com escala de cor (verde = correlação positiva, vermelho = negativa), tooltips com nomes completos e valores exatos.

**Tabelas de Frequência** — contagem e percentual para colunas categóricas, com gráficos de barras associados.

### Análise Avançada — Modelagem Preditiva

Três modelos estatísticos disponíveis, todos com label encoding automático para variáveis categóricas:

**Regressão Linear** — prevê valores numéricos contínuos. Métricas específicas: R², R² Ajustado, MAE, MSE, RMSE, MAPE, Variância Explicada. Gráfico de dispersão Real vs Previsto com linha de referência. Coeficientes com barras de magnitude relativa e intercepto. Métricas de classificação derivadas por binarização na mediana (AUC, Precision, Recall, F1, KS, Acurácia).

**Regressão Logística** — classifica em categorias, sem restrição de quantidade de classes. Solver `lbfgs` com fallback para `saga` + StandardScaler quando necessário (`max_iter=5000`). Métricas de classificação: Acurácia, Precision, Recall, F1-Score, AUC-ROC, KS (Kolmogorov-Smirnov). Matriz de confusão com highlighting diagonal. Curva ROC para classificação binária. Gráfico de distribuição das predições.

**Clusterização K-Means** — agrupamento não supervisionado sem variável alvo. O usuário pode definir a quantidade de clusters (2-20) ou deixar 0 para seleção automática via maior Silhouette Score. Métricas: Silhouette, Inertia, Calinski-Harabasz, Davies-Bouldin. Gráfico de dispersão colorido por cluster. Gráfico do Método do Cotovelo (Elbow) com curvas de Inertia e Silhouette em eixos duais, acompanhado de texto explicativo com o racional da seleção de K. Matriz de distância euclidiana entre centróides (heatmap + gráfico de barras). Tabela de perfis com médias por cluster.

Todos os modelos exibem o bloco padronizado de 6 métricas estatísticas: AUC, Precision, Recall, KS, F1-Score e Acurácia.

### Exportação e Email

Qualquer resultado de consulta pode ser exportado para `.xlsx` com um clique. O sistema também suporta envio direto por email via Exchange/Outlook — o resultado é enviado como corpo HTML com anexo Excel, incluindo introdução padronizada.

### API Externa

Endpoint REST (`/api/v1/query`) para integração com sistemas externos. Autenticação via header `X-API-Key` com hash SHA256 + salt. Chaves são geradas e gerenciadas pela interface administrativa. Permite que aplicações terceiras consultem os dados usando a mesma infraestrutura de linguagem natural.

### Histórico de Consultas

Todas as consultas são registradas automaticamente com pergunta original, SQL gerado, resumo dos resultados e tipo de análise utilizado. Acessível via interface para revisão e auditoria.

---

## Arquitetura

```
quick-insights/
├── run.py                              # Ponto de entrada (uvicorn)
├── requirements.txt                    # Dependências Python
├── pyproject.toml                      # Configuração do projeto
├── .env                                # Variáveis de ambiente
├── AGENTS.md                           # Identidade e instruções do agente
├── agent.py                            # CLI do agente (execução direta)
│
├── skills/                             # Workflows especializados (on-demand)
│   ├── query-writing/
│   │   └── SKILL.md                   # Como escrever e executar SQL
│   └── schema-exploration/
│       └── SKILL.md                   # Como descobrir estrutura do banco
│
├── app/
│   ├── main.py                         # FastAPI app + startup + routing
│   │
│   ├── api/
│   │   └── routes.py                  # 20+ endpoints REST
│   │
│   ├── core/
│   │   ├── config.py                  # Settings (pydantic-settings + .env)
│   │   ├── database.py                # SQLAlchemy engine + schema + SQL exec
│   │   └── security.py                # API keys com SHA256 + salt
│   │
│   ├── models/
│   │   └── schemas.py                 # Pydantic: request/response models
│   │
│   ├── services/
│   │   ├── agent_service.py           # Deep Agent: planning + tools + LLM
│   │   ├── analytics_service.py       # Motor estatístico + ML + renderização
│   │   ├── viz_service.py             # PyGWalker + Chart.js LLM config
│   │   ├── email_service.py           # Exchange/Outlook integration
│   │   └── excel_service.py           # Import Excel → SQLite
│   │
│   ├── templates/
│   │   └── default.html               # Frontend SPA (~1000 linhas)
│   │
│   └── static/                         # Assets estáticos
│
├── data/                               # Banco SQLite (auto-criado)
└── uploads/                            # Arquivos Excel enviados
```

---

## Stack Tecnológica

| Camada | Tecnologia | Função |
|---|---|---|
| Backend | Python 3.11 + FastAPI + Uvicorn | API REST assíncrona |
| Deep Agent | deepagents + LangGraph + LangChain + OpenAI | Planejamento e execução autônoma |
| SQL Toolkit | langchain-community SQLDatabaseToolkit + SQLAlchemy | Interação com banco de dados |
| Machine Learning | scikit-learn + SciPy + NumPy + Pandas | Modelos preditivos e estatísticas |
| Banco de Dados | SQLite | Armazenamento local zero-config |
| Visualização | PyGWalker + Chart.js | Exploração interativa + gráficos |
| Frontend | HTML + Tailwind CSS + JavaScript | SPA com dark theme |
| Email | exchangelib | Outlook / Exchange |
| Segurança | SHA256 + salt | Autenticação API |

---

## Endpoints da API

### Consulta e Dados

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Interface web (SPA) |
| `GET` | `/api/tables` | Listar tabelas com schema e contagem |
| `GET` | `/api/tables/{name}/preview` | Preview de uma tabela (até 100 linhas) |
| `POST` | `/api/upload` | Upload de arquivo Excel (.xlsx) |
| `POST` | `/api/query` | Consulta em linguagem natural |
| `GET` | `/api/history` | Histórico de consultas |

### Análise e Visualização

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/api/explore` | Gerar página PyGWalker (exploração interativa) |
| `POST` | `/api/chart` | Gerar gráfico Chart.js via recomendação LLM |
| `POST` | `/api/analytics` | Dashboard de análise estatística descritiva |
| `POST` | `/api/analytics/predict` | Executar modelo preditivo (linear/logistic/clustering) |

### Galeria e Exportação

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/gallery` | Listar itens da galeria |
| `POST` | `/api/gallery` | Salvar visualização na galeria |
| `DELETE` | `/api/gallery/{id}` | Excluir item da galeria |
| `GET` | `/api/gallery/{token}/view` | Visualizar item público via token |
| `POST` | `/api/export/excel` | Exportar dados para .xlsx |
| `POST` | `/api/email` | Enviar resultados por email |

### Configuração e Segurança

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/analysis-types` | Listar tipos de análise |
| `POST` | `/api/analysis-types` | Criar tipo de análise |
| `PUT` | `/api/analysis-types/{id}` | Atualizar tipo de análise |
| `DELETE` | `/api/analysis-types/{id}` | Excluir tipo de análise |
| `POST` | `/api/keys` | Gerar nova API key |
| `GET` | `/api/keys` | Listar API keys |
| `POST` | `/api/v1/query` | Endpoint externo (autenticado via X-API-Key) |

---

## Deep Agent

### Progressive Disclosure

O agente segue o padrão Deep Agents de carregamento progressivo para otimizar o uso de contexto:

**AGENTS.md** (sempre carregado) — identidade do agente, regras de segurança (read-only, sem DDL/DML), formato de resposta, idioma (português brasileiro), guidelines de SQL.

**skills/** (carregados sob demanda):

`query-writing/SKILL.md` — workflow para escrita de SQL: queries simples (single-table) até complexas (multi-table JOINs, subqueries, agregações). Inclui padrões de validação, tratamento de erros e reformulação automática.

`schema-exploration/SKILL.md` — workflow para descoberta de estrutura: listar tabelas, examinar colunas, tipos, relacionamentos, conteúdo amostral. Usado antes de queries complexas para entender o contexto.

### Tools do Agente

| Tool | Descrição |
|---|---|
| `sql_db_list_tables` | Listar todas as tabelas disponíveis |
| `sql_db_schema` | Schema detalhado de uma tabela específica |
| `sql_db_query` | Executar SQL read-only no banco |
| `sql_db_query_checker` | Validar sintaxe SQL antes da execução |

### Fluxo de Execução

1. O agente recebe a pergunta em português com o system prompt do tipo de análise selecionado.
2. Carrega AGENTS.md e determina quais skills são necessários.
3. Explora o schema do banco (tabelas, colunas, tipos, amostras).
4. Planeja a estratégia de query (identificação de tabelas, JOINs, agregações).
5. Gera SQL, valida via `query_checker`, executa.
6. Se a query falha, analisa o erro e reformula automaticamente.
7. Analisa os resultados e gera explicação com insights.
8. O sistema aplica o LIMIT configurado pelo usuário antes de retornar os dados.

### Segurança do Agente

O agente opera em modo **read-only estrito**. Comandos DDL/DML (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `TRUNCATE`) são bloqueados em duas camadas: nas instruções do AGENTS.md e na função `execute_readonly_sql()` que faz parsing de tokens antes da execução.

---

## Modelos Preditivos — Detalhamento Técnico

### Pré-processamento

Todas as variáveis categóricas (features e target) são automaticamente convertidas via `LabelEncoder` do scikit-learn antes de alimentar qualquer modelo. O usuário não precisa se preocupar com encoding — a interface marca cada coluna com badges `num` (numérica) ou `cat` (categórica) para referência visual.

### Regressão Linear

Modelo: `sklearn.linear_model.LinearRegression`. Split train/test: 80/20 (≥50 registros) ou 70/30 (< 50 registros). Métricas de regressão: R², R² Ajustado (corrigido por número de features), MAE, MSE, RMSE, MAPE, Variância Explicada. Métricas de classificação derivadas: binarização na mediana do target (acima/abaixo), com probabilidade proxy normalizada pela amplitude de predição.

### Regressão Logística

Modelo: `sklearn.linear_model.LogisticRegression` com solver `lbfgs`, `max_iter=5000`. Sem restrição de classes — aceita qualquer quantidade de valores únicos no target. Fallback para solver `saga` com StandardScaler se `lbfgs` falhar (tipicamente em datasets com muitas classes ou features mal escaladas). Para multiclass, precision/recall/F1 usam `average="weighted"`. AUC-ROC usa `multi_class="ovr"` com `average="weighted"`. KS (Kolmogorov-Smirnov) disponível apenas para classificação binária — compara distribuições de probabilidade das classes positiva e negativa via `scipy.stats.ks_2samp`.

### Clusterização K-Means

Modelo: `sklearn.cluster.KMeans` com StandardScaler. Busca de K ótimo: testa K=2 até min(10, n_points/3). Seleção automática (quando K=0): maior Silhouette Score. Seleção manual: o usuário define K (2-20) via input numérico na interface. O racional da seleção é exibido abaixo do gráfico de cotovelo, incluindo o ponto de maior queda percentual de inertia e o Silhouette Score do K escolhido.

Matriz de distância euclidiana entre centróides calculada via `scipy.spatial.distance.cdist`. Renderizada como heatmap tabular (cor por intensidade) e gráfico de barras horizontais (pares de clusters × distância).

### Bloco de Métricas Padronizado

Todos os três modelos exibem um bloco unificado com 6 métricas de classificação: Acurácia, Precision, Recall, F1-Score, AUC-ROC e KS. Para regressão linear, as métricas são calculadas sobre uma binarização pela mediana. Para clusterização, o bloco é exibido com valores "—" (não aplicável por ausência de variável alvo).

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

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `OPENAI_API_KEY` | Sim | — | Chave da API OpenAI |
| `OPENAI_MODEL` | Não | `gpt-4.1` | Modelo LLM utilizado pelo agente |
| `DATABASE_URL` | Não | `sqlite:///data/quick_insights.db` | Connection string do banco |
| `API_SALT` | Não | `default-salt` | Salt para hash SHA256 das API keys |
| `API_SECRET_KEY` | Não | `default-secret` | Chave secreta da aplicação |
| `EMAIL_ADDRESS` | Não | — | Email Outlook para envio de resultados |
| `EMAIL_PASSWORD` | Não | — | Senha ou app password do email |
| `EMAIL_SERVER` | Não | `outlook.office365.com` | Servidor Exchange |
| `HOST` | Não | `0.0.0.0` | Host do servidor |
| `PORT` | Não | `8000` | Porta do servidor |

---

## Schema do Banco de Dados

O SQLite armazena tanto os dados do usuário (tabelas criadas via upload de Excel) quanto as tabelas internas de metadados:

**`analysis_types`** — tipos de análise customizados com system prompt, guardrails de entrada e saída.

**`api_keys`** — chaves de API com hash SHA256, label e status ativo/inativo.

**`query_history`** — registro de todas as consultas com pergunta, SQL gerado, resumo e tipo de análise.

**`analysis_gallery`** — visualizações salvas com dados, config do gráfico, HTML completo do PyGWalker e token de compartilhamento.

Tabelas internas são automaticamente excluídas das listagens e consultas do agente.

---

## Dependências Principais

```
deepagents>=0.3.5          # Framework de agentes autônomos
langgraph>=1.0.6            # Orquestração de grafos de execução
langchain>=1.2.3            # Toolkit de LLM
langchain-openai>=0.3.0     # Integração OpenAI
langchain-community>=0.3.0  # SQL Database Toolkit
fastapi>=0.115.0            # Framework web assíncrono
sqlalchemy>=2.0.0           # ORM e toolkit SQL
pandas>=2.2.0               # Manipulação de dados
scikit-learn>=1.4.0         # Modelos de machine learning
scipy>=1.12.0               # Estatísticas e distâncias
pygwalker>=0.5.0            # Visualização interativa
openpyxl>=3.1.0             # Leitura/escrita de Excel
exchangelib>=5.4.0          # Integração Exchange/Outlook
```

---

## Licença

Apache 2.0

---

<p align="center">
  <strong>QUICK</strong><span style="color:#ff6347">INSIGHTS</span><br>
  <a href="https://www.falagaiotto.com.br">falagaiotto.com.br</a>
</p>
