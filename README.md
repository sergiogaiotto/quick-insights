# QUICK**INSIGHTS**

**Transforme perguntas em respostas.** Plataforma de análise de dados com linguagem natural, visualização interativa, modelagem preditiva e Deep Agent autônomo que converte português em SQL.

Desenvolvido por [Sergio Gaiotto](https://www.falagaiotto.com.br) — Especialista, pesquisador e educador em dados e inteligência artificial aplicada.

---

## Visão Geral

Quick Insights é uma aplicação open-source que elimina a barreira técnica entre pessoas e seus dados. O sistema combina um **Deep Agent** com planejamento autônomo, exploração de schema e escrita de queries via skills progressivas, um motor de **análise estatística descritiva e preditiva** com modelos de machine learning, e **visualização interativa** com PyGWalker e Chart.js — tudo acessível via interface web em português brasileiro, protegido por autenticação com controle de acesso por perfil.

Faz parte de uma iniciativa aplicada da [Fala Gaiotto](https://www.falagaiotto.com.br) para democratizar o acesso a dados e inteligência artificial.

---

## Funcionalidades

### Autenticação e Controle de Acesso

Toda a aplicação é protegida por autenticação obrigatória. O acesso exige login e senha, com sessões gerenciadas via cookie httponly com TTL de 24 horas.

**Primeiro acesso** — quando o banco de dados não possui nenhum usuário cadastrado, a tela de login detecta essa condição e informa que as credenciais digitadas criarão automaticamente uma conta **Administrador** com o nome "Super Usuário". Não há seed, migration manual ou setup externo — basta iniciar o servidor e fazer o primeiro login.

**Dois perfis de acesso:**

**Administrador** — acesso total. Pode criar, editar e excluir usuários, excluir tabelas do banco de dados e acessar todas as funcionalidades da aplicação. Criado automaticamente no primeiro acesso com o nome "Super Usuário".

**Usuário Comum** — acesso às funcionalidades de consulta, visualização, análise, exportação e galeria. Não vê a aba "Usuários" nem o botão de exclusão de tabelas.

Cada usuário possui login, senha (SHA256 + salt), tipo, nome de exibição e descrição do perfil.

### Gerenciamento de Usuários

Aba dedicada na interface (visível apenas para administradores) com tabela CRUD completa. Permite criar novos usuários com definição de tipo e perfil, editar dados existentes, alterar senhas e desativar ou excluir contas. Proteção de integridade: ninguém pode excluir a si mesmo.

### Consulta em Linguagem Natural

O núcleo da aplicação. O usuário digita uma pergunta em português e o Deep Agent autonomamente explora o banco de dados, identifica tabelas e colunas relevantes, gera SQL otimizado, executa e retorna resultados formatados com insights. Suporta contexto conversacional — perguntas de acompanhamento mantêm a referência da conversa anterior.

O limite de registros retornados é configurável via dropdown na interface (20, 50, 100, 500, 1000 ou Todos). A lógica de LIMIT é aplicada após a geração do SQL pelo agente, garantindo que o usuário controla o volume independente do que o LLM decide.

### Upload de Excel e Gestão de Tabelas

Arquivos `.xlsx` são importados diretamente pela interface. Cada aba da planilha é convertida em uma tabela SQLite — se a tabela já existe, é recriada com os novos dados. Não há limite de abas ou colunas. Após o upload, as tabelas ficam imediatamente disponíveis para consulta.

Administradores podem **excluir tabelas** diretamente pela aba Tabelas. O botão de exclusão (ícone de lixeira) aparece no hover de cada tabela e exige dupla confirmação antes de executar o `DROP TABLE`. Tabelas internas do sistema são protegidas contra exclusão.

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

**Regressão Linear** — prevê valores numéricos contínuos. Métricas específicas: R², R² Ajustado, MAE, MSE, RMSE, MAPE, Variância Explicada. Gráfico de dispersão Real vs Previsto com linha de referência. Tabela de coeficientes com inferência estatística completa (detalhada abaixo). Métricas de classificação derivadas por binarização na mediana (AUC, Precision, Recall, F1, KS, Acurácia).

**Regressão Logística** — classifica em categorias, sem restrição de quantidade de classes. Solver `lbfgs` com fallback para `saga` + StandardScaler quando necessário (`max_iter=5000`). Métricas de classificação: Acurácia, Precision, Recall, F1-Score, AUC-ROC, KS (Kolmogorov-Smirnov). Tabela de coeficientes com inferência estatística completa (detalhada abaixo). Matriz de confusão com highlighting diagonal. Curva ROC para classificação binária. Gráfico de distribuição das predições.

**Clusterização K-Means** — agrupamento não supervisionado sem variável alvo. O usuário pode definir a quantidade de clusters (2-20) ou deixar 0 para seleção automática via maior Silhouette Score. Métricas: Silhouette, Inertia, Calinski-Harabasz, Davies-Bouldin. Gráfico de dispersão colorido por cluster. Gráfico do Método do Cotovelo (Elbow) com curvas de Inertia e Silhouette em eixos duais, acompanhado de texto explicativo com o racional da seleção de K. Matriz de distância euclidiana entre centróides (heatmap + gráfico de barras). Tabela de perfis com médias por cluster.

Todos os modelos exibem o bloco padronizado de 6 métricas estatísticas: AUC, Precision, Recall, KS, F1-Score e Acurácia.

### Tabela de Coeficientes — Inferência Estatística

Para regressão linear e logística, o sistema calcula e exibe uma tabela completa de inferência para cada variável (incluindo intercepto), com 7 colunas:

**Coeff (B)** — coeficiente estimado. Magnitude e direção do efeito da variável sobre o alvo.

**S.E.** — erro padrão da estimativa do coeficiente. Quanto menor, mais precisa a estimativa.

**Wald / t** — estatística de teste. Na regressão logística: Wald χ² = (B / S.E.)². Na regressão linear: estatística t = B / S.E.

**p-valor** — probabilidade de observar o efeito por acaso. p < 0.05 indica significância estatística com 95% de confiança.

**Exp(B)** — exponencial do coeficiente. Na logística, representa o odds ratio (multiplicador da chance). Na linear, indica o fator multiplicativo por unidade de variação.

**Inferior** — limite inferior do intervalo de confiança a 95%. Na logística, calculado sobre Exp(B). Na linear, sobre o coeficiente B.

**Superior** — limite superior do intervalo de confiança a 95%.

Cada coluna possui um tooltip interativo com explicação detalhada, contextualizada conforme o tipo de modelo. Variáveis estatisticamente significativas (p < 0.05) são destacadas com ★ verde e background diferenciado. Todos os valores são formatados com até 10 casas decimais, sem separador de milhar.

**Cálculo na regressão linear:** matriz de covariância via `(XᵀX)⁻¹ · MSE`, erro padrão = raiz da diagonal, estatística t bilateral, p-value via distribuição t com n-p-1 graus de liberdade, IC 95% = B ± t₀.₉₇₅ · S.E.

**Cálculo na regressão logística:** Fisher Information Matrix `XᵀWX` onde W = p̂(1-p̂) para binário, média OVR para multiclass. Wald = (B/S.E.)², p-value via χ² com 1 grau de liberdade, IC 95% para Exp(B) = e^(B ± 1.96·S.E.).

### Recomendação de Variáveis

Logo abaixo da tabela de coeficientes, o sistema gera automaticamente uma recomendação baseada na significância estatística. Lista as variáveis significativas ordenadas por p-valor (mais relevantes primeiro), com direção do efeito e Exp(B). Identifica as variáveis não significativas e sugere sua remoção para simplificação do modelo.

### Exportação e Email

Qualquer resultado de consulta pode ser exportado para `.xlsx` com um clique. O sistema também gera arquivos `.eml` para envio via Outlook local — ao clicar em "Enviar Email", o sistema monta um arquivo `.eml` com destinatário, assunto, corpo HTML e anexo Excel embutido. O header `X-Unsent: 1` faz com que o Outlook abra o arquivo como rascunho pronto para envio, sem necessidade de configuração de servidor ou credenciais.

### API Externa

Endpoint REST (`/api/v1/query`) para integração com sistemas externos. Autenticação via header `X-API-Key` com hash SHA256 + salt (independente da autenticação por sessão). Chaves são geradas e gerenciadas pela interface administrativa. Permite que aplicações terceiras consultem os dados usando a mesma infraestrutura de linguagem natural.

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
│   ├── main.py                         # FastAPI app + auth middleware + routing
│   │
│   ├── api/
│   │   └── routes.py                  # 35 endpoints REST (auth + CRUD + analytics)
│   │
│   ├── core/
│   │   ├── config.py                  # Settings (pydantic-settings + .env)
│   │   ├── database.py                # SQLAlchemy engine + schema + SQL exec + drop
│   │   └── security.py                # Auth, sessões, API keys, gestão de usuários
│   │
│   ├── models/
│   │   └── schemas.py                 # Pydantic: request/response models
│   │
│   ├── services/
│   │   ├── agent_service.py           # Deep Agent: planning + tools + LLM
│   │   ├── analytics_service.py       # Motor estatístico + ML + coeff table + render
│   │   ├── viz_service.py             # PyGWalker + Chart.js LLM config
│   │   ├── email_service.py           # Exchange/Outlook integration
│   │   └── excel_service.py           # Import Excel → SQLite
│   │
│   ├── templates/
│   │   ├── login.html                 # Tela de login (standalone)
│   │   └── default.html               # Frontend SPA principal
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
| Email | Python email (stdlib) + .eml | Outlook local via download |
| Autenticação | SHA256 + salt + sessões httponly | Login, perfis, controle de acesso |
| API Keys | SHA256 + salt | Autenticação REST externa |

---

## Endpoints da API

### Autenticação

| Método | Rota | Acesso | Descrição |
|---|---|---|---|
| `POST` | `/api/auth/login` | Público | Autenticar e criar sessão |
| `POST` | `/api/auth/logout` | Autenticado | Encerrar sessão |
| `GET` | `/api/auth/me` | Autenticado | Dados do usuário logado |
| `GET` | `/api/auth/check` | Público | Verificar sessão + existência de usuários |

### Gerenciamento de Usuários

| Método | Rota | Acesso | Descrição |
|---|---|---|---|
| `GET` | `/api/users` | Admin | Listar todos os usuários |
| `POST` | `/api/users` | Admin | Criar novo usuário |
| `PUT` | `/api/users/{id}` | Admin | Atualizar dados do usuário |
| `PUT` | `/api/users/{id}/password` | Admin | Alterar senha |
| `DELETE` | `/api/users/{id}` | Admin | Excluir usuário |

### Consulta e Dados

| Método | Rota | Acesso | Descrição |
|---|---|---|---|
| `GET` | `/` | Autenticado | Interface web (SPA) |
| `GET` | `/login` | Público | Tela de login |
| `GET` | `/api/tables` | Autenticado | Listar tabelas com schema e contagem |
| `GET` | `/api/tables/{name}/preview` | Autenticado | Preview de uma tabela (até 100 linhas) |
| `DELETE` | `/api/tables/{name}` | Admin | Excluir tabela (DROP TABLE) |
| `POST` | `/api/upload` | Autenticado | Upload de arquivo Excel (.xlsx) |
| `POST` | `/api/query` | Autenticado | Consulta em linguagem natural |
| `GET` | `/api/history` | Autenticado | Histórico de consultas |

### Análise e Visualização

| Método | Rota | Acesso | Descrição |
|---|---|---|---|
| `POST` | `/api/explore` | Autenticado | Gerar página PyGWalker |
| `POST` | `/api/chart` | Autenticado | Gerar gráfico Chart.js via LLM |
| `POST` | `/api/analytics` | Autenticado | Dashboard estatístico descritivo |
| `POST` | `/api/analytics/predict` | Autenticado | Modelo preditivo (linear/logistic/clustering) |

### Galeria e Exportação

| Método | Rota | Acesso | Descrição |
|---|---|---|---|
| `GET` | `/api/gallery` | Autenticado | Listar itens da galeria |
| `POST` | `/api/gallery` | Autenticado | Salvar visualização na galeria |
| `DELETE` | `/api/gallery/{id}` | Autenticado | Excluir item da galeria |
| `GET` | `/api/gallery/{token}/view` | Público | Visualizar item público via token |
| `POST` | `/api/export/excel` | Autenticado | Exportar dados para .xlsx |
| `POST` | `/api/email` | Autenticado | Gerar .eml para Outlook local |

### Configuração e API Externa

| Método | Rota | Acesso | Descrição |
|---|---|---|---|
| `GET` | `/api/analysis-types` | Autenticado | Listar tipos de análise |
| `GET` | `/api/analysis-types/{id}` | Autenticado | Obter tipo de análise específico |
| `POST` | `/api/analysis-types` | Autenticado | Criar tipo de análise |
| `PUT` | `/api/analysis-types/{id}` | Autenticado | Atualizar tipo de análise |
| `DELETE` | `/api/analysis-types/{id}` | Autenticado | Excluir tipo de análise |
| `POST` | `/api/keys` | Autenticado | Gerar nova API key |
| `GET` | `/api/keys` | Autenticado | Listar API keys |
| `POST` | `/api/v1/query` | API Key | Endpoint externo (autenticado via X-API-Key) |

---

## Autenticação — Detalhamento Técnico

### Middleware HTTP

Todas as requisições passam pelo middleware `auth_middleware` em `main.py`. Rotas públicas (login, auth/check, auth/login, static, gallery view, API v1) são liberadas. Demais rotas exigem cookie de sessão válido — requisições de página redirecionam para `/login`, requisições de API retornam HTTP 401.

### Sessões

Token gerado via `secrets.token_urlsafe(48)`. Armazenado na tabela `sessions` com `user_id` e `expires_at`. Cookie `qi_session` com flags `httponly`, `samesite=lax`, `max_age=86400` (24h). Sessões expiradas são limpas automaticamente a cada novo login.

### Primeiro Acesso

A função `authenticate_user()` verifica se a tabela `users` tem 0 registros. Se sim, cria o usuário com as credenciais fornecidas como `admin` com display_name "Super Usuário" e descrição automática. A tela de login detecta essa condição via `/api/auth/check` e exibe aviso informativo.

### Hierarquia de Permissões

A dependency `require_admin` em FastAPI bloqueia endpoints de gestão para usuários comuns. O frontend oculta a aba "Usuários" e botões de exclusão de tabelas para quem não é administrador. Proteções adicionais no backend: ninguém pode excluir a si mesmo.

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

Modelo: `sklearn.linear_model.LinearRegression`. Split train/test: 80/20 (≥50 registros) ou 70/30 (< 50 registros). Métricas de regressão: R², R² Ajustado (corrigido por número de features), MAE, MSE, RMSE, MAPE, Variância Explicada. Métricas de classificação derivadas: binarização na mediana do target (acima/abaixo), com probabilidade proxy normalizada pela amplitude de predição. Tabela de coeficientes com S.E. via `(XᵀX)⁻¹ · MSE`, estatística t, p-value bilateral e IC 95%.

### Regressão Logística

Modelo: `sklearn.linear_model.LogisticRegression` com solver `lbfgs`, `max_iter=5000`. Sem restrição de classes — aceita qualquer quantidade de valores únicos no target. Fallback para solver `saga` com StandardScaler se `lbfgs` falhar (tipicamente em datasets com muitas classes ou features mal escaladas). Para multiclass, precision/recall/F1 usam `average="weighted"`. AUC-ROC usa `multi_class="ovr"` com `average="weighted"`. KS (Kolmogorov-Smirnov) disponível apenas para classificação binária — compara distribuições de probabilidade das classes positiva e negativa via `scipy.stats.ks_2samp`. Tabela de coeficientes com S.E. via Fisher Information Matrix, Wald χ², p-value e IC 95% para Exp(B).

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
SESSION_SECRET=minha-chave-secreta-de-sessao
```

### 4. Execute

```bash
python run.py
```

Acesse **http://localhost:8000** — será redirecionado para a tela de login. No primeiro acesso, as credenciais informadas criam automaticamente a conta administrador.

---

## Variáveis de Ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `OPENAI_API_KEY` | Sim | — | Chave da API OpenAI |
| `OPENAI_MODEL` | Não | `gpt-4.1` | Modelo LLM utilizado pelo agente |
| `DATABASE_URL` | Não | `sqlite:///data/quick_insights.db` | Connection string do banco |
| `API_SALT` | Não | `default-salt` | Salt para hash SHA256 (senhas e API keys) |
| `API_SECRET_KEY` | Não | `default-secret` | Chave secreta da aplicação |
| `SESSION_SECRET` | Não | `qi-session-secret-change-me` | Secret para gestão de sessões |
| `HOST` | Não | `0.0.0.0` | Host do servidor |
| `PORT` | Não | `8000` | Porta do servidor |

---

## Schema do Banco de Dados

O SQLite armazena tanto os dados do usuário (tabelas criadas via upload de Excel) quanto as tabelas internas de metadados:

**`users`** — contas de usuário com login, password_hash (SHA256+salt), user_type (superuser/admin/user), display_name, profile_description, is_active e timestamps.

**`sessions`** — sessões ativas com token, user_id e expires_at. Sessões expiradas são removidas automaticamente.

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
