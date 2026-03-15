"""
Quick Insights — Visualization Service

- Explorar: PyGWalker (drag-and-drop livre)
- Gráfico: Chart.js com config LLM ou tipo selecionado pelo usuário
- Galeria: Chart.js com config salva (dados + gráfico persistidos)
"""

import json
import pandas as pd
import pygwalker as pyg
from langchain_openai import ChatOpenAI
from app.core.config import settings


# ---------------------------------------------------------------------------
# LLM chart recommendation
# ---------------------------------------------------------------------------

_SPEC_PROMPT = """Você é um especialista em visualização de dados. Analise os dados abaixo e recomende a melhor visualização.

## Dados
Colunas: {columns}
Tipos: {dtypes}
Amostra (primeiras 5 linhas):
{sample}

## Regras
1. Identifique quais colunas são métricas (numéricas) e quais são dimensões (categóricas/texto)
2. Escolha o tipo de gráfico mais adequado:
   - Se há 1 dimensão + 1 métrica → bar chart (barra vertical)
   - Se há dimensão temporal + métrica → line chart
   - Se há 2 métricas → scatter plot
   - Se há apenas métricas → bar chart com a primeira como eixo
3. Gere APENAS o JSON válido, sem markdown, sem explicação

## Formato de saída (JSON puro)
{{
  "chart_type": "bar|line|scatter|area",
  "x_field": "nome_coluna_eixo_x",
  "y_field": "nome_coluna_eixo_y",
  "color_field": null,
  "agg": "sum|mean|count|none"
}}
"""


def _ask_llm_for_chart_config(df: pd.DataFrame) -> dict | None:
    if not settings.openai_api_key:
        return None
    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample = df.head(5).to_string(index=False)
        prompt = _SPEC_PROMPT.format(
            columns=list(df.columns),
            dtypes=json.dumps(dtypes_info),
            sample=sample,
        )
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()
        return json.loads(content)
    except Exception:
        return None


def _fallback_chart_config(df: pd.DataFrame, chart_type: str = "bar") -> dict:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return {
        "chart_type": chart_type,
        "x_field": non_numeric[0] if non_numeric else df.columns[0],
        "y_field": numeric[0] if numeric else df.columns[-1],
        "color_field": None,
        "agg": "sum",
    }


# ---------------------------------------------------------------------------
# Chart options analysis (for submenu)
# ---------------------------------------------------------------------------

def get_chart_options_for_data(data: dict) -> dict:
    """Analyze data and return available chart types with suitability."""
    df = _data_to_df(data)
    if df is None:
        return {"options": []}

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    n_rows = len(df)
    n_categories = len(non_numeric)
    n_metrics = len(numeric)

    options = [
        {"type": "auto", "label": "Auto (LLM)", "icon": "✨", "suitable": True},
        {"type": "bar", "label": "Barras", "icon": "📊", "suitable": n_categories > 0 or n_metrics > 0},
        {"type": "line", "label": "Linhas", "icon": "📈", "suitable": n_metrics > 0 and n_rows > 2},
        {"type": "scatter", "label": "Dispersão", "icon": "⚡", "suitable": n_metrics >= 2},
        {"type": "area", "label": "Área", "icon": "🗻", "suitable": n_metrics > 0 and n_rows > 2},
        {"type": "pie", "label": "Pizza", "icon": "🥧", "suitable": n_categories > 0 and n_metrics > 0 and n_rows <= 20},
        {"type": "doughnut", "label": "Rosca", "icon": "🍩", "suitable": n_categories > 0 and n_metrics > 0 and n_rows <= 20},
        {"type": "radar", "label": "Radar", "icon": "🕸️", "suitable": n_metrics >= 3 and n_rows <= 15},
        {"type": "polarArea", "label": "Polar", "icon": "🎯", "suitable": n_categories > 0 and n_metrics > 0 and n_rows <= 12},
    ]

    return {
        "options": options,
        "numeric_cols": numeric,
        "categorical_cols": non_numeric,
        "row_count": n_rows,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_to_df(data: dict) -> pd.DataFrame | None:
    rows = data.get("rows", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df if not df.empty else None


def _aggregate_data(df: pd.DataFrame, config: dict) -> tuple[list, list]:
    x_field = config.get("x_field", df.columns[0])
    y_field = config.get("y_field", df.columns[-1])
    agg = config.get("agg", "sum")

    if x_field not in df.columns:
        x_field = df.columns[0]
    if y_field not in df.columns:
        y_field = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

    if agg == "none" or not pd.api.types.is_numeric_dtype(df[y_field]):
        grouped = df[[x_field, y_field]].head(50)
        return grouped[x_field].astype(str).tolist(), grouped[y_field].tolist()

    agg_fn = "sum" if agg == "sum" else ("mean" if agg == "mean" else "count")
    if agg_fn == "count":
        grouped = df.groupby(x_field).size().reset_index(name=y_field)
    else:
        grouped = df.groupby(x_field)[y_field].agg(agg_fn).reset_index()
    grouped = grouped.sort_values(y_field, ascending=False).head(50)
    return grouped[x_field].astype(str).tolist(), grouped[y_field].tolist()


# ---------------------------------------------------------------------------
# Chart.js renderer
# ---------------------------------------------------------------------------

_PALETTE = [
    'rgba(255,99,71,0.7)', 'rgba(88,166,255,0.7)', 'rgba(57,211,83,0.7)',
    'rgba(240,136,62,0.7)', 'rgba(163,113,247,0.7)', 'rgba(63,185,80,0.7)',
    'rgba(210,168,255,0.7)', 'rgba(121,192,255,0.7)', 'rgba(255,166,87,0.7)',
    'rgba(255,123,114,0.7)',
]

_PALETTE_SOLID = [
    '#ff6347', '#58a6ff', '#39d353', '#f0883e', '#a371f7',
    '#3fb950', '#d2a8ff', '#79c0ff', '#ffa657', '#ff7b72',
]


def _render_chartjs_html(
    data: dict,
    config: dict,
    title: str = "Gráfico",
    subtitle: str = "Gráfico Auto-Configurado",
) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()

    x_field = config.get("x_field", df.columns[0])
    y_field = config.get("y_field", df.columns[-1])
    chart_type = config.get("chart_type", "bar")
    agg = config.get("agg", "sum")

    if x_field not in df.columns:
        x_field = df.columns[0]
    if y_field not in df.columns:
        y_field = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

    labels, values = _aggregate_data(df, config)

    # Map chart types to Chart.js
    cjs_type_map = {
        "bar": "bar", "line": "line", "scatter": "scatter", "area": "line",
        "pie": "pie", "doughnut": "doughnut", "radar": "radar", "polarArea": "polarArea",
    }
    cjs_type = cjs_type_map.get(chart_type, "bar")
    fill = "true" if chart_type == "area" else "false"
    is_circular = chart_type in ("pie", "doughnut", "polarArea")

    labels_json = json.dumps(labels, ensure_ascii=False)
    values_json = json.dumps(values)
    palette_json = json.dumps(_PALETTE[:len(labels)])
    palette_solid_json = json.dumps(_PALETTE_SOLID[:len(labels)])
    row_count = len(data.get("rows", []))
    col_count = len(df.columns)

    badge = (
        f'<span style="color:#39d353;font-family:JetBrains Mono,monospace;font-size:11px">'
        f'{chart_type} — X: {x_field} · Y: {y_field} · Agg: {agg}'
        f'</span>'
    )

    if cjs_type == "scatter":
        scatter_data = json.dumps([{"x": l, "y": v} for l, v in zip(labels, values)], ensure_ascii=False)
        dataset_config = f"""{{
                    label: '{y_field}',
                    data: {scatter_data},
                    backgroundColor: 'rgba(255,99,71,0.7)',
                    borderColor: '#ff6347',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                }}"""
    elif is_circular:
        dataset_config = f"""{{
                    label: '{y_field}',
                    data: {values_json},
                    backgroundColor: {palette_json},
                    borderColor: '#0d1117',
                    borderWidth: 2,
                }}"""
    elif cjs_type == "radar":
        dataset_config = f"""{{
                    label: '{y_field}',
                    data: {values_json},
                    backgroundColor: 'rgba(255,99,71,0.2)',
                    borderColor: '#ff6347',
                    borderWidth: 2,
                    pointBackgroundColor: '#ff6347',
                }}"""
    else:
        dataset_config = f"""{{
                    label: '{y_field}',
                    data: {values_json},
                    backgroundColor: {'rgba(255,99,71,0.35)' if not is_circular else palette_json},
                    borderColor: '#ff6347',
                    borderWidth: 2,
                    fill: {fill},
                    tension: 0.3,
                    borderRadius: {('4' if cjs_type == 'bar' else '0')},
                }}"""

    # Scale options (not used for circular/radar charts)
    scales_js = ""
    if not is_circular and cjs_type != "radar":
        scales_js = f"""
                scales: {{
                    x: {{
                        ticks: {{ color: '#8b949e', maxRotation: 45, font: {{ size: 11 }} }},
                        grid: {{ color: '#21262d' }},
                        title: {{ display: true, text: '{x_field}', color: '#c9d1d9', font: {{ size: 12, weight: 600 }} }},
                    }},
                    y: {{
                        ticks: {{ color: '#8b949e', font: {{ size: 11 }} }},
                        grid: {{ color: '#21262d' }},
                        title: {{ display: true, text: '{y_field}', color: '#c9d1d9', font: {{ size: 12, weight: 600 }} }},
                        beginAtZero: true,
                    }},
                }},"""

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quick Insights — {title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:#0d1117; color:#c9d1d9; font-family:'Space Grotesk',sans-serif; height:100vh; display:flex; flex-direction:column; }}
        .qi-header {{ background:#161b22; border-bottom:1px solid #30363d; padding:10px 20px; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }}
        .qi-logo {{ font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:600; }}
        .qi-logo span {{ color:#ff6347; }}
        .qi-info {{ font-size:11px; color:#8b949e; }}
        .qi-chart-wrap {{ flex:1; padding:24px; display:flex; align-items:center; justify-content:center; min-height:0; }}
        .qi-chart-inner {{ width:100%; max-width:1200px; height:100%; position:relative; }}
    </style>
</head>
<body>
    <div class="qi-header">
        <div class="qi-logo">QUICK<span>INSIGHTS</span> — {subtitle}</div>
        <div class="qi-info">
            {badge}
            <span style="margin-left:12px">{row_count} registros · {col_count} colunas</span>
        </div>
    </div>
    <div class="qi-chart-wrap">
        <div class="qi-chart-inner">
            <canvas id="autoChart"></canvas>
        </div>
    </div>
    <script>
        Chart.defaults.color = '#8b949e';
        Chart.defaults.borderColor = '#21262d';
        Chart.defaults.font.family = "'Space Grotesk', sans-serif";

        new Chart(document.getElementById('autoChart'), {{
            type: '{cjs_type}',
            data: {{
                labels: {labels_json},
                datasets: [{dataset_config}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: true, labels: {{ color: '#c9d1d9', font: {{ size: 12 }} }} }},
                    tooltip: {{
                        backgroundColor: '#161b22',
                        borderColor: '#30363d',
                        borderWidth: 1,
                        titleColor: '#ff6347',
                        bodyColor: '#c9d1d9',
                        padding: 10,
                        cornerRadius: 8,
                    }},
                }},
                {scales_js}
                animation: {{ duration: 600, easing: 'easeOutQuart' }},
            }}
        }});
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explore_html(data: dict) -> str:
    """PyGWalker in exploration mode + Save to Gallery button."""
    df = _data_to_df(data)
    if df is None:
        return _empty_html()

    try:
        walker_html = pyg.to_html(df, appearance="dark", default_tab="data")
    except Exception:
        walker_html = pyg.to_html(df, appearance="dark")

    data_json = json.dumps(data)

    toolbar_html = f"""
    <div class="qi-toolbar">
        <button onclick="saveToGallery()" class="qi-btn qi-btn-save">Salvar na Galeria</button>
    </div>
    <div id="saveModal" class="qi-modal" style="display:none">
        <div class="qi-modal-content">
            <h3>Salvar na Galeria de Análises</h3>
            <div class="qi-field">
                <label>Título</label>
                <input type="text" id="saveTitle" placeholder="Nome da análise">
            </div>
            <div class="qi-field">
                <label>Descrição</label>
                <input type="text" id="saveDesc" placeholder="Descrição opcional">
            </div>
            <div class="qi-modal-actions">
                <button onclick="confirmSave()" class="qi-btn qi-btn-confirm">Salvar</button>
                <button onclick="closeSaveModal()" class="qi-btn qi-btn-cancel">Cancelar</button>
            </div>
            <div id="saveStatus" style="margin-top:8px;font-size:11px"></div>
        </div>
    </div>
    <script>
        const _queryData = {data_json};
        function saveToGallery() {{ document.getElementById('saveModal').style.display = 'flex'; document.getElementById('saveTitle').focus(); }}
        function closeSaveModal() {{ document.getElementById('saveModal').style.display = 'none'; document.getElementById('saveStatus').textContent = ''; }}
        async function confirmSave() {{
            const title = document.getElementById('saveTitle').value.trim();
            if (!title) {{ alert('Informe um título.'); return; }}
            const desc = document.getElementById('saveDesc').value.trim();
            const statusEl = document.getElementById('saveStatus');
            statusEl.style.color = '#8b949e'; statusEl.textContent = 'Capturando estado...';
            const toolbar = document.querySelector('.qi-toolbar');
            const modal = document.getElementById('saveModal');
            const td = toolbar ? toolbar.style.display : '';
            const md = modal ? modal.style.display : '';
            if (toolbar) toolbar.style.display = 'none';
            if (modal) modal.style.display = 'none';
            const pageHtml = '<!DOCTYPE html>' + document.documentElement.outerHTML;
            if (toolbar) toolbar.style.display = td;
            if (modal) modal.style.display = 'flex';
            const lsData = {{}};
            for (let i = 0; i < localStorage.length; i++) {{ const key = localStorage.key(i); lsData[key] = localStorage.getItem(key); }}
            statusEl.textContent = 'Salvando...';
            try {{
                const res = await fetch('/api/gallery', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ title, description: desc, query_data: _queryData, page_html: pageHtml, local_storage: lsData }}) }});
                if (res.ok) {{ statusEl.style.color = '#39d353'; statusEl.textContent = 'Salvo na galeria.'; setTimeout(closeSaveModal, 1500); }}
                else {{ statusEl.style.color = '#ff6347'; statusEl.textContent = 'Erro ao salvar.'; }}
            }} catch(e) {{ statusEl.style.color = '#ff6347'; statusEl.textContent = 'Erro: ' + e.message; }}
        }}
    </script>
    """

    badge = '<span style="color:#58a6ff;font-family:JetBrains Mono,monospace;font-size:11px">Modo Exploração</span>'
    row_count = len(data.get("rows", []))
    col_count = len(df.columns)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quick Insights — Explorar</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{ margin:0; padding:0; background:#0d1117; color:#c9d1d9; font-family:'Space Grotesk',sans-serif; }}
        .qi-header {{ background:#161b22; border-bottom:1px solid #30363d; padding:10px 20px; display:flex; align-items:center; justify-content:space-between; }}
        .qi-logo {{ font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:600; }}
        .qi-logo span {{ color:#ff6347; }}
        .qi-info {{ font-size:11px; color:#8b949e; }}
        .qi-toolbar {{ background:#161b22; border-bottom:1px solid #30363d; padding:8px 20px; display:flex; align-items:center; justify-content:flex-end; }}
        .qi-btn {{ padding:5px 14px; border-radius:6px; font-size:11px; font-weight:600; cursor:pointer; border:1px solid; transition:all 0.15s; font-family:'Space Grotesk',sans-serif; }}
        .qi-btn-save {{ background:rgba(255,99,71,0.15); color:#ff6347; border-color:rgba(255,99,71,0.3); }}
        .qi-btn-save:hover {{ background:rgba(255,99,71,0.25); }}
        .qi-btn-confirm {{ background:#ff6347; color:white; border-color:#ff6347; }}
        .qi-btn-confirm:hover {{ background:#ff4500; }}
        .qi-btn-cancel {{ background:#21262d; color:#8b949e; border-color:#30363d; }}
        .qi-btn-cancel:hover {{ color:#c9d1d9; }}
        .qi-modal {{ position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:9999; display:flex; align-items:center; justify-content:center; }}
        .qi-modal-content {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:24px; width:100%; max-width:400px; }}
        .qi-modal-content h3 {{ font-size:13px; font-weight:700; color:#ff6347; text-transform:uppercase; letter-spacing:0.05em; margin:0 0 16px 0; font-family:'JetBrains Mono',monospace; }}
        .qi-field {{ margin-bottom:12px; }}
        .qi-field label {{ display:block; font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }}
        .qi-field input {{ width:100%; background:#0d1117; border:1px solid #30363d; color:#c9d1d9; padding:8px 12px; border-radius:8px; font-size:13px; box-sizing:border-box; }}
        .qi-field input:focus {{ border-color:#ff6347; outline:none; }}
        .qi-modal-actions {{ display:flex; gap:8px; }}
    </style>
</head>
<body>
    <div class="qi-header">
        <div class="qi-logo">QUICK<span>INSIGHTS</span> — Exploração de Dados</div>
        <div class="qi-info">
            {badge}
            <span style="margin-left:12px;color:#8b949e">PyGWalker · {row_count} registros · {col_count} colunas</span>
        </div>
    </div>
    {toolbar_html}
    {walker_html}
</body>
</html>"""


def generate_chart_html(data: dict) -> str:
    """Chart.js with LLM-recommended config."""
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    config = _ask_llm_for_chart_config(df) or _fallback_chart_config(df)
    return _render_chartjs_html(data, config, "Gráfico", "Gráfico Auto-Configurado")


def generate_typed_chart_html(data: dict, chart_type: str) -> str:
    """Chart.js with user-selected chart type."""
    if chart_type == "auto":
        return generate_chart_html(data)
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    # Use LLM for field selection but override chart_type
    config = _ask_llm_for_chart_config(df) or _fallback_chart_config(df, chart_type)
    config["chart_type"] = chart_type
    return _render_chartjs_html(data, config, "Gráfico", f"Gráfico — {chart_type.title()}")


def generate_gallery_view_html(data: dict, chart_config: dict | None, title: str) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    config = chart_config or _fallback_chart_config(df)
    return _render_chartjs_html(data, config, title, f"Galeria — {title}")


def _empty_html() -> str:
    return """<!DOCTYPE html>
<html><head><title>Quick Insights</title></head>
<body style="background:#0d1117;color:#8b949e;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<div style="text-align:center">
<h2 style="color:#ff6347">Sem dados para visualizar</h2>
<p>Execute uma consulta que retorne resultados tabulares.</p>
</div></body></html>"""
