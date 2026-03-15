"""
Quick Insights — Visualization Service

- Explorar: PyGWalker (drag-and-drop livre)
- Gráfico: Chart.js interativo com seletores de campo X, Y, agregação e tipo
- Galeria: Chart.js com config salva
"""

import json
import pandas as pd
import pygwalker as pyg
from langchain_openai import ChatOpenAI
from app.core.config import settings


# ---------------------------------------------------------------------------
# LLM chart recommendation
# ---------------------------------------------------------------------------

_SPEC_PROMPT = """You are a data visualization expert. Analyze the data below and recommend the best chart.

## Data
Columns: {columns}
Types: {dtypes}
Sample (first 5 rows):
{sample}

## Rules
1. Identify which columns are metrics (numeric) and which are dimensions (categorical/text)
2. Choose the most appropriate chart type:
   - 1 dimension + 1 metric -> bar
   - temporal dimension + metric -> line
   - 2 metrics -> scatter
   - only metrics -> bar with first column as axis
3. Return ONLY valid JSON, no markdown, no explanation

## Output format (pure JSON)
{{
  "chart_type": "bar|line|scatter|area|pie|doughnut",
  "x_field": "column_name_for_x_axis",
  "y_field": "column_name_for_y_axis",
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
        "agg": "sum",
    }


# ---------------------------------------------------------------------------
# Chart options (for submenu)
# ---------------------------------------------------------------------------

def get_chart_options_for_data(data: dict) -> dict:
    df = _data_to_df(data)
    if df is None:
        return {"options": []}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    n_rows = len(df)
    options = [
        {"type": "auto", "label": "Auto (LLM)", "icon": "\u2728", "suitable": True},
        {"type": "bar", "label": "Barras", "icon": "\uD83D\uDCCA", "suitable": True},
        {"type": "line", "label": "Linhas", "icon": "\uD83D\uDCC8", "suitable": len(numeric) > 0 and n_rows > 2},
        {"type": "scatter", "label": "Dispersão", "icon": "\u26A1", "suitable": len(numeric) >= 2},
        {"type": "area", "label": "Área", "icon": "\uD83C\uDFD4\uFE0F", "suitable": len(numeric) > 0 and n_rows > 2},
        {"type": "pie", "label": "Pizza", "icon": "\uD83E\uDD67", "suitable": len(numeric) > 0 and n_rows <= 20},
        {"type": "doughnut", "label": "Rosca", "icon": "\uD83C\uDF69", "suitable": len(numeric) > 0 and n_rows <= 20},
        {"type": "radar", "label": "Radar", "icon": "\uD83D\uDD78\uFE0F", "suitable": len(numeric) >= 3 and n_rows <= 15},
        {"type": "polarArea", "label": "Polar", "icon": "\uD83C\uDFAF", "suitable": len(numeric) > 0 and n_rows <= 12},
    ]
    return {"options": options, "numeric_cols": numeric, "categorical_cols": non_numeric, "row_count": n_rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_to_df(data: dict) -> pd.DataFrame | None:
    rows = data.get("rows", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Interactive Chart Page
# ---------------------------------------------------------------------------

def _render_interactive_chart_html(data: dict, initial_config: dict | None = None) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    all_cols = list(df.columns)

    if not initial_config:
        initial_config = _fallback_chart_config(df)

    if initial_config.get("x_field") not in all_cols:
        initial_config["x_field"] = all_cols[0]
    if initial_config.get("y_field") not in all_cols:
        initial_config["y_field"] = numeric_cols[0] if numeric_cols else all_cols[-1]

    data_json = json.dumps(data.get("rows", []), default=str, ensure_ascii=False)
    cols_json = json.dumps(all_cols)
    num_cols_json = json.dumps(numeric_cols)
    config_json = json.dumps(initial_config)
    row_count = len(data.get("rows", []))

    x_options = "".join(f'<option value="{c}">{c}</option>' for c in all_cols)
    y_options = "".join(f'<option value="{c}">{c}</option>' for c in all_cols)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quick Insights — Gráfico</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{background:#0d1117;color:#c9d1d9;font-family:'Space Grotesk',sans-serif;height:100vh;display:flex;flex-direction:column}}
        .qi-hdr{{background:#161b22;border-bottom:1px solid #30363d;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
        .qi-logo{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600}}
        .qi-logo span{{color:#ff6347}}
        .qi-bar{{background:#161b22;border-bottom:1px solid #30363d;padding:10px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;flex-shrink:0}}
        .qi-g{{display:flex;align-items:center;gap:6px}}
        .qi-lbl{{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;font-family:'JetBrains Mono',monospace;white-space:nowrap}}
        .qi-sel{{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:5px 10px;border-radius:6px;font-size:12px;font-family:'Space Grotesk',sans-serif;cursor:pointer}}
        .qi-sel:focus{{border-color:#ff6347;outline:none}}
        .qi-sel option{{background:#0d1117}}
        .qi-wrap{{flex:1;padding:20px;display:flex;align-items:center;justify-content:center;min-height:0}}
        .qi-inner{{width:100%;max-width:1200px;height:100%;position:relative}}
        .qi-nfo{{font-size:11px;color:#8b949e;font-family:'JetBrains Mono',monospace}}
    </style>
</head>
<body>
    <div class="qi-hdr">
        <div class="qi-logo">QUICK<span>INSIGHTS</span> — Gráfico Interativo</div>
        <div class="qi-nfo">{row_count} registros · {len(all_cols)} colunas</div>
    </div>
    <div class="qi-bar">
        <div class="qi-g"><span class="qi-lbl">Tipo</span>
            <select id="ctrlType" class="qi-sel" onchange="rebuild()">
                <option value="bar">Barras</option><option value="line">Linhas</option>
                <option value="scatter">Dispersão</option><option value="area">Área</option>
                <option value="pie">Pizza</option><option value="doughnut">Rosca</option>
                <option value="radar">Radar</option><option value="polarArea">Polar</option>
            </select>
        </div>
        <div class="qi-g"><span class="qi-lbl">Eixo X</span>
            <select id="ctrlX" class="qi-sel" onchange="rebuild()">{x_options}</select>
        </div>
        <div class="qi-g"><span class="qi-lbl">Eixo Y</span>
            <select id="ctrlY" class="qi-sel" onchange="rebuild()">{y_options}</select>
        </div>
        <div class="qi-g"><span class="qi-lbl">Agregação</span>
            <select id="ctrlAgg" class="qi-sel" onchange="rebuild()">
                <option value="sum">Soma</option><option value="mean">Média</option>
                <option value="count">Contagem</option><option value="none">Nenhuma</option>
            </select>
        </div>
        <div class="qi-g"><span class="qi-lbl">Limite</span>
            <select id="ctrlLimit" class="qi-sel" onchange="rebuild()">
                <option value="20">20</option><option value="50" selected>50</option>
                <option value="100">100</option><option value="0">Todos</option>
            </select>
        </div>
        <div class="qi-g"><span class="qi-lbl">Ordem</span>
            <select id="ctrlSort" class="qi-sel" onchange="rebuild()">
                <option value="asc" selected>Ascendente</option>
                <option value="desc">Descendente</option>
            </select>
        </div>
    </div>
    <div class="qi-wrap"><div class="qi-inner"><canvas id="mainChart"></canvas></div></div>

<script>
const RAW = {data_json};
const COLS = {cols_json};
const NUM = new Set({num_cols_json});
const INI = {config_json};
const PAL = ['rgba(255,99,71,.7)','rgba(88,166,255,.7)','rgba(57,211,83,.7)','rgba(240,136,62,.7)','rgba(163,113,247,.7)','rgba(63,185,80,.7)','rgba(210,168,255,.7)','rgba(121,192,255,.7)','rgba(255,166,87,.7)','rgba(255,123,114,.7)'];

Chart.defaults.color='#8b949e';
Chart.defaults.borderColor='#21262d';
Chart.defaults.font.family="'Space Grotesk',sans-serif";

let chart=null;

function agg(xF,yF,fn,lim,sort){{
    if(fn==='none'){{
        let r=RAW.map(d=>({{x:String(d[xF]??''),y:Number(d[yF])||0}}));
        // Sort by label (X)
        r.sort((a,b)=>{{
            const cmp=a.x.localeCompare(b.x,undefined,{{numeric:true,sensitivity:'base'}});
            return sort==='desc'?-cmp:cmp;
        }});
        if(lim>0)r=r.slice(0,lim);
        return{{l:r.map(d=>d.x),v:r.map(d=>d.y)}};
    }}
    const g={{}};
    RAW.forEach(d=>{{
        const k=String(d[xF]??'');
        if(!g[k])g[k]=[];
        const n=Number(d[yF]);
        if(!isNaN(n))g[k].push(n);
    }});
    let e=Object.entries(g).map(([k,vs])=>{{
        let val=0;
        if(fn==='sum')val=vs.reduce((a,b)=>a+b,0);
        else if(fn==='mean')val=vs.length?vs.reduce((a,b)=>a+b,0)/vs.length:0;
        else if(fn==='count')val=vs.length;
        return{{l:k,v:val}};
    }});
    e.sort((a,b)=>sort==='desc'?b.v-a.v:a.v-b.v);
    if(lim>0)e=e.slice(0,lim);
    return{{l:e.map(d=>d.l),v:e.map(d=>d.v)}};
}}

function rebuild(){{
    const tp=document.getElementById('ctrlType').value;
    const xF=document.getElementById('ctrlX').value;
    const yF=document.getElementById('ctrlY').value;
    const ag=document.getElementById('ctrlAgg').value;
    const lm=parseInt(document.getElementById('ctrlLimit').value)||0;
    const sort=document.getElementById('ctrlSort').value;
    const d=agg(xF,yF,ag,lm,sort);
    if(chart)chart.destroy();

    const mp={{bar:'bar',line:'line',scatter:'scatter',area:'line',pie:'pie',doughnut:'doughnut',radar:'radar',polarArea:'polarArea'}};
    const ct=mp[tp]||'bar';
    const circ=['pie','doughnut','polarArea'].includes(tp);
    const rad=tp==='radar';
    const fill=tp==='area';

    let ds;
    if(ct==='scatter'){{
        ds={{label:yF,data:d.l.map((x,i)=>({{x,y:d.v[i]}})),backgroundColor:'rgba(255,99,71,.7)',borderColor:'#ff6347',pointRadius:5,pointHoverRadius:7}};
    }}else if(circ){{
        ds={{label:yF,data:d.v,backgroundColor:d.l.map((_,i)=>PAL[i%PAL.length]),borderColor:'#0d1117',borderWidth:2}};
    }}else if(rad){{
        ds={{label:yF,data:d.v,backgroundColor:'rgba(255,99,71,.2)',borderColor:'#ff6347',borderWidth:2,pointBackgroundColor:'#ff6347'}};
    }}else{{
        ds={{label:yF,data:d.v,backgroundColor:'rgba(255,99,71,.35)',borderColor:'#ff6347',borderWidth:2,fill:fill,tension:.3,borderRadius:ct==='bar'?4:0}};
    }}

    const sc={{}};
    if(!circ&&!rad){{
        sc.x={{ticks:{{color:'#8b949e',maxRotation:45,font:{{size:11}}}},grid:{{color:'#21262d'}},title:{{display:true,text:xF,color:'#c9d1d9',font:{{size:12,weight:600}}}}}};
        sc.y={{ticks:{{color:'#8b949e',font:{{size:11}}}},grid:{{color:'#21262d'}},title:{{display:true,text:yF+(ag!=='none'?' ('+ag+')':''),color:'#c9d1d9',font:{{size:12,weight:600}}}},beginAtZero:true}};
    }}

    chart=new Chart(document.getElementById('mainChart'),{{
        type:ct,
        data:{{labels:d.l,datasets:[ds]}},
        options:{{
            responsive:true,maintainAspectRatio:false,
            plugins:{{legend:{{display:true,labels:{{color:'#c9d1d9',font:{{size:12}}}}}},tooltip:{{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,titleColor:'#ff6347',bodyColor:'#c9d1d9',padding:10,cornerRadius:8}}}},
            scales:sc,
            animation:{{duration:400,easing:'easeOutQuart'}},
        }},
    }});
}}

// Init
document.getElementById('ctrlType').value=INI.chart_type||'bar';
document.getElementById('ctrlX').value=INI.x_field||COLS[0];
document.getElementById('ctrlY').value=INI.y_field||(COLS.length>1?COLS[1]:COLS[0]);
document.getElementById('ctrlAgg').value=INI.agg||'sum';
rebuild();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explore_html(data: dict) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    try:
        walker_html = pyg.to_html(df, appearance="dark", default_tab="data")
    except Exception:
        walker_html = pyg.to_html(df, appearance="dark")
    data_json = json.dumps(data)
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
        body{{margin:0;padding:0;background:#0d1117;color:#c9d1d9;font-family:'Space Grotesk',sans-serif}}
        .qi-hdr{{background:#161b22;border-bottom:1px solid #30363d;padding:10px 20px;display:flex;align-items:center;justify-content:space-between}}
        .qi-logo{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600}}
        .qi-logo span{{color:#ff6347}}
        .qi-tb{{background:#161b22;border-bottom:1px solid #30363d;padding:8px 20px;display:flex;align-items:center;justify-content:flex-end}}
        .qi-btn{{padding:5px 14px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid;transition:all .15s;font-family:'Space Grotesk',sans-serif}}
        .qi-btn-s{{background:rgba(255,99,71,.15);color:#ff6347;border-color:rgba(255,99,71,.3)}}
        .qi-btn-s:hover{{background:rgba(255,99,71,.25)}}
        .qi-btn-c{{background:#ff6347;color:#fff;border-color:#ff6347}}
        .qi-btn-c:hover{{background:#ff4500}}
        .qi-btn-x{{background:#21262d;color:#8b949e;border-color:#30363d}}
        .qi-btn-x:hover{{color:#c9d1d9}}
        .qi-m{{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:9999;display:flex;align-items:center;justify-content:center}}
        .qi-mc{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;width:100%;max-width:400px}}
        .qi-mc h3{{font-size:13px;font-weight:700;color:#ff6347;text-transform:uppercase;letter-spacing:.05em;margin:0 0 16px 0;font-family:'JetBrains Mono',monospace}}
        .qi-f{{margin-bottom:12px}}
        .qi-f label{{display:block;font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
        .qi-f input{{width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:8px 12px;border-radius:8px;font-size:13px;box-sizing:border-box}}
        .qi-f input:focus{{border-color:#ff6347;outline:none}}
        .qi-ma{{display:flex;gap:8px}}
    </style>
</head>
<body>
    <div class="qi-hdr">
        <div class="qi-logo">QUICK<span>INSIGHTS</span> — Exploração de Dados</div>
        <div style="font-size:11px;color:#8b949e">PyGWalker · {row_count} registros · {col_count} colunas</div>
    </div>
    <div class="qi-tb">
        <button onclick="saveToGallery()" class="qi-btn qi-btn-s">Salvar na Galeria</button>
    </div>
    <div id="saveModal" class="qi-m" style="display:none">
        <div class="qi-mc">
            <h3>Salvar na Galeria</h3>
            <div class="qi-f"><label>Título</label><input type="text" id="saveTitle" placeholder="Nome da análise"></div>
            <div class="qi-f"><label>Descrição</label><input type="text" id="saveDesc" placeholder="Descrição opcional"></div>
            <div class="qi-ma">
                <button onclick="confirmSave()" class="qi-btn qi-btn-c">Salvar</button>
                <button onclick="closeSaveModal()" class="qi-btn qi-btn-x">Cancelar</button>
            </div>
            <div id="saveStatus" style="margin-top:8px;font-size:11px"></div>
        </div>
    </div>
    {walker_html}
    <script>
        const _qd={data_json};
        function saveToGallery(){{document.getElementById('saveModal').style.display='flex';document.getElementById('saveTitle').focus()}}
        function closeSaveModal(){{document.getElementById('saveModal').style.display='none';document.getElementById('saveStatus').textContent=''}}
        async function confirmSave(){{
            const t=document.getElementById('saveTitle').value.trim();
            if(!t){{alert('Informe um título.');return}}
            const d=document.getElementById('saveDesc').value.trim();
            const s=document.getElementById('saveStatus');
            s.style.color='#8b949e';s.textContent='Capturando...';
            const tb=document.querySelector('.qi-tb'),md=document.getElementById('saveModal');
            const td=tb?tb.style.display:'';
            if(tb)tb.style.display='none';if(md)md.style.display='none';
            const ph='<!DOCTYPE html>'+document.documentElement.outerHTML;
            if(tb)tb.style.display=td;if(md)md.style.display='flex';
            const ls={{}};for(let i=0;i<localStorage.length;i++){{const k=localStorage.key(i);ls[k]=localStorage.getItem(k)}}
            s.textContent='Salvando...';
            try{{
                const r=await fetch('/api/gallery',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{title:t,description:d,query_data:_qd,page_html:ph,local_storage:ls}})}});
                if(r.ok){{s.style.color='#39d353';s.textContent='Salvo.';setTimeout(closeSaveModal,1500)}}
                else{{s.style.color='#ff6347';s.textContent='Erro.'}}
            }}catch(e){{s.style.color='#ff6347';s.textContent='Erro: '+e.message}}
        }}
    </script>
</body>
</html>"""


def generate_chart_html(data: dict) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    config = _ask_llm_for_chart_config(df) or _fallback_chart_config(df)
    return _render_interactive_chart_html(data, config)


def generate_typed_chart_html(data: dict, chart_type: str) -> str:
    if chart_type == "auto":
        return generate_chart_html(data)
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    config = _ask_llm_for_chart_config(df) or _fallback_chart_config(df, chart_type)
    config["chart_type"] = chart_type
    return _render_interactive_chart_html(data, config)


def generate_gallery_view_html(data: dict, chart_config: dict | None, title: str) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()
    config = chart_config or _fallback_chart_config(df)
    return _render_interactive_chart_html(data, config)


def _empty_html() -> str:
    return """<!DOCTYPE html>
<html><head><title>Quick Insights</title></head>
<body style="background:#0d1117;color:#8b949e;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<div style="text-align:center">
<h2 style="color:#ff6347">Sem dados para visualizar</h2>
<p>Execute uma consulta que retorne resultados tabulares.</p>
</div></body></html>"""
