import json
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Query
from fastapi.responses import StreamingResponse, HTMLResponse
import io

from app.models.schemas import (
    QueryRequest, AnalysisTypeCreate, AnalysisTypeUpdate,
    EmailRequest, ApiKeyCreate, ApiQueryRequest, GallerySaveRequest, PredictionRequest,
)
from app.core.database import get_sync_connection, get_all_tables, execute_readonly_sql
from app.core.security import validate_api_key, create_api_key
from app.core.config import settings
from app.services.excel_service import import_excel
from app.services.agent_service import run_query, reset_agent
from app.services.email_service import send_email_with_excel, export_to_excel_bytes
from app.services.viz_service import generate_explore_html, generate_chart_html, generate_gallery_view_html
from app.services.analytics_service import generate_analytics_html, run_prediction

router = APIRouter(prefix="/api")


# --- Tables ---

@router.get("/tables")
async def list_tables():
    return get_all_tables()


@router.get("/tables/{table_name}/preview")
async def preview_table(table_name: str, limit: int = Query(20, le=100)):
    result = execute_readonly_sql(f'SELECT * FROM "{table_name}" LIMIT {limit}')
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# --- Excel Upload ---

@router.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Apenas arquivos Excel (.xlsx, .xls) são aceitos.")

    dest = settings.upload_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        report = import_excel(dest)
        reset_agent()
        return {"filename": file.filename, "sheets": report}
    except Exception as e:
        raise HTTPException(500, f"Erro ao processar Excel: {str(e)}")


# --- Query (Natural Language via Deep Agent) ---

@router.post("/query")
async def query_nl(req: QueryRequest):
    try:
        result = await run_query(
            question=req.question,
            analysis_type_id=req.analysis_type_id,
            context=req.conversation_context,
            result_limit=req.result_limit,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Erro na consulta: {str(e)}")


# --- Analysis Types ---

@router.get("/analysis-types")
async def list_analysis_types():
    conn = get_sync_connection()
    try:
        cursor = conn.execute("SELECT * FROM analysis_types ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


@router.get("/analysis-types/{type_id}")
async def get_analysis_type(type_id: int):
    conn = get_sync_connection()
    try:
        cursor = conn.execute("SELECT * FROM analysis_types WHERE id = ?", (type_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "Tipo de análise não encontrado.")
        return dict(row)
    finally:
        conn.close()


@router.post("/analysis-types")
async def create_analysis_type(data: AnalysisTypeCreate):
    conn = get_sync_connection()
    try:
        conn.execute(
            "INSERT INTO analysis_types (name, system_prompt, guardrails_input, guardrails_output) VALUES (?, ?, ?, ?)",
            (data.name, data.system_prompt, data.guardrails_input, data.guardrails_output),
        )
        conn.commit()
        return {"success": True}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()


@router.put("/analysis-types/{type_id}")
async def update_analysis_type(type_id: int, data: AnalysisTypeUpdate):
    conn = get_sync_connection()
    try:
        fields, values = [], []
        for field_name in ("name", "system_prompt", "guardrails_input", "guardrails_output"):
            val = getattr(data, field_name)
            if val is not None:
                fields.append(f"{field_name} = ?")
                values.append(val)
        if not fields:
            raise HTTPException(400, "Nenhum campo para atualizar.")
        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(type_id)
        conn.execute(
            f"UPDATE analysis_types SET {', '.join(fields)} WHERE id = ?", values,
        )
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


@router.delete("/analysis-types/{type_id}")
async def delete_analysis_type(type_id: int):
    conn = get_sync_connection()
    try:
        conn.execute("DELETE FROM analysis_types WHERE id = ?", (type_id,))
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


# --- Export Excel ---

@router.post("/export/excel")
async def export_excel(data: dict):
    excel_bytes = export_to_excel_bytes(data)
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=quick_insights_export.xlsx"},
    )


# --- Visualization (PyGWalker) ---

@router.post("/explore", response_class=HTMLResponse)
async def explore_data(data: dict):
    """Open PyGWalker in exploration mode — no pre-configured chart.
    User drags fields to build their own visualizations."""
    try:
        html = generate_explore_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao abrir explorador: {str(e)}")


@router.post("/chart", response_class=HTMLResponse)
async def chart_data(data: dict):
    """Generate PyGWalker with an LLM-recommended chart already rendered.
    The Deep Agent analyzes data shape and auto-configures the best chart."""
    try:
        html = generate_chart_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar gráfico: {str(e)}")


# --- Analytics (Análise Avançada) ---

@router.post("/analytics", response_class=HTMLResponse)
async def analytics_page(data: dict):
    """Generate full Análise Avançada page with descriptive stats + predictive UI."""
    try:
        html = generate_analytics_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar análise: {str(e)}")


@router.post("/analytics/predict")
async def analytics_predict(req: PredictionRequest):
    """Run predictive model (linear, logistic, or clustering)."""
    try:
        result = run_prediction(req.query_data, req.target, req.features, req.model_type, n_clusters=req.n_clusters)
        return result
    except Exception as e:
        return {"error": str(e)}


# --- Email ---

@router.post("/email")
async def send_email(req: EmailRequest):
    result = send_email_with_excel(
        to_email=req.to_email,
        subject=req.subject,
        body_html=req.body_html,
        data=req.excel_data,
    )
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


# --- API Keys ---

@router.post("/keys")
async def create_key(data: ApiKeyCreate):
    return create_api_key(data.label)


@router.get("/keys")
async def list_keys():
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT id, label, is_active, created_at FROM api_keys ORDER BY created_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# --- External API (with API Key auth) ---

@router.post("/v1/query")
async def external_query(req: ApiQueryRequest, x_api_key: str = Header(...)):
    if not validate_api_key(x_api_key):
        raise HTTPException(401, "API key inválida ou inativa.")
    try:
        result = await run_query(
            question=req.question,
            analysis_type_id=req.analysis_type_id,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Erro na consulta: {str(e)}")


# --- Gallery ---

@router.get("/gallery")
async def list_gallery():
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT id, title, description, share_token, created_at FROM analysis_gallery ORDER BY created_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


@router.post("/gallery")
async def save_to_gallery(req: GallerySaveRequest):
    token = uuid.uuid4().hex[:12]
    conn = get_sync_connection()
    try:
        conn.execute(
            "INSERT INTO analysis_gallery (title, description, query_data, chart_config, page_html, share_token) VALUES (?, ?, ?, ?, ?, ?)",
            (
                req.title,
                req.description,
                json.dumps(req.query_data),
                json.dumps(req.local_storage) if req.local_storage else "",
                req.page_html,
                token,
            ),
        )
        conn.commit()
        return {"success": True, "share_token": token}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()


@router.delete("/gallery/{gallery_id}")
async def delete_gallery_item(gallery_id: int):
    conn = get_sync_connection()
    try:
        conn.execute("DELETE FROM analysis_gallery WHERE id = ?", (gallery_id,))
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


@router.get("/gallery/{token}/view", response_class=HTMLResponse)
async def view_gallery_item(token: str):
    """Public shareable view of a gallery analysis — serves the saved PyGWalker page."""
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM analysis_gallery WHERE share_token = ?", (token,)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "Análise não encontrada.")
        item = dict(row)

        page_html = item.get("page_html", "")
        if page_html:
            # Restore localStorage (PyGWalker state) before page scripts run
            ls_data = item.get("chart_config", "")
            if ls_data:
                try:
                    ls_obj = json.loads(ls_data)
                    ls_json = json.dumps(ls_obj)
                    restore_script = f"""<script>(function(){{var d={ls_json};Object.keys(d).forEach(function(k){{try{{localStorage.setItem(k,d[k])}}catch(e){{}}}})}})()</script>"""
                    # Inject before first <script> or before </head>
                    if "<head>" in page_html:
                        page_html = page_html.replace("<head>", "<head>" + restore_script, 1)
                    else:
                        page_html = restore_script + page_html
                except (json.JSONDecodeError, TypeError):
                    pass
            return HTMLResponse(content=page_html)

        # Fallback for old items without page_html: render PyGWalker with data
        query_data = json.loads(item["query_data"])
        chart_config = None
        try:
            chart_config = json.loads(item["chart_config"]) if item["chart_config"] else None
        except (json.JSONDecodeError, TypeError):
            pass
        html = generate_gallery_view_html(query_data, chart_config, item["title"])
        return HTMLResponse(content=html)
    finally:
        conn.close()


# --- Query History ---

@router.get("/history")
async def query_history(limit: int = Query(20, le=100)):
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?", (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
