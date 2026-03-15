import json
import re
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Query, Request, Response, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import io

from app.models.schemas import (
    QueryRequest, AnalysisTypeCreate, AnalysisTypeUpdate,
    EmailRequest, ApiKeyCreate, ApiQueryRequest, GallerySaveRequest, PredictionRequest,
    LoginRequest, UserCreate, UserUpdate, PasswordChange,
    SkillCreate, SkillUpdate,
    DataMartCreate, DataMartUpdate, DataMartAssignTable, ChartRequest,
)
from app.core.database import (
    get_sync_connection, get_all_tables, execute_readonly_sql,
    get_all_skills, get_active_skills, get_skill_by_id,
    create_skill as db_create_skill, update_skill as db_update_skill, delete_skill as db_delete_skill,
    get_all_datamarts, get_datamart_by_id, get_datamart_by_name,
    create_datamart as db_create_datamart, update_datamart as db_update_datamart,
    delete_datamart as db_delete_datamart,
    assign_table_to_datamart, remove_table_from_datamart,
    get_user_datamarts, set_user_datamarts, get_tables_for_datamarts,
    INTERNAL_TABLES,
)
from app.core.security import (
    validate_api_key, create_api_key,
    authenticate_user, create_session, validate_session, destroy_session,
    get_user_count, create_user, list_users, get_user_by_id, update_user,
    change_password, delete_user,
    is_admin, is_root, hash_password,
)
from app.core.config import settings
from app.services.excel_service import import_excel
from app.services.agent_service import run_query, reset_agent
from app.services.email_service import build_eml, export_to_excel_bytes
from app.services.viz_service import (
    generate_explore_html, generate_chart_html, generate_gallery_view_html,
    generate_typed_chart_html, get_chart_options_for_data,
)
from app.services.analytics_service import generate_analytics_html, run_prediction

router = APIRouter(prefix="/api")

COOKIE_NAME = "qi_session"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _sanitize_filename(filename: str) -> str:
    base_name = Path(filename or "").name
    if not base_name or not SAFE_FILENAME.fullmatch(base_name):
        raise HTTPException(400, "Nome de arquivo inválido.")
    return base_name


def _safe_table_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
        raise HTTPException(400, "Nome de tabela inválido")
    return name


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    user = validate_session(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Não autenticado")
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Acesso restrito a administradores")
    return user


async def require_root(user: dict = Depends(get_current_user)) -> dict:
    if not is_root(user):
        raise HTTPException(status_code=403, detail="Acesso restrito ao Root")
    return user


# ---------------------------------------------------------------------------
# Auth routes (public)
# ---------------------------------------------------------------------------

@router.post("/auth/login")
async def login(req: LoginRequest, response: Response):
    user = authenticate_user(req.login, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Login ou senha inválidos")
    token = create_session(user["id"])
    response = JSONResponse(content={
        "success": True,
        "user": {
            "id": user["id"], "login": user["login"],
            "user_type": user["user_type"], "display_name": user["display_name"],
        },
    })
    response.set_cookie(
        key=COOKIE_NAME, value=token, httponly=True,
        samesite="lax", secure=settings.cookie_secure, max_age=86400,
    )
    return response


@router.post("/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        destroy_session(token)
    response = JSONResponse(content={"success": True})
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/auth/me")
async def auth_me(user: dict = Depends(get_current_user)):
    dm_list = get_user_datamarts(user["id"]) if not is_root(user) else get_all_datamarts()
    return {
        "id": user["id"], "login": user["login"],
        "user_type": user["user_type"], "display_name": user["display_name"],
        "profile_description": user.get("profile_description", ""),
        "datamarts": dm_list,
        "is_root": is_root(user),
    }


@router.get("/auth/check")
async def auth_check(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    user = validate_session(token) if token else None
    if user is None:
        return JSONResponse(status_code=401, content={"authenticated": False, "has_users": get_user_count() > 0})
    return {"authenticated": True, "user": {
        "id": user["id"], "login": user["login"],
        "user_type": user["user_type"], "display_name": user["display_name"],
    }}


# ---------------------------------------------------------------------------
# User Management (admin only)
# ---------------------------------------------------------------------------

@router.get("/users")
async def list_users_route(user: dict = Depends(require_admin)):
    return list_users()


@router.post("/users")
async def create_user_route(req: UserCreate, user: dict = Depends(require_admin)):
    # Only root can create root users
    if req.user_type == "root" and not is_root(user):
        raise HTTPException(403, "Apenas Root pode criar usuários Root")
    try:
        new_user = create_user(
            login=req.login, password=req.password, user_type=req.user_type,
            display_name=req.display_name, profile_description=req.profile_description,
            datamart_ids=req.datamart_ids,
        )
        return new_user
    except Exception as e:
        raise HTTPException(400, str(e))


@router.put("/users/{user_id}")
async def update_user_route(user_id: int, req: UserUpdate, user: dict = Depends(require_admin)):
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "Usuário não encontrado")
    if target["user_type"] == "root" and not is_root(user):
        raise HTTPException(403, "Apenas Root pode editar outros Root")
    if req.user_type == "root" and not is_root(user):
        raise HTTPException(403, "Apenas Root pode promover a Root")

    data = req.model_dump(exclude_none=True)
    update_user(user_id, **data)
    return {"success": True}


@router.put("/users/{user_id}/password")
async def change_password_route(user_id: int, req: PasswordChange, user: dict = Depends(require_admin)):
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "Usuário não encontrado")
    if target["user_type"] == "root" and not is_root(user):
        raise HTTPException(403, "Apenas Root pode alterar senha de Root")
    change_password(user_id, req.new_password)
    return {"success": True}


@router.delete("/users/{user_id}")
async def delete_user_route(user_id: int, user: dict = Depends(require_admin)):
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "Usuário não encontrado")
    if target["user_type"] == "root":
        raise HTTPException(403, "Root não pode ser excluído")
    if target["id"] == user["id"]:
        raise HTTPException(400, "Você não pode excluir a si mesmo")
    delete_user(user_id)
    return {"success": True}


# --- User Export/Import ---

@router.get("/users/export")
async def export_users(user: dict = Depends(require_admin)):
    """Export users to Excel."""
    import pandas as pd
    users = list_users()
    dms = get_all_datamarts()
    dm_map = {d["id"]: d["name"] for d in dms}

    rows = []
    for u in users:
        dm_names = ", ".join(dm_map.get(did, str(did)) for did in u.get("datamart_ids", []))
        rows.append({
            "login": u["login"],
            "user_type": u["user_type"],
            "display_name": u.get("display_name", ""),
            "profile_description": u.get("profile_description", ""),
            "is_active": u.get("is_active", 1),
            "datamarts": dm_names,
            "created_at": u.get("created_at", ""),
        })
    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=users_export.xlsx"},
    )


@router.post("/users/import")
async def import_users(file: UploadFile = File(...), user: dict = Depends(require_admin)):
    """Import users from Excel. Default password: minhasenha01, type: admin, datamart: default."""
    import pandas as pd
    data = await file.read()
    df = pd.read_excel(io.BytesIO(data), engine="openpyxl")

    # Get default datamart id
    default_dm = get_datamart_by_name("default")
    default_dm_id = default_dm["id"] if default_dm else None

    created = []
    errors = []
    for _, row in df.iterrows():
        login = str(row.get("login", "")).strip()
        if not login or len(login) < 2:
            errors.append(f"Login vazio ou inválido: '{login}'")
            continue
        display_name = str(row.get("display_name", login)).strip()
        profile_desc = str(row.get("profile_description", "")).strip()
        if profile_desc == "nan":
            profile_desc = ""
        dm_ids = [default_dm_id] if default_dm_id else []
        try:
            create_user(
                login=login, password="minhasenha01", user_type="admin",
                display_name=display_name, profile_description=profile_desc,
                datamart_ids=dm_ids,
            )
            created.append(login)
        except Exception as e:
            errors.append(f"{login}: {str(e)}")

    return {"created": created, "errors": errors, "total": len(created)}


# ---------------------------------------------------------------------------
# DataMart Management
# ---------------------------------------------------------------------------

@router.get("/datamarts")
async def list_datamarts(user: dict = Depends(get_current_user)):
    return get_all_datamarts()


@router.post("/datamarts")
async def create_datamart_route(req: DataMartCreate, user: dict = Depends(require_admin)):
    existing = get_datamart_by_name(req.name)
    if existing:
        raise HTTPException(400, f"DataMart '{req.name}' já existe")
    return db_create_datamart(req.name, req.description)


@router.put("/datamarts/{dm_id}")
async def update_datamart_route(dm_id: int, req: DataMartUpdate, user: dict = Depends(require_admin)):
    dm = get_datamart_by_id(dm_id)
    if not dm:
        raise HTTPException(404, "DataMart não encontrado")
    db_update_datamart(dm_id, **req.model_dump(exclude_none=True))
    return {"success": True}


@router.delete("/datamarts/{dm_id}")
async def delete_datamart_route(dm_id: int, user: dict = Depends(require_admin)):
    dm = get_datamart_by_id(dm_id)
    if not dm:
        raise HTTPException(404, "DataMart não encontrado")
    if dm["name"] == "default":
        raise HTTPException(400, "O DataMart 'default' não pode ser excluído")
    if not db_delete_datamart(dm_id):
        raise HTTPException(400, "Erro ao excluir DataMart")
    return {"success": True}


@router.post("/datamarts/{dm_id}/tables")
async def add_table_to_datamart(dm_id: int, req: DataMartAssignTable, user: dict = Depends(require_admin)):
    dm = get_datamart_by_id(dm_id)
    if not dm:
        raise HTTPException(404, "DataMart não encontrado")
    assign_table_to_datamart(dm_id, req.table_name)
    return {"success": True}


@router.delete("/datamarts/{dm_id}/tables/{table_name}")
async def remove_table_from_dm(dm_id: int, table_name: str, user: dict = Depends(require_admin)):
    remove_table_from_datamart(dm_id, table_name)
    return {"success": True}


@router.get("/datamarts/user")
async def get_my_datamarts(user: dict = Depends(get_current_user)):
    """Return datamarts assigned to current user (root gets all)."""
    if is_root(user):
        return get_all_datamarts()
    return get_user_datamarts(user["id"])


# --- Tables ---

@router.get("/tables")
async def list_tables(request: Request):
    user = getattr(request.state, "user", None)
    all_tables = get_all_tables()
    # Root sees everything
    if user and is_root(user):
        return all_tables
    # Filter by user datamarts
    if user:
        dm_list = get_user_datamarts(user["id"])
        if dm_list:
            allowed = get_tables_for_datamarts([d["id"] for d in dm_list])
            allowed_set = set(allowed)
            return [t for t in all_tables if t["name"] in allowed_set]
    return all_tables


@router.get("/tables/{table_name}/preview")
async def preview_table(table_name: str, limit: int = Query(20, ge=1, le=100)):
    safe_name = _safe_table_name(table_name)
    result = execute_readonly_sql(f'SELECT * FROM "{safe_name}" LIMIT {limit}')
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/tables/{table_name}")
async def drop_table(table_name: str, user: dict = Depends(require_admin)):
    from app.core.database import drop_user_table
    safe_name = _safe_table_name(table_name)
    result = drop_user_table(safe_name)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# --- Excel Upload ---

@router.post("/upload")
async def upload_excel(
    file: UploadFile = File(...),
    datamart_name: str = Query("default", description="Nome do DataMart"),
    user: dict = Depends(require_admin),
):
    filename = _sanitize_filename(file.filename or "")
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Apenas arquivos Excel (.xlsx, .xls) são aceitos.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, "Arquivo excede o tamanho máximo permitido (10MB).")

    dest = settings.upload_dir / filename
    with open(dest, "wb") as f:
        f.write(data)

    # Resolve or create datamart
    dm = get_datamart_by_name(datamart_name)
    if not dm:
        dm = db_create_datamart(datamart_name)
    dm_id = dm["id"]

    try:
        report = import_excel(dest)
        # Assign created tables to the datamart
        for sheet_info in report:
            if sheet_info.get("table"):
                assign_table_to_datamart(dm_id, sheet_info["table"])
        reset_agent()
        return {"filename": filename, "sheets": report, "datamart": dm["name"]}
    except Exception:
        raise HTTPException(500, "Erro ao processar Excel.")


# --- Query (Natural Language via Deep Agent) ---

@router.post("/query")
async def query_nl(req: QueryRequest, request: Request):
    user = getattr(request.state, "user", None)
    user_login = user["login"] if user else ""

    # Determine accessible tables
    accessible_tables = None
    if user and not is_root(user):
        if req.datamart_ids:
            # Use requested datamarts (intersected with user's)
            user_dms = get_user_datamarts(user["id"])
            user_dm_ids = {d["id"] for d in user_dms}
            allowed_ids = [did for did in req.datamart_ids if did in user_dm_ids]
            if allowed_ids:
                accessible_tables = get_tables_for_datamarts(allowed_ids)
        else:
            # Use all user datamarts
            user_dms = get_user_datamarts(user["id"])
            if user_dms:
                accessible_tables = get_tables_for_datamarts([d["id"] for d in user_dms])
    elif user and is_root(user) and req.datamart_ids:
        accessible_tables = get_tables_for_datamarts(req.datamart_ids)

    try:
        result = await run_query(
            question=req.question,
            analysis_type_id=req.analysis_type_id,
            context=req.conversation_context,
            result_limit=req.result_limit,
            user_login=user_login,
            accessible_tables=accessible_tables,
        )
        return result
    except Exception:
        raise HTTPException(500, "Erro na consulta.")


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
async def create_analysis_type(data: AnalysisTypeCreate, user: dict = Depends(require_admin)):
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
async def update_analysis_type(type_id: int, data: AnalysisTypeUpdate, user: dict = Depends(require_admin)):
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
async def delete_analysis_type(type_id: int, user: dict = Depends(require_admin)):
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
    try:
        html = generate_explore_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao abrir explorador: {str(e)}")


@router.post("/chart", response_class=HTMLResponse)
async def chart_data(data: dict):
    """Generate Chart.js with LLM-recommended chart (auto mode)."""
    try:
        html = generate_chart_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar gráfico: {str(e)}")


@router.post("/chart/typed", response_class=HTMLResponse)
async def chart_data_typed(req: ChartRequest):
    """Generate Chart.js with user-selected chart type."""
    try:
        html = generate_typed_chart_html(req.query_data, req.chart_type)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar gráfico: {str(e)}")


@router.post("/chart/options")
async def chart_options(data: dict):
    """Return available chart types for the given data."""
    try:
        return get_chart_options_for_data(data)
    except Exception as e:
        raise HTTPException(500, f"Erro ao analisar opções: {str(e)}")


# --- Analytics (Análise Avançada) ---

@router.post("/analytics", response_class=HTMLResponse)
async def analytics_page(data: dict):
    try:
        html = generate_analytics_html(data)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar análise: {str(e)}")


@router.post("/analytics/predict")
async def analytics_predict(req: PredictionRequest):
    try:
        result = run_prediction(req.query_data, req.target, req.features, req.model_type, n_clusters=req.n_clusters)
        return result
    except Exception as e:
        return {"error": str(e)}


# --- Email ---

@router.post("/email")
async def send_email(req: EmailRequest):
    try:
        eml_bytes = build_eml(
            to_email=req.to_email, subject=req.subject,
            body_html=req.body_html, data=req.excel_data,
        )
        return StreamingResponse(
            io.BytesIO(eml_bytes),
            media_type="message/rfc822",
            headers={"Content-Disposition": f'attachment; filename="{req.subject}.eml"'},
        )
    except Exception as e:
        raise HTTPException(500, f"Erro ao gerar email: {str(e)}")


# --- API Keys ---

@router.post("/keys")
async def create_key(data: ApiKeyCreate, user: dict = Depends(require_admin)):
    return create_api_key(data.label)


@router.get("/keys")
async def list_keys(user: dict = Depends(require_admin)):
    conn = get_sync_connection()
    try:
        cursor = conn.execute(
            "SELECT id, label, is_active, created_at FROM api_keys ORDER BY created_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# --- Custom Skills ---

@router.get("/skills")
async def list_skills():
    return get_all_skills()


@router.get("/skills/active")
async def list_active_skills():
    return get_active_skills()


@router.post("/skills")
async def create_skill_route(req: SkillCreate, request: Request, user: dict = Depends(require_admin)):
    current_user = getattr(request.state, "user", None)
    created_by = current_user["login"] if current_user else ""
    try:
        return db_create_skill(
            name=req.name, description=req.description,
            content=req.content, created_by=created_by,
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/skills/{skill_id}")
async def get_skill_route(skill_id: int):
    skill = get_skill_by_id(skill_id)
    if not skill:
        raise HTTPException(404, "Skill não encontrada")
    return skill


@router.put("/skills/{skill_id}")
async def update_skill_route(skill_id: int, req: SkillUpdate, user: dict = Depends(require_admin)):
    skill = get_skill_by_id(skill_id)
    if not skill:
        raise HTTPException(404, "Skill não encontrada")
    db_update_skill(skill_id, **req.model_dump(exclude_none=True))
    return {"success": True}


@router.put("/skills/{skill_id}/toggle")
async def toggle_skill(skill_id: int, user: dict = Depends(require_admin)):
    skill = get_skill_by_id(skill_id)
    if not skill:
        raise HTTPException(404, "Skill não encontrada")
    new_state = 0 if skill["is_active"] else 1
    db_update_skill(skill_id, is_active=new_state)
    return {"success": True, "is_active": new_state}


@router.delete("/skills/{skill_id}")
async def delete_skill_route(skill_id: int, user: dict = Depends(require_admin)):
    skill = get_skill_by_id(skill_id)
    if not skill:
        raise HTTPException(404, "Skill não encontrada")
    db_delete_skill(skill_id)
    return {"success": True}


# --- Skills Export/Import ---

@router.get("/skills/export/excel")
async def export_skills(user: dict = Depends(require_admin)):
    """Export all skills to Excel."""
    import pandas as pd
    skills = get_all_skills()
    rows = []
    for s in skills:
        rows.append({
            "name": s["name"],
            "description": s.get("description", ""),
            "content": s.get("content", ""),
            "is_active": s.get("is_active", 1),
            "created_by": s.get("created_by", ""),
            "created_at": s.get("created_at", ""),
        })
    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=skills_export.xlsx"},
    )


@router.post("/skills/import")
async def import_skills(file: UploadFile = File(...), request: Request = None, user: dict = Depends(require_admin)):
    """Import skills from Excel."""
    import pandas as pd
    data = await file.read()
    df = pd.read_excel(io.BytesIO(data), engine="openpyxl")

    current_user = getattr(request.state, "user", None) if request else None
    created_by = current_user["login"] if current_user else ""

    created = []
    errors = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or len(name) < 2:
            errors.append(f"Nome vazio ou inválido: '{name}'")
            continue
        description = str(row.get("description", "")).strip()
        content = str(row.get("content", "")).strip()
        if description == "nan":
            description = ""
        if content == "nan":
            content = ""
        try:
            db_create_skill(name=name, description=description, content=content, created_by=created_by)
            created.append(name)
        except Exception as e:
            errors.append(f"{name}: {str(e)}")

    return {"created": created, "errors": errors, "total": len(created)}


# --- External API (with API Key auth) ---

@router.post("/v1/query")
async def external_query(req: ApiQueryRequest, x_api_key: str = Header(...)):
    if not validate_api_key(x_api_key):
        raise HTTPException(401, "API key inválida ou inativa.")
    try:
        result = await run_query(
            question=req.question,
            analysis_type_id=req.analysis_type_id,
            user_login=f"api-key:{x_api_key[:8]}",
        )
        return result
    except Exception:
        raise HTTPException(500, "Erro na consulta.")


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
async def save_to_gallery(req: GallerySaveRequest, user: dict = Depends(get_current_user)):
    token = uuid.uuid4().hex[:12]
    conn = get_sync_connection()
    try:
        conn.execute(
            "INSERT INTO analysis_gallery (title, description, query_data, chart_config, page_html, share_token) VALUES (?, ?, ?, ?, ?, ?)",
            (
                req.title, req.description,
                json.dumps(req.query_data),
                json.dumps(req.local_storage) if req.local_storage else "",
                req.page_html, token,
            ),
        )
        conn.commit()
        return {"success": True, "share_token": token}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()


@router.delete("/gallery/{gallery_id}")
async def delete_gallery_item(gallery_id: int, user: dict = Depends(require_admin)):
    conn = get_sync_connection()
    try:
        conn.execute("DELETE FROM analysis_gallery WHERE id = ?", (gallery_id,))
        conn.commit()
        return {"success": True}
    finally:
        conn.close()


@router.get("/gallery/{token}/view", response_class=HTMLResponse)
async def view_gallery_item(token: str):
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
            ls_data = item.get("chart_config", "")
            if ls_data:
                try:
                    ls_obj = json.loads(ls_data)
                    ls_json = json.dumps(ls_obj)
                    restore_script = f"""<script>(function(){{var d={ls_json};Object.keys(d).forEach(function(k){{try{{localStorage.setItem(k,d[k])}}catch(e){{}}}})}})()</script>"""
                    if "<head>" in page_html:
                        page_html = page_html.replace("<head>", "<head>" + restore_script, 1)
                    else:
                        page_html = restore_script + page_html
                except (json.JSONDecodeError, TypeError):
                    pass
            return HTMLResponse(content=page_html)

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
