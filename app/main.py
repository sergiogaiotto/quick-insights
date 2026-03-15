from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.config import settings
from app.core.database import init_metadata_tables
from app.core.security import validate_session
from app.api.routes import router as api_router, COOKIE_NAME

app = FastAPI(
    title="Quick Insights",
    description="Consulte seus dados usando linguagem natural e obtenha insights instantâneos. Conecte-se aos bancos de dados, explore seus dados e crie visualizações interativas sem escrever uma única linha de código.",
    version="2.1.0",
)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))

app.include_router(api_router)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Standalone HTML pages (PyGWalker, Chart.js, Analytics) load external CDNs
    # and open in new tabs — do NOT apply restrictive CSP to them.
    path = request.url.path
    standalone_prefixes = ("/api/explore", "/api/chart", "/api/analytics", "/api/gallery/")
    if not any(path.startswith(p) for p in standalone_prefixes):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
    return response


PUBLIC_PATHS = {"/login", "/api/auth/login", "/api/auth/logout", "/api/auth/check"}
PUBLIC_PREFIXES = ("/static/", "/api/gallery/", "/api/v1/")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await call_next(request)

    token = request.cookies.get(COOKIE_NAME)
    user = validate_session(token) if token else None

    if user is None:
        if path.startswith("/api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Não autenticado"})
        return RedirectResponse(url="/login", status_code=302)

    request.state.user = user
    return await call_next(request)


@app.on_event("startup")
async def startup():
    init_metadata_tables()


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token and validate_session(token):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("default.html", {"request": request})
