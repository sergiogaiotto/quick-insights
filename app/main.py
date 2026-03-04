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
    description="Consulte seus dados usando linguagem natural com Deep Agents + OpenAI",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))

app.include_router(api_router)

# Paths that don't require authentication
PUBLIC_PATHS = {"/login", "/api/auth/login", "/api/auth/logout", "/api/auth/check"}
PUBLIC_PREFIXES = ("/static/", "/api/gallery/", "/api/v1/")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    # Allow public paths
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await call_next(request)

    # Check session
    token = request.cookies.get(COOKIE_NAME)
    user = validate_session(token) if token else None

    if user is None:
        # API calls return 401, page requests redirect to login
        if path.startswith("/api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Não autenticado"})
        return RedirectResponse(url="/login", status_code=302)

    # Attach user to request state
    request.state.user = user
    return await call_next(request)


@app.on_event("startup")
async def startup():
    init_metadata_tables()


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # If already authenticated, go to home
    token = request.cookies.get(COOKIE_NAME)
    if token and validate_session(token):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("default.html", {"request": request})
