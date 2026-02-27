from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.core.database import init_metadata_tables
from app.api.routes import router as api_router

app = FastAPI(
    title="Quick Insights",
    description="Consulte seus dados usando linguagem natural com Deep Agents + OpenAI",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
templates = Jinja2Templates(directory=str(settings.templates_dir))

app.include_router(api_router)


@app.on_event("startup")
async def startup():
    init_metadata_tables()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("default.html", {"request": request})
