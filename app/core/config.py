import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'quick_insights.db'}"
    )
    db_path: Path = BASE_DIR / "data" / "quick_insights.db"

    # API Security
    api_salt: str = os.getenv("API_SALT", "default-salt")
    api_secret_key: str = os.getenv("API_SECRET_KEY", "default-secret")

    # Email
    email_address: str = os.getenv("EMAIL_ADDRESS", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")
    email_server: str = os.getenv("EMAIL_SERVER", "outlook.office365.com")

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Paths
    project_dir: Path = BASE_DIR
    upload_dir: Path = BASE_DIR / "uploads"
    templates_dir: Path = BASE_DIR / "app" / "templates"
    static_dir: Path = BASE_DIR / "app" / "static"
    agents_md: Path = BASE_DIR / "AGENTS.md"
    skills_dir: Path = BASE_DIR / "skills"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
