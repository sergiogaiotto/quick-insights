import io
import pandas as pd
from exchangelib import (
    Credentials, Account, Message, Mailbox,
    FileAttachment, DELEGATE, Configuration,
)
from app.core.config import settings


def send_email_with_excel(
    to_email: str,
    subject: str,
    body_html: str,
    data: dict | None = None,
) -> dict:
    if not settings.email_address or not settings.email_password:
        return {"error": "Credenciais de email não configuradas no .env"}

    try:
        credentials = Credentials(settings.email_address, settings.email_password)
        config = Configuration(server=settings.email_server, credentials=credentials)
        account = Account(
            settings.email_address,
            credentials=credentials,
            config=config,
            autodiscover=False,
            access_type=DELEGATE,
        )

        intro = (
            "<p>Prezado(a),</p>"
            "<p>Segue em anexo os dados solicitados via Quick Insights.</p>"
        )
        full_body = f"{intro}{body_html}"

        msg = Message(
            account=account,
            subject=subject,
            body=full_body,
            to_recipients=[Mailbox(email_address=to_email)],
        )

        if data and "rows" in data and data["rows"]:
            df = pd.DataFrame(data["rows"])
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            attachment = FileAttachment(
                name=f"{subject}.xlsx",
                content=buffer.read(),
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            msg.attach(attachment)

        msg.send()
        return {"success": True, "message": f"Email enviado para {to_email}"}

    except Exception as e:
        return {"error": f"Erro ao enviar email: {str(e)}"}


def export_to_excel_bytes(data: dict) -> bytes:
    if not data or "rows" not in data or not data["rows"]:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data["rows"])

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer.read()
