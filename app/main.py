"""FastAPI app: the public /chat endpoint plus a password-gated /admin/api for
usage stats and conversation browsing. Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import json
import os
import secrets
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from app import db
from app.luther import ask_luther

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Luther Bot")
security = HTTPBasic()


@app.on_event("startup")
def on_startup():
    db.init_db()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    expected_password = os.environ.get("ADMIN_PASSWORD", "")
    is_valid = secrets.compare_digest(credentials.password, expected_password)
    if not expected_password or not is_valid:
        raise HTTPException(status_code=401, detail="Invalid admin credentials",
                             headers={"WWW-Authenticate": "Basic"})
    return credentials.username


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class Source(BaseModel):
    source: str
    year: int
    category: str
    passage_count: int


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list[Source]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page(_admin: str = Depends(require_admin)):
    return FileResponse(STATIC_DIR / "admin.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    db.ensure_session(session_id)

    history = db.get_history(session_id)
    result = ask_luther(req.message, history=history)

    db.log_message(
        session_id=session_id,
        question=req.message,
        response=result["response"],
        retrieved_json=json.dumps(result["retrieved"]),
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cache_creation_input_tokens=result["cache_creation_input_tokens"],
        cache_read_input_tokens=result["cache_read_input_tokens"],
    )

    # Group retrieved chunks by (source, year) -- multiple chunks commonly come from
    # the same work (Table Talk especially, since it's ~45% of the corpus), so we
    # show one citation per work with how many distinct passages fed the answer,
    # rather than a single short excerpt that may only represent one of several
    # chunks and reads as unrelated to other parts of the response. No page numbers
    # are available in these public-domain editions, so a snippet longer than a
    # phrase can't be pinned to a specific claim -- the passage count is more honest.
    counts: dict[tuple[str, int], dict] = {}
    order: list[tuple[str, int]] = []
    for p in result["retrieved"]:
        meta = p["metadata"]
        key = (meta["source"], meta["year"])
        if key not in counts:
            counts[key] = {"category": meta["category"], "count": 0}
            order.append(key)
        counts[key]["count"] += 1

    sources = [
        Source(source=key[0], year=key[1], category=counts[key]["category"], passage_count=counts[key]["count"])
        for key in order
    ]

    return ChatResponse(response=result["response"], session_id=session_id, sources=sources)


@app.get("/admin/api/stats")
def admin_stats(_admin: str = Depends(require_admin)):
    return db.get_stats()


@app.get("/admin/api/messages")
def admin_messages(limit: int = 50, offset: int = 0, _admin: str = Depends(require_admin)):
    return db.list_messages(limit=limit, offset=offset)
