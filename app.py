import os, io, hashlib, requests
from typing import List, Optional

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import psycopg2

from pypdf import PdfReader
import docx as docx_reader

load_dotenv()

APP = FastAPI(title="RAG Upload Admin (FastAPI)")

# ===== 환경설정 =====
SUPABASE_DB_CONN = os.getenv("SUPABASE_DB_CONN")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ===== 유틸 =====
def extract_text_from_file(file_bytes: bytes, filename: str, mime: Optional[str]) -> str:
    name_lower = (filename or "").lower()
    mime = (mime or "").lower()
    if name_lower.endswith(".pdf") or "pdf" in mime:
        return extract_pdf_text(file_bytes)
    if name_lower.endswith(".docx") or "word" in mime or "officedocument" in mime:
        return extract_docx_text(file_bytes)
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except:
        return ""

def extract_pdf_text(file_bytes: bytes) -> str:
    text = []
    with io.BytesIO(file_bytes) as buf:
        reader = PdfReader(buf)
        for page in reader.pages:
            t = page.extract_text() or ""
            text.append(t)
    return "\n".join(text).strip()

def extract_docx_text(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as buf:
        doc = docx_reader.Document(buf)
        return "\n".join(p.text for p in doc.paragraphs).strip()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(start + size, N)
        chunks.append(text[start:end])
        if end == N:
            break
        start = max(0, end - overlap)
    return [c.strip() for c in chunks if c.strip()]

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ollama_embed(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=120
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out

def upsert_to_db(title: str, filename: str, mime: Optional[str], bytes_len: int,
                 whole_text: str, chunks: List[str], embeds: List[List[float]]) -> dict:
    conn = psycopg2.connect(SUPABASE_DB_CONN)
    conn.autocommit = True
    cur = conn.cursor()

    doc_hash = sha256_text(whole_text)
    cur.execute("""
        insert into documents (title, source_path, mime_type, bytes, hash)
        values (%s, %s, %s, %s, %s)
        on conflict (hash) do update set updated_at = now()
        returning id;
    """, (title, filename, mime, bytes_len, doc_hash))
    document_id = cur.fetchone()[0]

    cur.execute("delete from chunks where document_id = %s", (document_id,))
    for idx, (c, e) in enumerate(zip(chunks, embeds)):
        vec = f"[{','.join(str(x) for x in e)}]"
        cur.execute("""
            insert into chunks (document_id, chunk_index, content, embedding)
            values (%s, %s, %s, %s::vector)
        """, (document_id, idx, c, vec))

    cur.close()
    conn.close()
    return {"document_id": str(document_id), "chunks": len(chunks)}

# ===== 뷰 (1페이지) =====
FORM_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>RAG 업로드 어드민</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; max-width: 720px; margin: 40px auto; padding: 0 16px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; }
    h1 { font-size: 20px; margin: 0 0 12px; }
    label { display:block; margin:12px 0 6px; font-weight:600; }
    input[type="text"], input[type="file"] { width:100%; padding:10px; border:1px solid #d1d5db; border-radius:8px; }
    button { margin-top:16px; padding:10px 16px; border-radius:10px; border:1px solid #111827; background:#111827; color:white; cursor:pointer; }
    .msg { margin-top:16px; white-space:pre-wrap; font-family:ui-monospace,Menlo,Consolas,monospace; }
    .hint { color:#6b7280; font-size:13px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>RAG 업로드 어드민</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <label>문서 제목</label>
      <input type="text" name="title" placeholder="(선택) 미입력 시 파일명 사용" />
      <label>파일 선택</label>
      <input type="file" name="file" required />
      <div class="hint">PDF / DOCX / TXT 권장</div>
      <button type="submit">업로드 → 인덱싱 → DB 저장</button>
    </form>
    <div class="msg">업로드 후 결과가 이 페이지로 표시됩니다.</div>
  </div>
</body>
</html>
"""

@APP.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(FORM_HTML)

@APP.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile, title: Optional[str] = Form(default=None)):
    file_bytes = await file.read()
    mime = file.content_type or "application/octet-stream"
    name = file.filename or "uploaded_file"
    title = title.strip() if title else name

    # 1) 텍스트 추출
    text = extract_text_from_file(file_bytes, name, mime)
    if not text.strip():
        return HTMLResponse(f"<pre>텍스트 추출 실패: {name} ({mime})</pre>", status_code=400)

    # 2) 청크
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return HTMLResponse(f"<pre>청크 생성 실패(빈 텍스트)</pre>", status_code=400)

    # 3) 임베딩 (Ollama)
    try:
        embeds = ollama_embed(chunks)
    except Exception as e:
        return HTMLResponse(f"<pre>임베딩 오류: {str(e)}</pre>", status_code=500)

    # 4) DB 저장
    try:
        res = upsert_to_db(
            title=title,
            filename=name,
            mime=mime,
            bytes_len=len(file_bytes),
            whole_text=text,
            chunks=chunks,
            embeds=embeds
        )
    except Exception as e:
        return HTMLResponse(f"<pre>DB 저장 오류: {str(e)}</pre>", status_code=500)

    pretty = f"""
업로드/인덱싱 완료

- title: {title}
- filename: {name}
- mime: {mime}
- bytes: {len(file_bytes)}
- chunks: {res.get('chunks')}

document_id: {res.get('document_id')}
"""
    back = '<p><a href="/">← 돌아가기</a></p>'
    return HTMLResponse(f"<pre>{pretty}</pre>{back}")
