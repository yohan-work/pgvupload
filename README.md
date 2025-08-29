# RAG Upload Admin (FastAPI)

업로드 1페이지 → 텍스트 추출 → 청크 → 임베딩(Ollama) → Supabase(pgvector)에 저장하는 초간단 어드민.

## 요구사항
- Python 3.10+
- 로컬에서 Ollama 실행 및 임베딩 모델 다운로드
- Supabase (Postgres + pgvector)

## 1) 데이터베이스 준비
Supabase SQL Editor에서 `sql/schema.sql` 실행

## 2) Ollama 준비
```bash
ollama pull nomic-embed-text
ollama serve
```

## 3) 로컬 실행
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

cp env.example .env               # .env 값 채우기
uvicorn app:APP --reload --port 8000
```

브라우저에서 `http://127.0.0.1:8000` 접속 → 업로드 테스트

## Troubleshooting

- **DB 저장 오류**: `.env`의 `SUPABASE_DB_CONN` 확인 (방화벽/SSL 옵션 필요 시 커넥션스트링에 추가)
- **임베딩 오류**: `ollama serve` 실행 여부, `EMBED_MODEL` 이름 확인
- **PDF 추출 품질이 낮으면** `pymupdf` 도입 고려

## 확장 아이디어

- Supabase Storage 원본 업로드
- `/search` 엔드포인트 추가 (질의 임베딩 → top-k 검색)
- 중복 방지 기준을 바이너리 SHA256로 전환
