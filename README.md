# RAG Upload Admin (FastAPI)

**문서 업로드** → **텍스트 추출** → **청크 분할** → **임베딩 생성(Ollama)** → **벡터 저장(Supabase)**

RAG(Retrieval-Augmented Generation) 시스템을 위한 문서 인덱싱 도구입니다.

## Process 정리

### **처리 과정**

1. **파일 업로드**: PDF, DOCX, TXT 파일 지원
2. **텍스트 추출**: 각 파일 형식에서 순수 텍스트 추출
3. **청크 분할**: 800자 단위로 분할 (150자 중복)
4. **임베딩 생성**: Ollama의 nomic-embed-text 모델 (768차원)
5. **벡터 저장**: Supabase pgvector에 검색 가능한 형태로 저장

### **DB Structure**

```sql
documents 테이블 (메타데이터)
├─ id: 문서 고유 ID (UUID)
├─ title: 문서 제목
├─ source_path: 원본 파일명
├─ mime_type: 파일 타입
├─ bytes: 파일 크기
└─ hash: 중복 방지용 해시

chunks 테이블 (검색 데이터)
├─ document_id: 상위 문서 참조
├─ chunk_index: 청크 순서
├─ content: 실제 텍스트 내용
└─ embedding: 768차원 벡터 (검색용)
```

## 설정

### **Ollama 설정**

```bash
# Ollama 설치 (https://ollama.ai)
ollama pull nomic-embed-text    # 임베딩 모델 다운로드
ollama serve                    # 서버 실행
```

### **4단계: 프로젝트 실행**

```bash
# 1) 가상환경 생성
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 환경변수 설정
cp env.example .env
# .env 파일 편집:
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_ANON_KEY=your-anon-key

# 4) 서버 실행
uvicorn app:APP --reload --port 8000
```

## 사용 예시

**입력**: `Contact Point_240826.pdf` (193KB)

```
업로드/인덱싱 완료

- title: pdf test v0
- filename: Contact Point_240826.pdf
- mime: application/pdf
- bytes: 193985
- chunks: 6

document_id: 888088ee-d143-4327-9d97-1f309f342de6
```

**데이터베이스 저장 결과**:

- `documents` 테이블: 1개 레코드 (메타데이터)
- `chunks` 테이블: 6개 레코드 (검색 가능한 벡터 데이터)

## Troubleshooting

- **Supabase 연결 오류**: `.env`의 `SUPABASE_URL`, `SUPABASE_ANON_KEY` 확인
- **임베딩 오류**: `ollama serve` 실행 상태 및 `nomic-embed-text` 모델 확인
- **파일 업로드 실패**: 지원 형식(PDF/DOCX/TXT) 및 파일 크기 확인
- **RPC 함수 오류**: Supabase SQL Editor에서 함수 생성 확인

## 🚀 확장 아이디어

### **검색 기능 추가**

```python
@app.post("/search")
def search_documents(query: str, top_k: int = 5):
    # 1) 질의 임베딩 생성
    # 2) 코사인 유사도 검색
    # 3) 상위 K개 청크 반환
```

### **기타 확장 고려중..**

- **Supabase Storage**: 원본 파일 보관용
- **다국어 지원**: 다양한 임베딩 모델 처리 가능해보임
- **배치 업로드**: 여러 파일 동시 처리 가능하면 유용할듯.

---
