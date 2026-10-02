import requests
from functools import lru_cache
from app.embedding import embedding_model
from app.qdrant import qdrant
from app.config import OLLAMA_URL, OLLAMA_MODEL, TOP_K

def generate_answer(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 대학 공식 정보 안내 AI입니다. 반드시 제공된 문서에 근거해서만 간결하고 명확하게 답변하십시오."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 256
            }
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Ollama 응답 오류: {data}")

    return data["message"]["content"]

def create_prompt(question, search_results):
    context_list = []
    for i, result in enumerate(search_results):
        payload = result.payload
        document_name = payload.get("document_name", "알 수 없음")
        page = payload.get("page", "-")
        text = payload.get("text", "")
        context_list.append(f"[문서 {i + 1}] 문서명: {document_name} (페이지: {page})\n내용: {text}")

    context = "\n\n".join(context_list)

    # 퓨샷(Few-shot) 및 엄격한 제어 프롬프트 적용
    prompt = f"""[대학 공식 문서]
{context}

[사용자 질문]
{question}

[지침]
1. 위 문서 내용만을 근거로 사실에 기반하여 답변하십시오.
2. 문서에 없는 내용은 절대 추측하거나 지어내지 마십시오.
3. 문서에서 근거를 찾을 수 없는 경우, 반드시 "현재 등록된 대학 공식 자료에서는 해당 내용을 확인할 수 없습니다."라고 답변하십시오.
4. 답변 시 출처가 되는 문서명과 조항(또는 페이지)을 함께 명시하십시오.

[답변]"""
    return prompt

# 동일한 질문에 대한 속도 개선을 위한 캐싱 처리 (메모이제이션)
@lru_cache(maxsize=100)
def cached_rag_query(question: str):
    query_vector = embedding_model.encode(question)
    results = qdrant.search(query_vector, limit=TOP_K)

    if not results:
        return {
            "answer": "현재 등록된 대학 공식 자료에서는 해당 내용을 확인할 수 없습니다.",
            "sources": []
        }

    prompt = create_prompt(question, results)
    answer = generate_answer(prompt)

    sources = []
    for result in results:
        payload = result.payload
        sources.append({
            "document_name": payload.get("document_name", ""),
            "page": payload.get("page", ""),
            "score": result.score
        })

    return {
        "answer": answer,
        "sources": sources
    }

def rag(question: str):
    return cached_rag_query(question)