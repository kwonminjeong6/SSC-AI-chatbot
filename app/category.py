"""질문 텍스트로 카테고리를 자동 분류합니다."""
from typing import Optional

KEYWORDS: dict[str, tuple[str, ...]] = {
    "장학금": ("장학", "감면", "면제", "수혜", "다자녀", "소득구간", "분위"),
    "학사규정": (
        "휴학", "복학", "학점", "졸업", "학적", "수강", "성적", "출석", "제적", "전과", "편입",
        "입학정원", "정원", "학과", "교육과정", "학칙", "직제", "부속기관", "개정", "시수",
    ),
}


def detect_category(query: str) -> Optional[str]:
    """가장 많은 키워드가 맞은 카테고리를 반환합니다.

    매칭이 없거나 동점(예: "휴학하면 장학금은?")이면 None → 전체 검색.
    """
    scores = {cat: sum(kw in query for kw in kws) for cat, kws in KEYWORDS.items()}
    best = max(scores.values())
    winners = [cat for cat, score in scores.items() if score == best]
    return winners[0] if best > 0 and len(winners) == 1 else None
