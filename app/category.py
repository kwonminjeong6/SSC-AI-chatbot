"""'세부 검색'에서 사용자가 고른 분야(대분류 번호)를 검증합니다.

질문 내용으로 분야를 추측하지 않습니다. 사용자가 고르지 않으면 항상 전체 규정에서 검색합니다.
"""
from typing import Optional

from app import config


def normalize_groups(requested: Optional[list[str]]) -> Optional[list[str]]:
    """유효한 분야 번호만 중복 없이 남깁니다.

    비었거나 전부 잘못된 값이면 None(전체 검색). 모든 분야를 고른 경우도 None으로 처리해
    번호가 없는 문서까지 포함한 전체 검색이 되게 합니다.
    """
    if not requested:
        return None
    valid = [g for g in dict.fromkeys(str(x).strip() for x in requested) if g in config.GROUPS]
    if not valid or len(valid) == len(config.GROUPS):
        return None
    return valid


def list_groups() -> list[dict]:
    """화면의 선택지로 내려줄 분야 목록 (번호 순)."""
    return [{"id": gid, "name": name} for gid, name in config.GROUPS.items()]
