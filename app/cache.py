"""질문-답변 메모리 캐시. 서버 한 대 안에서만 유효하고, 재시작하면 비워집니다."""
import re
import time
from collections import OrderedDict
from typing import Optional

from app import config


def make_key(query: str, groups: Optional[list[str]]) -> tuple:
    q = re.sub(r"\s+", "", query).lower()
    q = re.sub(r"[?？!！.。,，~]+$", "", q)  # 끝의 문장부호 제거
    return (q, tuple(sorted(groups)) if groups else None)


class TTLCache:
    def __init__(self, max_size: int, ttl: float):
        self.max_size, self.ttl = max_size, ttl
        self._data: OrderedDict = OrderedDict()

    def get(self, key):
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():  # 만료
            del self._data[key]
            return None
        self._data.move_to_end(key)  # 최근 사용 표시
        return value

    def set(self, key, value) -> None:
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:  # 가장 오래 안 쓴 것부터 제거
            self._data.popitem(last=False)


answer_cache = TTLCache(config.CACHE_MAX, config.CACHE_TTL)