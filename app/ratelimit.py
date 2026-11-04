"""요청 제한 (슬라이딩 윈도우). 서버 한 대 안에서만 세고, 재시작하면 초기화됩니다."""
import time
from collections import deque
from typing import Optional


class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0, max_clients: int = 5000):
        self.limit, self.window, self.max_clients = limit, window, max_clients
        self._hits: dict[str, deque] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        """허용하면 True(그리고 이번 요청을 기록), 한도를 넘으면 False."""
        now = time.monotonic() if now is None else now
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self.max_clients:
                self._prune(now)
            hits = self._hits[key] = deque()
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def _prune(self, now: float) -> None:
        """기록이 만료된 접속자를 정리합니다. 그래도 가득 차면 전부 비웁니다(메모리 보호)."""
        cutoff = now - self.window
        self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
        if len(self._hits) >= self.max_clients:
            self._hits.clear()