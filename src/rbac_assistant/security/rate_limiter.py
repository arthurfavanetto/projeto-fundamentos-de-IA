"""Rate limiter determinístico por usuário para proteção contra abuso na interface Telegram."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """Resultado da verificação de limite de taxa para um usuário (ou limite global)."""

    allowed: bool
    retry_after_seconds: float = 0.0
    remaining_in_window: int = 0
    scope: str = "user"


class UserRateLimiter:
    """Controle de taxa por janela deslizante + rajada curta por usuário e limite global."""

    def __init__(
        self,
        *,
        requests_per_minute: int = 15,
        burst_size: int = 5,
        burst_window_seconds: float = 10.0,
        window_seconds: float = 60.0,
        global_requests_per_minute: int | None = None,
        max_tracked_users: int = 4096,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute deve ser >= 1.")
        if burst_size < 1:
            raise ValueError("burst_size deve ser >= 1.")
        if burst_window_seconds <= 0 or window_seconds <= 0:
            raise ValueError("Janelas de tempo devem ser > 0.")
        if global_requests_per_minute is not None and global_requests_per_minute < 1:
            raise ValueError("global_requests_per_minute deve ser >= 1.")

        self._requests_per_minute = requests_per_minute
        self._burst_size = min(burst_size, requests_per_minute)
        self._burst_window_seconds = burst_window_seconds
        self._window_seconds = window_seconds
        self._global_requests_per_minute = global_requests_per_minute
        self._max_tracked_users = max_tracked_users
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: dict[int, deque[float]] = {}
        self._global_bucket: deque[float] = deque()

    def _evict_if_needed(self, now: float) -> None:
        if len(self._buckets) < self._max_tracked_users:
            return
        stale_keys = [
            uid
            for uid, timestamps in self._buckets.items()
            if not timestamps or (now - timestamps[-1]) > self._window_seconds
        ]
        for uid in stale_keys:
            del self._buckets[uid]
        if len(self._buckets) >= self._max_tracked_users:
            oldest_uid = min(
                self._buckets,
                key=lambda uid: self._buckets[uid][-1] if self._buckets[uid] else -1.0,
            )
            del self._buckets[oldest_uid]

    def check(self, user_id: int) -> RateLimitResult:
        """Registra tentativa e informa se o usuário (e o teto global) podem prosseguir."""
        now = self._clock()
        with self._lock:
            self._evict_if_needed(now)
            cutoff = now - self._window_seconds

            # 1. Verifica teto global (protege cota Free Tier sob tráfego público simultâneo)
            if self._global_requests_per_minute is not None:
                while self._global_bucket and self._global_bucket[0] <= cutoff:
                    self._global_bucket.popleft()
                if len(self._global_bucket) >= self._global_requests_per_minute:
                    retry_after = max(0.1, self._window_seconds - (now - self._global_bucket[0]))
                    return RateLimitResult(
                        allowed=False,
                        retry_after_seconds=round(retry_after, 1),
                        remaining_in_window=0,
                        scope="global",
                    )

            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = deque()
                self._buckets[user_id] = bucket

            # 2. Remove timestamps fora da janela principal de 60s
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            # 3. Verifica limite por usuário na janela completa (requests_per_minute)
            if len(bucket) >= self._requests_per_minute:
                retry_after = max(0.1, self._window_seconds - (now - bucket[0]))
                return RateLimitResult(
                    allowed=False,
                    retry_after_seconds=round(retry_after, 1),
                    remaining_in_window=0,
                    scope="user",
                )

            # 4. Verifica limite de rajada curta (burst_size nos últimos burst_window_seconds)
            burst_cutoff = now - self._burst_window_seconds
            recent_burst = [ts for ts in bucket if ts > burst_cutoff]
            if len(recent_burst) >= self._burst_size:
                retry_after = max(0.1, self._burst_window_seconds - (now - recent_burst[0]))
                return RateLimitResult(
                    allowed=False,
                    retry_after_seconds=round(retry_after, 1),
                    remaining_in_window=max(0, self._requests_per_minute - len(bucket)),
                    scope="user",
                )

            bucket.append(now)
            if self._global_requests_per_minute is not None:
                self._global_bucket.append(now)
            remaining = max(0, self._requests_per_minute - len(bucket))
            return RateLimitResult(
                allowed=True,
                retry_after_seconds=0.0,
                remaining_in_window=remaining,
                scope="user",
            )
