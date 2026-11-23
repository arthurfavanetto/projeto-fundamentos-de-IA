"""Rate limiter determinístico por usuário para proteção contra abuso na interface Telegram."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """Resultado da verificação de limite de taxa para um usuário."""

    allowed: bool
    retry_after_seconds: float = 0.0
    remaining_in_window: int = 0


class UserRateLimiter:
    """Controle de taxa por janela deslizante + rajada curta (`burst_size`) por `user_id`."""

    def __init__(
        self,
        *,
        requests_per_minute: int = 15,
        burst_size: int = 5,
        burst_window_seconds: float = 10.0,
        window_seconds: float = 60.0,
        max_tracked_users: int = 4096,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute deve ser >= 1.")
        if burst_size < 1:
            raise ValueError("burst_size deve ser >= 1.")
        if burst_window_seconds <= 0 or window_seconds <= 0:
            raise ValueError("Janelas de tempo devem ser > 0.")

        self._requests_per_minute = requests_per_minute
        self._burst_size = min(burst_size, requests_per_minute)
        self._burst_window_seconds = burst_window_seconds
        self._window_seconds = window_seconds
        self._max_tracked_users = max_tracked_users
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: dict[int, deque[float]] = {}

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

    def check(self, user_id: int) -> RateLimitResult:
        """Registra tentativa e informa se o usuário pode executar nova requisição."""
        now = self._clock()
        with self._lock:
            self._evict_if_needed(now)
            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = deque()
                self._buckets[user_id] = bucket

            # Remove timestamps fora da janela principal de 60s
            cutoff = now - self._window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            # Verifica limite de janela completa (requests_per_minute)
            if len(bucket) >= self._requests_per_minute:
                retry_after = max(0.1, self._window_seconds - (now - bucket[0]))
                return RateLimitResult(
                    allowed=False,
                    retry_after_seconds=round(retry_after, 1),
                    remaining_in_window=0,
                )

            # Verifica limite de rajada curta (burst_size nos últimos burst_window_seconds)
            burst_cutoff = now - self._burst_window_seconds
            recent_burst = [ts for ts in bucket if ts > burst_cutoff]
            if len(recent_burst) >= self._burst_size:
                retry_after = max(0.1, self._burst_window_seconds - (now - recent_burst[0]))
                return RateLimitResult(
                    allowed=False,
                    retry_after_seconds=round(retry_after, 1),
                    remaining_in_window=max(0, self._requests_per_minute - len(bucket)),
                )

            bucket.append(now)
            remaining = max(0, self._requests_per_minute - len(bucket))
            return RateLimitResult(
                allowed=True,
                retry_after_seconds=0.0,
                remaining_in_window=remaining,
            )
