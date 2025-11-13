"""
Redis 기반 특징 캐시.

실시간 특징 버퍼를 관리하고 빠른 조회를 제공합니다.
"""

import logging
import json
import pickle
from typing import Optional, List
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install redis-py to use caching features.")

from ..exceptions import CacheError

logger = logging.getLogger(__name__)


class RedisFeatureCache:
    """
    Redis 기반 특징 캐시.

    실시간 특징을 캐싱하고 슬라이딩 윈도우를 관리합니다.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        Args:
            host: Redis 호스트
            port: Redis 포트
            db: 데이터베이스 번호
            password: 비밀번호
        """
        if not REDIS_AVAILABLE:
            raise CacheError("Redis is not available. Install redis-py package.")

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )
            # 연결 테스트
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")

        except Exception as e:
            raise CacheError(f"Failed to connect to Redis: {e}")

    def cache_features(
        self,
        key: str,
        features: np.ndarray,
        ttl: int = 3600
    ) -> None:
        """
        특징을 캐시에 저장합니다.

        Args:
            key: 캐시 키
            features: 특징 배열
            ttl: Time To Live (초)
        """
        try:
            # NumPy 배열을 바이트로 직렬화
            serialized = pickle.dumps(features)
            self.client.setex(key, ttl, serialized)

            logger.debug(f"Cached features with key: {key}")

        except Exception as e:
            raise CacheError(f"Failed to cache features: {e}")

    def get_cached_features(self, key: str) -> Optional[np.ndarray]:
        """
        캐시에서 특징을 조회합니다.

        Args:
            key: 캐시 키

        Returns:
            특징 배열 (없으면 None)
        """
        try:
            data = self.client.get(key)

            if data is None:
                return None

            # 역직렬화
            features = pickle.loads(data)
            return features

        except Exception as e:
            logger.error(f"Failed to get cached features: {e}")
            return None

    def manage_sliding_window(
        self,
        key: str,
        new_data: np.ndarray,
        max_size: int
    ) -> List[np.ndarray]:
        """
        슬라이딩 윈도우 버퍼를 관리합니다.

        Args:
            key: 버퍼 키
            new_data: 새로운 데이터
            max_size: 최대 버퍼 크기

        Returns:
            현재 버퍼의 모든 데이터
        """
        try:
            # 리스트에 새 데이터 추가
            serialized = pickle.dumps(new_data)
            self.client.lpush(key, serialized)

            # 최대 크기 유지
            self.client.ltrim(key, 0, max_size - 1)

            # 전체 데이터 조회
            buffer_data = self.client.lrange(key, 0, -1)
            return [pickle.loads(data) for data in buffer_data]

        except Exception as e:
            raise CacheError(f"Failed to manage sliding window: {e}")

    def get_window_data(
        self,
        key: str,
        window_size: int
    ) -> List[np.ndarray]:
        """
        윈도우 버퍼에서 데이터를 조회합니다.

        Args:
            key: 버퍼 키
            window_size: 조회할 윈도우 크기

        Returns:
            윈도우 데이터 리스트
        """
        try:
            buffer_data = self.client.lrange(key, 0, window_size - 1)
            return [pickle.loads(data) for data in buffer_data]

        except Exception as e:
            logger.error(f"Failed to get window data: {e}")
            return []

    def clear_cache(self, pattern: str = "*") -> int:
        """
        패턴에 맞는 캐시를 삭제합니다.

        Args:
            pattern: 키 패턴

        Returns:
            삭제된 키 개수
        """
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    def get_cache_stats(self) -> dict:
        """
        캐시 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        try:
            info = self.client.info()
            return {
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "total_keys": self.client.dbsize(),
                "hit_rate": info.get("keyspace_hits", 0) /
                           max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def close(self) -> None:
        """Redis 연결을 종료합니다."""
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Failed to close Redis connection: {e}")


class InMemoryCache:
    """
    Redis를 사용할 수 없을 때의 대체 인메모리 캐시.
    """

    def __init__(self):
        self.cache = {}
        self.buffers = {}

    def cache_features(self, key: str, features: np.ndarray, ttl: int = 3600) -> None:
        """특징 캐싱 (TTL은 무시됨)."""
        self.cache[key] = features.copy()

    def get_cached_features(self, key: str) -> Optional[np.ndarray]:
        """캐시 조회."""
        return self.cache.get(key)

    def manage_sliding_window(
        self,
        key: str,
        new_data: np.ndarray,
        max_size: int
    ) -> List[np.ndarray]:
        """슬라이딩 윈도우 관리."""
        if key not in self.buffers:
            self.buffers[key] = []

        self.buffers[key].insert(0, new_data.copy())

        # 최대 크기 유지
        if len(self.buffers[key]) > max_size:
            self.buffers[key] = self.buffers[key][:max_size]

        return self.buffers[key].copy()

    def get_window_data(self, key: str, window_size: int) -> List[np.ndarray]:
        """윈도우 데이터 조회."""
        if key not in self.buffers:
            return []

        return self.buffers[key][:window_size]

    def clear_cache(self, pattern: str = "*") -> int:
        """캐시 삭제."""
        count = len(self.cache)
        self.cache.clear()
        self.buffers.clear()
        return count

    def get_cache_stats(self) -> dict:
        """캐시 통계."""
        return {
            "total_keys": len(self.cache) + len(self.buffers),
            "cache_keys": len(self.cache),
            "buffer_keys": len(self.buffers)
        }

    def close(self) -> None:
        """연결 종료 (아무 작업 안함)."""
        pass
