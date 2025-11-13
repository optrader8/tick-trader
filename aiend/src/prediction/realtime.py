"""
실시간 예측 서비스.

실시간 틱데이터에 대한 즉시 예측을 수행합니다.
"""

import logging
from typing import Dict, Optional, Any
from collections import deque
import numpy as np

from ..data.models import OrderBookSnapshot, PredictionResult
from ..features.pipeline import FeaturePipeline
from ..storage.redis_cache import RedisFeatureCache, InMemoryCache
from ..exceptions import PredictionError, InsufficientDataError

logger = logging.getLogger(__name__)


class RealTimePredictor:
    """
    실시간 예측기.

    슬라이딩 윈도우 버퍼를 관리하며 실시간 예측을 수행합니다.
    """

    def __init__(
        self,
        model: Any,
        feature_pipeline: FeaturePipeline,
        window_size: int = 100,
        use_redis: bool = False,
        model_version: str = "1.0.0"
    ):
        """
        Args:
            model: 학습된 모델
            feature_pipeline: 특징 파이프라인
            window_size: 윈도우 크기
            use_redis: Redis 캐시 사용 여부
            model_version: 모델 버전
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.window_size = window_size
        self.model_version = model_version

        # 버퍼 초기화
        self.snapshot_buffer = deque(maxlen=window_size)

        # 캐시 초기화
        if use_redis:
            try:
                self.cache = RedisFeatureCache()
            except Exception as e:
                logger.warning(f"Failed to initialize Redis, using in-memory cache: {e}")
                self.cache = InMemoryCache()
        else:
            self.cache = InMemoryCache()

    def update_and_predict(
        self,
        snapshot: OrderBookSnapshot
    ) -> Optional[PredictionResult]:
        """
        새로운 스냅샷으로 버퍼를 업데이트하고 예측을 수행합니다.

        Args:
            snapshot: 새로운 호가창 스냅샷

        Returns:
            예측 결과 (버퍼가 충분하지 않으면 None)
        """
        # 버퍼에 추가
        self.snapshot_buffer.append(snapshot)

        # 버퍼가 충분한지 확인
        if len(self.snapshot_buffer) < self.window_size:
            logger.debug(f"Buffer not full yet: {len(self.snapshot_buffer)}/{self.window_size}")
            return None

        try:
            # 특징 추출
            snapshots = list(self.snapshot_buffer)
            features_df = self.feature_pipeline.transform(snapshots)

            # 스케일링
            if self.feature_pipeline._is_fitted:
                features_df = self.feature_pipeline.scale_features(features_df)

            # 예측 수행
            X = features_df.select_dtypes(include=[np.number]).values
            X = X.reshape(1, *X.shape)  # 배치 차원 추가

            # 모델 예측
            if hasattr(self.model, 'predict'):
                prediction_probs = self.model.predict(X)
                if len(prediction_probs.shape) > 1:
                    prediction = prediction_probs[0][1]  # 상승 확률
                    prediction_class = np.argmax(prediction_probs[0])
                    confidence = np.max(prediction_probs[0])
                else:
                    prediction = prediction_probs[0]
                    prediction_class = 1 if prediction > 0.5 else 0
                    confidence = abs(prediction - 0.5) * 2
            else:
                raise PredictionError("Model does not have predict method")

            # 결과 생성
            result = PredictionResult(
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                prediction=float(prediction),
                confidence=float(confidence),
                model_version=self.model_version,
                prediction_class=int(prediction_class)
            )

            logger.info(
                f"Prediction: {result.prediction:.3f} "
                f"(class: {result.prediction_class}, confidence: {result.confidence:.3f})"
            )

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Failed to make prediction: {e}")

    def reset_buffer(self) -> None:
        """버퍼를 초기화합니다."""
        self.snapshot_buffer.clear()
        logger.info("Buffer reset")

    def get_buffer_size(self) -> int:
        """현재 버퍼 크기를 반환합니다."""
        return len(self.snapshot_buffer)
