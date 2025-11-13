"""
앙상블 모델.

여러 기계학습 모델을 결합하여 예측 성능을 향상시킵니다.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        StackingClassifier,
        VotingClassifier
    )
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn, lightgbm, or xgboost not available")

from ..exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    앙상블 분류 모델.

    LightGBM, XGBoost, RandomForest를 결합한 스태킹 앙상블입니다.
    """

    def __init__(
        self,
        ensemble_type: str = "stacking",
        n_estimators: int = 100
    ):
        """
        Args:
            ensemble_type: 앙상블 타입 ('stacking' 또는 'voting')
            n_estimators: 기본 추정기의 트리 개수
        """
        if not SKLEARN_AVAILABLE:
            raise ModelTrainingError("Required libraries not available")

        self.ensemble_type = ensemble_type
        self.n_estimators = n_estimators
        self.model = None

    def create_stacking_ensemble(self) -> StackingClassifier:
        """스태킹 앙상블을 생성합니다."""
        # 기본 모델들
        base_models = [
            ('lgbm', LGBMClassifier(n_estimators=self.n_estimators, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=self.n_estimators, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=self.n_estimators, random_state=42))
        ]

        # 메타 모델
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression()

        # 스태킹 앙상블
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )

        return ensemble

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """모델을 학습합니다."""
        if self.ensemble_type == "stacking":
            self.model = self.create_stacking_ensemble()
        else:
            raise NotImplementedError(f"Ensemble type {self.ensemble_type} not implemented")

        logger.info(f"Training {self.ensemble_type} ensemble...")
        self.model.fit(X_train, y_train)
        logger.info("Ensemble training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 확률을 반환합니다."""
        if self.model is None:
            raise ModelTrainingError("Model not trained")
        return self.model.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """클래스 예측을 반환합니다."""
        if self.model is None:
            raise ModelTrainingError("Model not trained")
        return self.model.predict(X)
