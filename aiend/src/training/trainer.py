"""
모델 학습 서비스.

다양한 모델의 학습 및 하이퍼파라미터 최적화를 관리합니다.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split

from ..models.lstm import LSTMModel
from ..models.ensemble import EnsembleModel
from ..models.transformer import TransformerModel
from ..models.cnn_lstm import CNNLSTMModel
from ..exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    모델 학습 orchestrator.

    다양한 모델 타입의 학습을 관리합니다.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 학습 설정
        """
        self.config = config or {}
        self.model = None
        self.history = None

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """
        모델을 학습합니다.

        Args:
            model_type: 모델 타입 ('lstm', 'ensemble', 'transformer', 'cnn_lstm')
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            **kwargs: 모델별 추가 파라미터

        Returns:
            학습된 모델
        """
        logger.info(f"Training {model_type} model...")

        if model_type == "lstm":
            self.model = self._train_lstm(X_train, y_train, X_val, y_val, **kwargs)
        elif model_type == "ensemble":
            self.model = self._train_ensemble(X_train, y_train, **kwargs)
        elif model_type == "transformer":
            self.model = self._train_transformer(X_train, y_train, X_val, y_val, **kwargs)
        elif model_type == "cnn_lstm":
            self.model = self._train_cnn_lstm(X_train, y_train, X_val, y_val, **kwargs)
        else:
            raise ModelTrainingError(f"Unknown model type: {model_type}")

        logger.info("Training completed")
        return self.model

    def _train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        **kwargs
    ) -> LSTMModel:
        """LSTM 모델 학습."""
        model = LSTMModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            **kwargs
        )

        self.history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32)
        )

        return model

    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> EnsembleModel:
        """앙상블 모델 학습."""
        # 3D -> 2D 변환 (앙상블 모델용)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        model = EnsembleModel(**kwargs)
        model.train(X_train, y_train)

        return model

    def _train_transformer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        **kwargs
    ) -> TransformerModel:
        """Transformer 모델 학습."""
        model = TransformerModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            **kwargs
        )

        self.history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32)
        )

        return model

    def _train_cnn_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        **kwargs
    ) -> CNNLSTMModel:
        """CNN-LSTM 모델 학습."""
        model = CNNLSTMModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            **kwargs
        )

        model.build_model()
        model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=kwargs.get('epochs', 50),
            batch_size=kwargs.get('batch_size', 32)
        )

        return model
