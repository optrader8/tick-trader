"""
LSTM 기반 시계열 분류 모델.

장단기 메모리 네트워크를 사용하여 틱데이터의 시계열 패턴을 학습합니다.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Install tensorflow to use LSTM models.")

from ..exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM 기반 시계열 분류 모델.

    다층 LSTM 구조로 시계열 패턴을 학습하여 상승/하락을 예측합니다.
    """

    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        lstm_units: list = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Args:
            sequence_length: 입력 시퀀스 길이
            n_features: 특징 개수
            lstm_units: 각 LSTM 레이어의 유닛 수 리스트
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
        """
        if not TF_AVAILABLE:
            raise ModelTrainingError("TensorFlow is not available")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self.history = None

    def build_model(self) -> keras.Model:
        """
        LSTM 모델을 구축합니다.

        Returns:
            Keras 모델
        """
        model = keras.Sequential(name="LSTMModel")

        # 첫 번째 LSTM 레이어
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            name="lstm_1"
        ))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_1"))

        # 중간 LSTM 레이어들
        for i, units in enumerate(self.lstm_units[1:-1], start=2):
            model.add(layers.LSTM(
                units,
                return_sequences=True,
                name=f"lstm_{i}"
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # 마지막 LSTM 레이어 (return_sequences=False)
        if len(self.lstm_units) > 1:
            model.add(layers.LSTM(
                self.lstm_units[-1],
                return_sequences=False,
                name=f"lstm_{len(self.lstm_units)}"
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{len(self.lstm_units)}"))

        # Dense 레이어
        model.add(layers.Dense(16, activation='relu', name="dense_1"))
        model.add(layers.Dense(2, activation='softmax', name="output"))

        # 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        모델을 학습합니다.

        Args:
            X_train: 학습 데이터 (samples, sequence_length, features)
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            epochs: 에폭 수
            batch_size: 배치 크기

        Returns:
            학습 히스토리
        """
        if self.model is None:
            self.build_model()

        # 콜백 설정
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # 검증 데이터 설정
        validation_data = (X_val, y_val) if X_val is not None else None

        # 학습
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )

        logger.info("Model training completed")

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측을 수행합니다.

        Args:
            X: 입력 데이터

        Returns:
            예측 확률 배열
        """
        if self.model is None:
            raise ModelTrainingError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """
        클래스 예측을 수행합니다.

        Args:
            X: 입력 데이터

        Returns:
            예측 클래스 배열
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        모델을 평가합니다.

        Args:
            X_test: 테스트 데이터
            y_test: 테스트 레이블

        Returns:
            평가 메트릭
        """
        if self.model is None:
            raise ModelTrainingError("Model not trained")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return {
            "loss": loss,
            "accuracy": accuracy
        }

    def save(self, filepath: str) -> None:
        """모델을 저장합니다."""
        if self.model is None:
            raise ModelTrainingError("No model to save")

        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """모델을 로드합니다."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
