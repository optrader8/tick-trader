"""
CNN-LSTM 하이브리드 모델.

CNN으로 지역적 패턴을 추출하고 LSTM으로 시간적 의존성을 학습합니다.
"""

import logging
import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class CNNLSTMModel:
    """CNN-LSTM 하이브리드 분류 모델."""

    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        conv_filters: list = None,
        lstm_units: int = 64
    ):
        """
        Args:
            sequence_length: 입력 시퀀스 길이
            n_features: 특징 개수
            conv_filters: CNN 필터 개수 리스트
            lstm_units: LSTM 유닛 수
        """
        if not TF_AVAILABLE:
            raise ModelTrainingError("TensorFlow not available")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.conv_filters = conv_filters or [64, 32]
        self.lstm_units = lstm_units
        self.model = None

    def build_model(self) -> keras.Model:
        """하이브리드 모델을 구축합니다."""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))

        # CNN 레이어들 (지역 패턴 추출)
        x = inputs
        for filters in self.conv_filters:
            x = layers.Conv1D(filters, kernel_size=3, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(pool_size=2)(x)

        # LSTM 레이어 (시간적 의존성)
        x = layers.LSTM(self.lstm_units)(x)

        # Classification head
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model
