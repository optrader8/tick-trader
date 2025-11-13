"""
Transformer 기반 어텐션 모델.

셀프 어텐션 메커니즘을 사용하여 시계열 패턴을 학습합니다.
"""

import logging
from typing import Dict, Optional
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class TransformerModel:
    """
    Transformer 기반 분류 모델.

    Multi-head self-attention을 사용하여 시계열을 학습합니다.
    """

    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            sequence_length: 입력 시퀀스 길이
            n_features: 특징 개수
            d_model: 모델 차원
            n_heads: 어텐션 헤드 수
            num_layers: 트랜스포머 레이어 수
            dropout_rate: 드롭아웃 비율
        """
        if not TF_AVAILABLE:
            raise ModelTrainingError("TensorFlow not available")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None

    def build_attention_model(self) -> keras.Model:
        """Transformer 모델을 구축합니다."""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))

        # 입력 프로젝션
        x = layers.Dense(self.d_model)(inputs)

        # Positional encoding
        x = self._add_positional_encoding(x)

        # Transformer 레이어들
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=self.dropout_rate
            )(x, x)

            # Add & Norm
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed forward
            ff_output = self._feed_forward(x)

            # Add & Norm
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Classification head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info(f"Built Transformer model with {model.count_params()} parameters")
        return model

    def _add_positional_encoding(self, x):
        """Positional encoding을 추가합니다."""
        position = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.d_model
        )(position)
        return x + position_embedding

    def _feed_forward(self, x):
        """Feed-forward 네트워크."""
        ff_dim = self.d_model * 4
        x = layers.Dense(ff_dim, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.d_model)(x)
        return x

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """모델을 학습합니다."""
        if self.model is None:
            self.build_attention_model()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        return history.history
