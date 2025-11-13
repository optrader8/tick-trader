"""
Core data models for tick trading system.

이 모듈은 틱데이터 분석을 위한 핵심 데이터 구조를 정의합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np


@dataclass
class PriceLevel:
    """
    호가창의 단일 가격 레벨을 표현하는 데이터 클래스.

    Attributes:
        price: 호가 가격
        volume: 해당 가격의 주문 수량
        open_interest: 미결제약정 수량
        order_count: 주문 건수
    """
    price: float
    volume: float
    open_interest: float = 0.0
    order_count: int = 0

    def __post_init__(self):
        """데이터 유효성 검증."""
        if self.price < 0:
            raise ValueError("Price must be non-negative")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")
        if self.open_interest < 0:
            raise ValueError("Open interest must be non-negative")
        if self.order_count < 0:
            raise ValueError("Order count must be non-negative")


@dataclass
class OrderBookSnapshot:
    """
    특정 시점의 호가창 스냅샷.

    Attributes:
        timestamp: 스냅샷 시각
        symbol: 종목 코드 (예: ES, NQ 등)
        bid_levels: 매수 호가 레벨 리스트 (가격 높은 순으로 정렬)
        ask_levels: 매도 호가 레벨 리스트 (가격 낮은 순으로 정렬)
        last_price: 최종 체결가
        total_volume: 누적 거래량
    """
    timestamp: datetime
    symbol: str
    bid_levels: List[PriceLevel] = field(default_factory=list)
    ask_levels: List[PriceLevel] = field(default_factory=list)
    last_price: Optional[float] = None
    total_volume: float = 0.0

    def __post_init__(self):
        """데이터 유효성 검증 및 정렬."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

        # 매수 호가는 가격 높은 순으로 정렬
        self.bid_levels.sort(key=lambda x: x.price, reverse=True)

        # 매도 호가는 가격 낮은 순으로 정렬
        self.ask_levels.sort(key=lambda x: x.price)

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """최우선 매수 호가."""
        return self.bid_levels[0] if self.bid_levels else None

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """최우선 매도 호가."""
        return self.ask_levels[0] if self.ask_levels else None

    @property
    def spread(self) -> Optional[float]:
        """호가 스프레드 (매도 최우선가 - 매수 최우선가)."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """중간 가격 (매수 최우선가 + 매도 최우선가) / 2."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    def total_bid_volume(self, depth: Optional[int] = None) -> float:
        """
        총 매수 호가 수량.

        Args:
            depth: 계산할 호가 레벨 깊이 (None이면 전체)

        Returns:
            총 매수 수량
        """
        levels = self.bid_levels[:depth] if depth else self.bid_levels
        return sum(level.volume for level in levels)

    def total_ask_volume(self, depth: Optional[int] = None) -> float:
        """
        총 매도 호가 수량.

        Args:
            depth: 계산할 호가 레벨 깊이 (None이면 전체)

        Returns:
            총 매도 수량
        """
        levels = self.ask_levels[:depth] if depth else self.ask_levels
        return sum(level.volume for level in levels)


@dataclass
class TradeRecord:
    """
    체결 거래 기록.

    Attributes:
        timestamp: 체결 시각
        symbol: 종목 코드
        price: 체결 가격
        volume: 체결 수량
        side: 체결 방향 ('buy' 또는 'sell')
        trade_id: 거래 고유 ID (선택사항)
    """
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str
    trade_id: Optional[str] = None

    def __post_init__(self):
        """데이터 유효성 검증."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.volume <= 0:
            raise ValueError("Volume must be positive")
        if self.side not in ('buy', 'sell'):
            raise ValueError("Side must be 'buy' or 'sell'")

    @property
    def notional_value(self) -> float:
        """거래 금액 (가격 × 수량)."""
        return self.price * self.volume


@dataclass
class FeatureVector:
    """
    모델 학습/예측을 위한 특징 벡터.

    Attributes:
        timestamp: 특징 생성 시각
        symbol: 종목 코드
        features: 특징 값 배열 (NumPy array)
        feature_names: 각 특징의 이름 리스트
        label: 레이블 (학습 데이터의 경우)
        metadata: 추가 메타데이터
    """
    timestamp: datetime
    symbol: str
    features: np.ndarray
    feature_names: List[str]
    label: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """데이터 유효성 검증."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

        if len(self.features.shape) != 1:
            raise ValueError("Features must be a 1D array")

        if len(self.feature_names) != len(self.features):
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) "
                f"must match number of features ({len(self.features)})"
            )

    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환 (특징 이름을 키로 사용)."""
        return dict(zip(self.feature_names, self.features))

    def get_feature(self, name: str) -> float:
        """
        특징 이름으로 특징 값 조회.

        Args:
            name: 특징 이름

        Returns:
            특징 값

        Raises:
            KeyError: 특징 이름이 존재하지 않는 경우
        """
        try:
            idx = self.feature_names.index(name)
            return self.features[idx]
        except ValueError:
            raise KeyError(f"Feature '{name}' not found")


@dataclass
class PredictionResult:
    """
    모델 예측 결과.

    Attributes:
        timestamp: 예측 시각
        symbol: 종목 코드
        prediction: 예측 값 (상승 확률)
        confidence: 예측 신뢰도
        model_version: 사용된 모델 버전
        prediction_class: 예측 클래스 (0: 하락, 1: 상승)
        features_used: 예측에 사용된 특징 벡터 (선택사항)
    """
    timestamp: datetime
    symbol: str
    prediction: float
    confidence: float
    model_version: str
    prediction_class: int
    features_used: Optional[FeatureVector] = None

    def __post_init__(self):
        """데이터 유효성 검증."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if not 0 <= self.prediction <= 1:
            raise ValueError("Prediction must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.prediction_class not in (0, 1):
            raise ValueError("Prediction class must be 0 or 1")

    @property
    def is_bullish(self) -> bool:
        """상승 예측 여부."""
        return self.prediction_class == 1

    @property
    def is_bearish(self) -> bool:
        """하락 예측 여부."""
        return self.prediction_class == 0
