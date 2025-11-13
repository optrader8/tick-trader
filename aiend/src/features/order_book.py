"""
호가창 특징 추출기.

호가창 데이터로부터 예측에 유용한 특징을 생성합니다.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from ..data.models import OrderBookSnapshot, PriceLevel
from ..exceptions import FeatureCalculationError

logger = logging.getLogger(__name__)


class OrderBookFeatureExtractor:
    """
    호가창 특징 추출기.

    호가 불균형, 압력 지표, 스프레드 등 다양한 호가창 특징을 계산합니다.
    """

    def __init__(self, depth: int = 10):
        """
        Args:
            depth: 계산에 사용할 호가 레벨 깊이
        """
        self.depth = depth

    def extract_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        호가창 스냅샷으로부터 모든 특징을 추출합니다.

        Args:
            snapshot: 호가창 스냅샷

        Returns:
            특징 딕셔너리
        """
        features = {}

        try:
            # 기본 특징
            features.update(self._extract_basic_features(snapshot))

            # 불균형 특징
            features.update(self._extract_imbalance_features(snapshot))

            # 압력 특징
            features.update(self._extract_pressure_features(snapshot))

            # 스프레드 특징
            features.update(self._extract_spread_features(snapshot))

            # 깊이 특징
            features.update(self._extract_depth_features(snapshot))

            # 미결제약정 특징
            features.update(self._extract_open_interest_features(snapshot))

        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            raise FeatureCalculationError(f"Feature extraction failed: {e}")

        return features

    def calculate_order_imbalance(
        self,
        bid_volume: float,
        ask_volume: float
    ) -> float:
        """
        호가 불균형 지표를 계산합니다.

        Formula: (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Args:
            bid_volume: 매수 호가 총량
            ask_volume: 매도 호가 총량

        Returns:
            호가 불균형 (-1 ~ 1)
        """
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    def calculate_pressure_ratio(
        self,
        bid_levels: List[PriceLevel],
        ask_levels: List[PriceLevel]
    ) -> float:
        """
        호가 압력 비율을 계산합니다.

        가격과 수량을 가중치로 사용하여 매수/매도 압력을 계산합니다.

        Args:
            bid_levels: 매수 호가 레벨 리스트
            ask_levels: 매도 호가 레벨 리스트

        Returns:
            압력 비율 (0 ~ 1)
        """
        bid_pressure = sum(level.price * level.volume for level in bid_levels)
        ask_pressure = sum(level.price * level.volume for level in ask_levels)

        total_pressure = bid_pressure + ask_pressure
        if total_pressure == 0:
            return 0.5

        return bid_pressure / total_pressure

    def calculate_spread_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        스프레드 관련 특징을 계산합니다.

        Args:
            snapshot: 호가창 스냅샷

        Returns:
            스프레드 특징 딕셔너리
        """
        features = {}

        if snapshot.spread is not None and snapshot.mid_price is not None:
            # 절대 스프레드
            features["spread_abs"] = snapshot.spread

            # 상대 스프레드 (%)
            features["spread_pct"] = (snapshot.spread / snapshot.mid_price) * 100

            # 스프레드 대 중간가 비율
            features["spread_to_mid_ratio"] = snapshot.spread / snapshot.mid_price

        return features

    def _extract_basic_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """기본 특징 추출."""
        features = {}

        if snapshot.best_bid:
            features["best_bid_price"] = snapshot.best_bid.price
            features["best_bid_volume"] = snapshot.best_bid.volume

        if snapshot.best_ask:
            features["best_ask_price"] = snapshot.best_ask.price
            features["best_ask_volume"] = snapshot.best_ask.volume

        if snapshot.mid_price:
            features["mid_price"] = snapshot.mid_price

        if snapshot.spread:
            features["spread"] = snapshot.spread

        features["total_volume"] = snapshot.total_volume

        return features

    def _extract_imbalance_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """불균형 특징 추출."""
        features = {}

        # 전체 불균형
        total_bid = snapshot.total_bid_volume(self.depth)
        total_ask = snapshot.total_ask_volume(self.depth)
        features["order_imbalance"] = self.calculate_order_imbalance(total_bid, total_ask)

        # 레벨별 불균형
        for level in range(min(5, len(snapshot.bid_levels), len(snapshot.ask_levels))):
            bid_vol = snapshot.bid_levels[level].volume if level < len(snapshot.bid_levels) else 0
            ask_vol = snapshot.ask_levels[level].volume if level < len(snapshot.ask_levels) else 0
            features[f"imbalance_level_{level}"] = self.calculate_order_imbalance(bid_vol, ask_vol)

        return features

    def _extract_pressure_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """압력 특징 추출."""
        features = {}

        bid_levels = snapshot.bid_levels[:self.depth]
        ask_levels = snapshot.ask_levels[:self.depth]

        # 압력 비율
        features["pressure_ratio"] = self.calculate_pressure_ratio(bid_levels, ask_levels)

        # 가중 평균 매수/매도 가격
        if bid_levels:
            total_bid_vol = sum(l.volume for l in bid_levels)
            if total_bid_vol > 0:
                features["weighted_bid_price"] = sum(
                    l.price * l.volume for l in bid_levels
                ) / total_bid_vol

        if ask_levels:
            total_ask_vol = sum(l.volume for l in ask_levels)
            if total_ask_vol > 0:
                features["weighted_ask_price"] = sum(
                    l.price * l.volume for l in ask_levels
                ) / total_ask_vol

        return features

    def _extract_spread_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """스프레드 특징 추출."""
        return self.calculate_spread_features(snapshot)

    def _extract_depth_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """호가창 깊이 특징 추출."""
        features = {}

        # 총 호가 수량
        features["total_bid_volume"] = snapshot.total_bid_volume(self.depth)
        features["total_ask_volume"] = snapshot.total_ask_volume(self.depth)

        # 호가 레벨 개수
        features["num_bid_levels"] = len(snapshot.bid_levels)
        features["num_ask_levels"] = len(snapshot.ask_levels)

        # 호가 수량 표준편차
        if len(snapshot.bid_levels) > 1:
            bid_volumes = [l.volume for l in snapshot.bid_levels[:self.depth]]
            features["bid_volume_std"] = np.std(bid_volumes)

        if len(snapshot.ask_levels) > 1:
            ask_volumes = [l.volume for l in snapshot.ask_levels[:self.depth]]
            features["ask_volume_std"] = np.std(ask_volumes)

        # 호가 가격 범위
        if len(snapshot.bid_levels) >= 2:
            features["bid_price_range"] = (
                snapshot.bid_levels[0].price - snapshot.bid_levels[-1].price
            )

        if len(snapshot.ask_levels) >= 2:
            features["ask_price_range"] = (
                snapshot.ask_levels[-1].price - snapshot.ask_levels[0].price
            )

        return features

    def _extract_open_interest_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """미결제약정 특징 추출."""
        features = {}

        # 매수 측 미결제약정
        bid_oi = sum(
            l.open_interest for l in snapshot.bid_levels[:self.depth]
        )
        features["bid_open_interest"] = bid_oi

        # 매도 측 미결제약정
        ask_oi = sum(
            l.open_interest for l in snapshot.ask_levels[:self.depth]
        )
        features["ask_open_interest"] = ask_oi

        # 미결제약정 불균형
        total_oi = bid_oi + ask_oi
        if total_oi > 0:
            features["oi_imbalance"] = (bid_oi - ask_oi) / total_oi

        return features

    def calculate_vpin(
        self,
        trades: List,
        volume_bucket_size: int = 10000
    ) -> float:
        """
        VPIN (Volume-synchronized Probability of Informed Trading) 계산.

        Args:
            trades: 거래 리스트
            volume_bucket_size: 볼륨 버킷 크기

        Returns:
            VPIN 값
        """
        # VPIN 계산 로직 (간략화된 버전)
        # 실제로는 더 복잡한 구현이 필요합니다
        if not trades:
            return 0.0

        buy_volume = sum(t.volume for t in trades if t.side == "buy")
        sell_volume = sum(t.volume for t in trades if t.side == "sell")
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0

        return abs(buy_volume - sell_volume) / total_volume
