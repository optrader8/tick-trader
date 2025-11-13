"""
시계열 특징 생성기.

시간에 따른 특징과 기술적 지표를 생성합니다.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..data.models import TradeRecord, OrderBookSnapshot
from ..exceptions import FeatureCalculationError

logger = logging.getLogger(__name__)


class TimeSeriesFeatureGenerator:
    """
    시계열 특징 생성기.

    VWAP, 체결강도, 롤링 통계 등 시간 기반 특징을 생성합니다.
    """

    def __init__(self, windows: Optional[List[int]] = None):
        """
        Args:
            windows: 롤링 윈도우 크기 리스트 (기본값: [5, 10, 20, 50])
        """
        self.windows = windows or [5, 10, 20, 50]

    def create_rolling_features(
        self,
        data: pd.DataFrame,
        value_col: str = "price"
    ) -> pd.DataFrame:
        """
        롤링 윈도우 기반 통계 특징을 생성합니다.

        Args:
            data: 입력 데이터프레임
            value_col: 계산할 값 컬럼명

        Returns:
            특징이 추가된 데이터프레임
        """
        result = data.copy()

        for window in self.windows:
            # 롤링 평균
            result[f"{value_col}_ma_{window}"] = (
                result[value_col].rolling(window=window).mean()
            )

            # 롤링 표준편차
            result[f"{value_col}_std_{window}"] = (
                result[value_col].rolling(window=window).std()
            )

            # 롤링 최소값
            result[f"{value_col}_min_{window}"] = (
                result[value_col].rolling(window=window).min()
            )

            # 롤링 최대값
            result[f"{value_col}_max_{window}"] = (
                result[value_col].rolling(window=window).max()
            )

            # 볼린저 밴드 (평균 ± 2*표준편차)
            if "volume" in data.columns:
                result[f"bb_upper_{window}"] = (
                    result[f"{value_col}_ma_{window}"] +
                    2 * result[f"{value_col}_std_{window}"]
                )
                result[f"bb_lower_{window}"] = (
                    result[f"{value_col}_ma_{window}"] -
                    2 * result[f"{value_col}_std_{window}"]
                )

        return result

    def calculate_vwap(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        VWAP (Volume Weighted Average Price) 계산.

        Args:
            prices: 가격 배열
            volumes: 수량 배열

        Returns:
            VWAP 값
        """
        if len(prices) == 0 or len(volumes) == 0:
            return 0.0

        if len(prices) != len(volumes):
            raise FeatureCalculationError(
                "Prices and volumes must have the same length"
            )

        total_volume = np.sum(volumes)
        if total_volume == 0:
            return 0.0

        return np.sum(prices * volumes) / total_volume

    def calculate_trade_intensity(
        self,
        trades: List[TradeRecord],
        time_window: int = 60
    ) -> Dict[str, float]:
        """
        거래 강도를 계산합니다.

        Args:
            trades: 거래 기록 리스트
            time_window: 시간 윈도우 (초)

        Returns:
            거래 강도 특징 딕셔너리
        """
        if not trades:
            return {
                "trade_count": 0,
                "total_volume": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "trade_intensity": 0.0
            }

        # 매수/매도 수량 계산
        buy_volume = sum(t.volume for t in trades if t.side == "buy")
        sell_volume = sum(t.volume for t in trades if t.side == "sell")
        total_volume = buy_volume + sell_volume

        # 거래 강도 = 거래량 / 시간
        trade_intensity = total_volume / time_window if time_window > 0 else 0

        return {
            "trade_count": len(trades),
            "total_volume": total_volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "trade_intensity": trade_intensity,
            "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else 0
        }

    def calculate_execution_strength(
        self,
        trades: List[TradeRecord]
    ) -> float:
        """
        체결 강도를 계산합니다.

        Formula: (buy_volume - sell_volume) / total_volume

        Args:
            trades: 거래 기록 리스트

        Returns:
            체결 강도 (-1 ~ 1)
        """
        if not trades:
            return 0.0

        buy_volume = sum(t.volume for t in trades if t.side == "buy")
        sell_volume = sum(t.volume for t in trades if t.side == "sell")
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0

        return (buy_volume - sell_volume) / total_volume

    def calculate_oi_change_rate(
        self,
        snapshots: List[OrderBookSnapshot],
        period: int = 10
    ) -> float:
        """
        미결제약정 변화율을 계산합니다.

        Args:
            snapshots: 호가창 스냅샷 리스트
            period: 비교 기간

        Returns:
            변화율 (%)
        """
        if len(snapshots) < period + 1:
            return 0.0

        # 현재와 과거의 미결제약정 계산
        current_oi = sum(
            level.open_interest
            for level in snapshots[-1].bid_levels + snapshots[-1].ask_levels
        )

        past_oi = sum(
            level.open_interest
            for level in snapshots[-period-1].bid_levels + snapshots[-period-1].ask_levels
        )

        if past_oi == 0:
            return 0.0

        return ((current_oi - past_oi) / past_oi) * 100

    def generate_technical_indicators(
        self,
        price_series: pd.Series
    ) -> Dict[str, float]:
        """
        기술적 지표를 생성합니다.

        Args:
            price_series: 가격 시계열

        Returns:
            기술적 지표 딕셔너리
        """
        indicators = {}

        if len(price_series) < 2:
            return indicators

        # RSI (Relative Strength Index)
        indicators["rsi"] = self._calculate_rsi(price_series)

        # MACD (Moving Average Convergence Divergence)
        macd_dict = self._calculate_macd(price_series)
        indicators.update(macd_dict)

        # 모멘텀
        indicators["momentum"] = self._calculate_momentum(price_series)

        # 변동성
        indicators["volatility"] = price_series.std()

        # 가격 변화율
        indicators["price_change_pct"] = (
            (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0] * 100
        )

        return indicators

    def _calculate_rsi(
        self,
        price_series: pd.Series,
        period: int = 14
    ) -> float:
        """RSI (Relative Strength Index) 계산."""
        if len(price_series) < period + 1:
            return 50.0  # 중립값

        # 가격 변화
        delta = price_series.diff()

        # 상승/하락 분리
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # 평균 계산
        avg_gain = gain.rolling(window=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        price_series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, float]:
        """MACD (Moving Average Convergence Divergence) 계산."""
        if len(price_series) < slow_period:
            return {
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_histogram": 0.0
            }

        # EMA 계산
        ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()

        # MACD 라인
        macd_line = ema_fast - ema_slow

        # 시그널 라인
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # 히스토그램
        histogram = macd_line - signal_line

        return {
            "macd": macd_line.iloc[-1],
            "macd_signal": signal_line.iloc[-1],
            "macd_histogram": histogram.iloc[-1]
        }

    def _calculate_momentum(
        self,
        price_series: pd.Series,
        period: int = 10
    ) -> float:
        """모멘텀 계산."""
        if len(price_series) < period + 1:
            return 0.0

        return price_series.iloc[-1] - price_series.iloc[-period-1]

    def calculate_price_impact(
        self,
        trades: List[TradeRecord],
        snapshots: List[OrderBookSnapshot]
    ) -> float:
        """
        가격 영향력을 계산합니다.

        Args:
            trades: 거래 리스트
            snapshots: 호가창 스냅샷 리스트

        Returns:
            가격 영향력
        """
        if len(trades) < 2 or len(snapshots) < 2:
            return 0.0

        # 첫 거래와 마지막 거래 사이의 가격 변화
        price_change = trades[-1].price - trades[0].price

        # 총 거래량
        total_volume = sum(t.volume for t in trades)

        if total_volume == 0:
            return 0.0

        # 가격 영향력 = 가격 변화 / 거래량
        return price_change / total_volume

    def aggregate_features(
        self,
        data: pd.DataFrame,
        time_intervals: List[str] = None
    ) -> pd.DataFrame:
        """
        시간 간격별로 특징을 집계합니다.

        Args:
            data: 입력 데이터프레임 (timestamp 인덱스 필요)
            time_intervals: 집계 간격 리스트 (예: ['1min', '5min', '30min'])

        Returns:
            집계된 데이터프레임
        """
        if time_intervals is None:
            time_intervals = ['1min', '5min', '30min']

        results = {}

        for interval in time_intervals:
            # 리샘플링 및 집계
            aggregated = data.resample(interval).agg({
                'price': ['first', 'last', 'min', 'max', 'mean'],
                'volume': 'sum'
            })

            # 컬럼명 평탄화
            aggregated.columns = [
                f"{col[0]}_{col[1]}_{interval}"
                for col in aggregated.columns
            ]

            results[interval] = aggregated

        # 모든 간격의 데이터 병합
        combined = pd.concat(results.values(), axis=1)

        return combined
