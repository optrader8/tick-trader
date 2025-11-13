"""
특징 엔지니어링 파이프라인.

호가창 특징과 시계열 특징을 통합하여 모델 학습용 데이터를 생성합니다.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..data.models import OrderBookSnapshot, TradeRecord, FeatureVector
from .order_book import OrderBookFeatureExtractor
from .time_series import TimeSeriesFeatureGenerator
from ..exceptions import FeatureEngineeringError, FeatureMissingError

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    특징 엔지니어링 파이프라인.

    호가창 및 시계열 특징을 추출하고 전처리하여 모델 입력을 준비합니다.
    """

    def __init__(
        self,
        orderbook_depth: int = 10,
        time_windows: Optional[List[int]] = None,
        scaling_method: str = "standard"
    ):
        """
        Args:
            orderbook_depth: 호가창 깊이
            time_windows: 시계열 윈도우 크기 리스트
            scaling_method: 스케일링 방법 ('standard', 'minmax', 'none')
        """
        self.orderbook_extractor = OrderBookFeatureExtractor(depth=orderbook_depth)
        self.timeseries_generator = TimeSeriesFeatureGenerator(windows=time_windows)
        self.scaling_method = scaling_method

        # 스케일러
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        self._feature_names = None
        self._is_fitted = False

    def transform(
        self,
        snapshots: List[OrderBookSnapshot],
        trades: Optional[List[TradeRecord]] = None
    ) -> pd.DataFrame:
        """
        호가창 스냅샷 리스트를 특징 데이터프레임으로 변환합니다.

        Args:
            snapshots: 호가창 스냅샷 리스트
            trades: 거래 기록 리스트 (선택사항)

        Returns:
            특징 데이터프레임
        """
        if not snapshots:
            raise FeatureEngineeringError("Empty snapshots list")

        # 호가창 특징 추출
        orderbook_features = []
        for snapshot in snapshots:
            features = self.orderbook_extractor.extract_features(snapshot)
            features['timestamp'] = snapshot.timestamp
            features['symbol'] = snapshot.symbol
            orderbook_features.append(features)

        df = pd.DataFrame(orderbook_features)
        df.set_index('timestamp', inplace=True)

        # 시계열 특징 추가
        if 'mid_price' in df.columns:
            df = self.timeseries_generator.create_rolling_features(df, 'mid_price')

        # 거래 데이터 특징 추가
        if trades:
            trade_features = self._extract_trade_features(trades, df.index)
            df = df.join(trade_features, how='left')

        # 결측값 처리
        df = self._handle_missing_values(df)

        return df

    def create_sliding_windows(
        self,
        data: pd.DataFrame,
        window_size: int,
        step: int = 1,
        include_labels: bool = True,
        future_steps: int = 10
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        슬라이딩 윈도우 방식으로 시퀀스 데이터를 생성합니다.

        Args:
            data: 특징 데이터프레임
            window_size: 윈도우 크기 (시퀀스 길이)
            step: 슬라이딩 스텝 크기
            include_labels: 레이블 포함 여부
            future_steps: 미래 예측 스텝 (레이블 생성용)

        Returns:
            (윈도우 데이터, 레이블) 튜플
        """
        # 숫자형 컬럼만 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data_values = data[numeric_cols].values

        windows = []
        labels = [] if include_labels else None

        for i in range(0, len(data_values) - window_size - future_steps, step):
            # 윈도우 데이터 추출
            window = data_values[i:i + window_size]
            windows.append(window)

            # 레이블 생성 (미래 가격 변화)
            if include_labels and 'mid_price' in numeric_cols:
                current_price = data_values[i + window_size - 1, numeric_cols.get_loc('mid_price')]
                future_price = data_values[i + window_size + future_steps, numeric_cols.get_loc('mid_price')]

                # 상승(1) / 하락(0)
                label = 1 if future_price > current_price else 0
                labels.append(label)

        X = np.array(windows)
        y = np.array(labels) if include_labels else None

        logger.info(f"Created {len(windows)} windows of size {window_size}")

        return X, y

    def fit_scaler(self, data: pd.DataFrame) -> None:
        """
        스케일러를 학습 데이터에 피팅합니다.

        Args:
            data: 학습 데이터프레임
        """
        if self.scaler is None:
            logger.info("No scaling method specified, skipping fit")
            return

        # 숫자형 컬럼만 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.scaler.fit(data[numeric_cols])
        self._feature_names = list(numeric_cols)
        self._is_fitted = True

        logger.info(f"Scaler fitted on {len(numeric_cols)} features")

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        특징을 스케일링합니다.

        Args:
            data: 입력 데이터프레임

        Returns:
            스케일링된 데이터프레임
        """
        if self.scaler is None:
            return data

        if not self._is_fitted:
            raise FeatureEngineeringError("Scaler not fitted. Call fit_scaler first.")

        result = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # 공통 특징만 스케일링
        common_cols = [col for col in numeric_cols if col in self._feature_names]

        if common_cols:
            result[common_cols] = self.scaler.transform(data[common_cols])

        return result

    def create_feature_vectors(
        self,
        snapshots: List[OrderBookSnapshot],
        trades: Optional[List[TradeRecord]] = None,
        labels: Optional[List[int]] = None
    ) -> List[FeatureVector]:
        """
        FeatureVector 객체 리스트를 생성합니다.

        Args:
            snapshots: 호가창 스냅샷 리스트
            trades: 거래 기록 리스트
            labels: 레이블 리스트

        Returns:
            FeatureVector 객체 리스트
        """
        # 특징 추출
        df = self.transform(snapshots, trades)

        # 스케일링
        if self._is_fitted:
            df = self.scale_features(df)

        # FeatureVector 객체 생성
        feature_vectors = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for idx, (timestamp, row) in enumerate(df.iterrows()):
            features = row[numeric_cols].values

            label = labels[idx] if labels and idx < len(labels) else None

            feature_vector = FeatureVector(
                timestamp=timestamp,
                symbol=snapshots[idx].symbol,
                features=features,
                feature_names=numeric_cols,
                label=label
            )

            feature_vectors.append(feature_vector)

        return feature_vectors

    def _extract_trade_features(
        self,
        trades: List[TradeRecord],
        time_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """거래 데이터로부터 특징을 추출합니다."""
        # 거래 데이터를 데이터프레임으로 변환
        trade_data = []
        for trade in trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'volume': trade.volume,
                'side': 1 if trade.side == 'buy' else -1
            })

        trade_df = pd.DataFrame(trade_data)
        trade_df.set_index('timestamp', inplace=True)

        # 시간 인덱스에 맞춰 리샘플링
        resampled = trade_df.resample('1s').agg({
            'price': 'last',
            'volume': 'sum',
            'side': 'sum'
        }).fillna(0)

        # 체결 강도 계산
        resampled['execution_strength'] = resampled['side']

        # VWAP 계산
        resampled['vwap'] = (
            (resampled['price'] * resampled['volume']).rolling(window=10).sum() /
            resampled['volume'].rolling(window=10).sum()
        ).fillna(method='ffill')

        # 원래 인덱스에 맞춰 리인덱싱
        resampled = resampled.reindex(time_index, method='ffill')

        return resampled

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """결측값 처리."""
        # Forward fill
        data = data.fillna(method='ffill')

        # Backward fill (첫 번째 값들)
        data = data.fillna(method='bfill')

        # 남은 결측값은 0으로
        data = data.fillna(0)

        return data

    def get_feature_names(self) -> List[str]:
        """특징 이름 리스트를 반환합니다."""
        if self._feature_names is None:
            raise FeatureMissingError("Feature names not available. Run transform first.")
        return self._feature_names.copy()

    def get_feature_importance(
        self,
        model,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        모델의 특징 중요도를 반환합니다.

        Args:
            model: 학습된 모델 (feature_importances_ 속성 필요)
            feature_names: 특징 이름 리스트

        Returns:
            특징 중요도 데이터프레임
        """
        if feature_names is None:
            feature_names = self.get_feature_names()

        if not hasattr(model, 'feature_importances_'):
            raise FeatureEngineeringError(
                "Model does not have feature_importances_ attribute"
            )

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df
