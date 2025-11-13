"""
성능 평가 메트릭.

금융 도메인에 특화된 평가 지표를 계산합니다.
"""

import logging
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..exceptions import PerformanceCalculationError

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    성능 평가 메트릭 계산기.

    Sharpe Ratio, Maximum Drawdown 등 금융 지표를 계산합니다.
    """

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Sharpe Ratio를 계산합니다.

        Args:
            returns: 수익률 배열
            risk_free_rate: 무위험 이자율
            periods_per_year: 연간 기간 수 (252 for daily)

        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return sharpe

    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """
        Maximum Drawdown을 계산합니다.

        Args:
            returns: 수익률 배열

        Returns:
            Maximum Drawdown (%)
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) * 100

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        분류 성능 메트릭을 계산합니다.

        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블

        Returns:
            메트릭 딕셔너리
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='binary'),
            "recall": recall_score(y_true, y_pred, average='binary'),
            "f1_score": f1_score(y_true, y_pred, average='binary')
        }

    @staticmethod
    def generate_pnl_curve(
        predictions: np.ndarray,
        actual_returns: np.ndarray
    ) -> pd.Series:
        """
        PnL 곡선을 생성합니다.

        Args:
            predictions: 예측 신호 (1: 매수, 0: 관망)
            actual_returns: 실제 수익률

        Returns:
            누적 PnL 시리즈
        """
        strategy_returns = predictions * actual_returns
        cumulative_pnl = np.cumprod(1 + strategy_returns) - 1

        return pd.Series(cumulative_pnl)
