"""
백테스팅 엔진.

과거 데이터를 사용하여 전략의 성능을 검증합니다.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics
from ..exceptions import BacktestError

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    백테스팅 엔진.

    Walk-forward analysis를 지원합니다.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 백테스트 설정
        """
        self.config = config or {}
        self.metrics = PerformanceMetrics()

    def run_backtest(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prices: np.ndarray
    ) -> Dict[str, Any]:
        """
        백테스트를 실행합니다.

        Args:
            model: 학습된 모델
            X_test: 테스트 데이터
            y_test: 실제 레이블
            prices: 가격 배열

        Returns:
            백테스트 결과
        """
        logger.info("Running backtest...")

        # 예측 수행
        if hasattr(model, 'predict_class'):
            predictions = model.predict_class(X_test)
        else:
            predictions = model.predict(X_test)
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)

        # 수익률 계산
        returns = np.diff(prices) / prices[:-1]
        returns = returns[:len(predictions)]

        # 성능 메트릭 계산
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        max_dd = self.metrics.calculate_max_drawdown(returns)
        classification_metrics = self.metrics.calculate_classification_metrics(
            y_test[:len(predictions)], predictions
        )

        # PnL 곡선
        pnl_curve = self.metrics.generate_pnl_curve(predictions, returns)

        results = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_pnl": pnl_curve.iloc[-1] if len(pnl_curve) > 0 else 0,
            **classification_metrics,
            "pnl_curve": pnl_curve
        }

        logger.info(f"Backtest completed - Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%")

        return results
