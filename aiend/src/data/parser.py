"""
틱데이터 파싱 컴포넌트.

원시 틱데이터를 파싱하여 표준화된 데이터 구조로 변환합니다.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd

from .models import OrderBookSnapshot, PriceLevel, TradeRecord
from ..exceptions import DataParsingError, DataValidationError

logger = logging.getLogger(__name__)


class TickDataParser:
    """
    틱데이터 파서.

    다양한 형식의 원시 틱데이터를 파싱하여
    OrderBookSnapshot과 TradeRecord 객체로 변환합니다.
    """

    def __init__(self, strict_validation: bool = True):
        """
        Args:
            strict_validation: 엄격한 유효성 검증 활성화 여부
        """
        self.strict_validation = strict_validation

    def parse_order_book(
        self,
        raw_data: Union[Dict, str, bytes],
        data_format: str = "json"
    ) -> OrderBookSnapshot:
        """
        호가창 데이터를 파싱합니다.

        Args:
            raw_data: 원시 데이터 (딕셔너리, JSON 문자열, 또는 바이트)
            data_format: 데이터 형식 ('json', 'csv', 'binary')

        Returns:
            OrderBookSnapshot 객체

        Raises:
            DataParsingError: 파싱 실패 시
            DataValidationError: 유효성 검증 실패 시
        """
        try:
            # 데이터 형식에 따라 딕셔너리로 변환
            if data_format == "json":
                data = self._parse_json(raw_data)
            elif data_format == "csv":
                data = self._parse_csv_orderbook(raw_data)
            elif data_format == "binary":
                data = self._parse_binary_orderbook(raw_data)
            else:
                raise DataParsingError(f"Unsupported format: {data_format}")

            # 데이터 유효성 검증
            self._validate_order_book_data(data)

            # OrderBookSnapshot 객체 생성
            snapshot = self._create_order_book_snapshot(data)

            return snapshot

        except (ValueError, KeyError, TypeError) as e:
            raise DataParsingError(f"Failed to parse order book data: {e}")

    def parse_trade_data(
        self,
        raw_data: Union[Dict, str, bytes],
        data_format: str = "json"
    ) -> TradeRecord:
        """
        체결 데이터를 파싱합니다.

        Args:
            raw_data: 원시 데이터
            data_format: 데이터 형식

        Returns:
            TradeRecord 객체

        Raises:
            DataParsingError: 파싱 실패 시
        """
        try:
            # 데이터 형식에 따라 딕셔너리로 변환
            if data_format == "json":
                data = self._parse_json(raw_data)
            elif data_format == "csv":
                data = self._parse_csv_trade(raw_data)
            elif data_format == "binary":
                data = self._parse_binary_trade(raw_data)
            else:
                raise DataParsingError(f"Unsupported format: {data_format}")

            # 데이터 유효성 검증
            self._validate_trade_data(data)

            # TradeRecord 객체 생성
            trade = self._create_trade_record(data)

            return trade

        except (ValueError, KeyError, TypeError) as e:
            raise DataParsingError(f"Failed to parse trade data: {e}")

    def parse_batch(
        self,
        raw_data_list: List[Union[Dict, str, bytes]],
        data_type: str = "order_book",
        data_format: str = "json"
    ) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """
        배치 데이터를 파싱합니다.

        Args:
            raw_data_list: 원시 데이터 리스트
            data_type: 데이터 타입 ('order_book' 또는 'trade')
            data_format: 데이터 형식

        Returns:
            파싱된 객체 리스트
        """
        results = []
        errors = []

        for idx, raw_data in enumerate(raw_data_list):
            try:
                if data_type == "order_book":
                    result = self.parse_order_book(raw_data, data_format)
                elif data_type == "trade":
                    result = self.parse_trade_data(raw_data, data_format)
                else:
                    raise DataParsingError(f"Unknown data type: {data_type}")

                results.append(result)

            except Exception as e:
                error_msg = f"Error parsing item {idx}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)

        if errors and self.strict_validation:
            raise DataParsingError(
                f"Batch parsing failed with {len(errors)} errors: {errors[:5]}"
            )

        logger.info(
            f"Parsed {len(results)}/{len(raw_data_list)} items "
            f"({len(errors)} errors)"
        )

        return results

    def _parse_json(self, raw_data: Union[Dict, str, bytes]) -> Dict:
        """JSON 데이터를 딕셔너리로 변환."""
        if isinstance(raw_data, dict):
            return raw_data
        elif isinstance(raw_data, (str, bytes)):
            return json.loads(raw_data)
        else:
            raise DataParsingError(f"Unexpected data type: {type(raw_data)}")

    def _parse_csv_orderbook(self, raw_data: str) -> Dict:
        """CSV 형식의 호가창 데이터를 파싱."""
        # 구현 예시 - 실제 CSV 형식에 맞게 조정 필요
        raise NotImplementedError("CSV parsing for order book not implemented")

    def _parse_binary_orderbook(self, raw_data: bytes) -> Dict:
        """바이너리 형식의 호가창 데이터를 파싱."""
        # 구현 예시 - 실제 바이너리 형식에 맞게 조정 필요
        raise NotImplementedError("Binary parsing for order book not implemented")

    def _parse_csv_trade(self, raw_data: str) -> Dict:
        """CSV 형식의 체결 데이터를 파싱."""
        raise NotImplementedError("CSV parsing for trade data not implemented")

    def _parse_binary_trade(self, raw_data: bytes) -> Dict:
        """바이너리 형식의 체결 데이터를 파싱."""
        raise NotImplementedError("Binary parsing for trade data not implemented")

    def _validate_order_book_data(self, data: Dict) -> None:
        """호가창 데이터 유효성 검증."""
        required_fields = ["timestamp", "symbol", "bids", "asks"]

        for field in required_fields:
            if field not in data:
                raise DataValidationError(f"Missing required field: {field}")

        # 매수/매도 호가가 리스트인지 확인
        if not isinstance(data["bids"], list):
            raise DataValidationError("Bids must be a list")
        if not isinstance(data["asks"], list):
            raise DataValidationError("Asks must be a list")

    def _validate_trade_data(self, data: Dict) -> None:
        """체결 데이터 유효성 검증."""
        required_fields = ["timestamp", "symbol", "price", "volume", "side"]

        for field in required_fields:
            if field not in data:
                raise DataValidationError(f"Missing required field: {field}")

        # 체결 방향 검증
        if data["side"] not in ("buy", "sell"):
            raise DataValidationError(
                f"Invalid side: {data['side']} (must be 'buy' or 'sell')"
            )

    def _create_order_book_snapshot(self, data: Dict) -> OrderBookSnapshot:
        """딕셔너리로부터 OrderBookSnapshot 객체 생성."""
        # 타임스탬프 파싱
        timestamp = self._parse_timestamp(data["timestamp"])

        # 매수 호가 레벨 파싱
        bid_levels = [
            self._create_price_level(level)
            for level in data["bids"]
        ]

        # 매도 호가 레벨 파싱
        ask_levels = [
            self._create_price_level(level)
            for level in data["asks"]
        ]

        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=data["symbol"],
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            last_price=data.get("last_price"),
            total_volume=data.get("total_volume", 0.0)
        )

    def _create_price_level(self, level_data: Union[Dict, List]) -> PriceLevel:
        """가격 레벨 객체 생성."""
        if isinstance(level_data, dict):
            return PriceLevel(
                price=float(level_data["price"]),
                volume=float(level_data["volume"]),
                open_interest=float(level_data.get("open_interest", 0.0)),
                order_count=int(level_data.get("order_count", 0))
            )
        elif isinstance(level_data, (list, tuple)):
            # [price, volume, open_interest, order_count] 형식
            return PriceLevel(
                price=float(level_data[0]),
                volume=float(level_data[1]),
                open_interest=float(level_data[2]) if len(level_data) > 2 else 0.0,
                order_count=int(level_data[3]) if len(level_data) > 3 else 0
            )
        else:
            raise DataParsingError(f"Invalid price level format: {type(level_data)}")

    def _create_trade_record(self, data: Dict) -> TradeRecord:
        """딕셔너리로부터 TradeRecord 객체 생성."""
        timestamp = self._parse_timestamp(data["timestamp"])

        return TradeRecord(
            timestamp=timestamp,
            symbol=data["symbol"],
            price=float(data["price"]),
            volume=float(data["volume"]),
            side=data["side"],
            trade_id=data.get("trade_id")
        )

    def _parse_timestamp(self, timestamp_value: Union[str, int, datetime]) -> datetime:
        """타임스탬프를 datetime 객체로 변환."""
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        elif isinstance(timestamp_value, (int, float)):
            # Unix timestamp (milliseconds)
            return datetime.fromtimestamp(timestamp_value / 1000)
        elif isinstance(timestamp_value, str):
            # ISO 형식 문자열
            try:
                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            except ValueError:
                # 다른 형식 시도
                return pd.to_datetime(timestamp_value).to_pydatetime()
        else:
            raise DataParsingError(f"Invalid timestamp format: {type(timestamp_value)}")

    def validate_data_integrity(self, data: Union[OrderBookSnapshot, TradeRecord]) -> bool:
        """
        데이터 무결성 검증.

        Args:
            data: 검증할 데이터 객체

        Returns:
            유효성 여부
        """
        try:
            if isinstance(data, OrderBookSnapshot):
                return self._validate_orderbook_integrity(data)
            elif isinstance(data, TradeRecord):
                return self._validate_trade_integrity(data)
            else:
                return False
        except Exception as e:
            logger.warning(f"Data integrity validation failed: {e}")
            return False

    def _validate_orderbook_integrity(self, snapshot: OrderBookSnapshot) -> bool:
        """호가창 데이터 무결성 검증."""
        # 매수 호가가 가격 높은 순으로 정렬되었는지 확인
        bid_prices = [level.price for level in snapshot.bid_levels]
        if bid_prices != sorted(bid_prices, reverse=True):
            logger.warning("Bid levels are not properly sorted")
            return False

        # 매도 호가가 가격 낮은 순으로 정렬되었는지 확인
        ask_prices = [level.price for level in snapshot.ask_levels]
        if ask_prices != sorted(ask_prices):
            logger.warning("Ask levels are not properly sorted")
            return False

        # 스프레드가 양수인지 확인
        if snapshot.spread is not None and snapshot.spread < 0:
            logger.warning(f"Invalid spread: {snapshot.spread}")
            return False

        return True

    def _validate_trade_integrity(self, trade: TradeRecord) -> bool:
        """체결 데이터 무결성 검증."""
        # 가격과 수량이 양수인지 확인 (이미 데이터 모델에서 검증됨)
        return trade.price > 0 and trade.volume > 0
