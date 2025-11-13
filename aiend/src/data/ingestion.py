"""
데이터 수집 파이프라인.

대용량 틱데이터를 효율적으로 처리하는 배치 처리 파이프라인입니다.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union, Callable
from collections import deque
import pandas as pd

from .models import OrderBookSnapshot, TradeRecord
from .parser import TickDataParser
from ..exceptions import DataIngestionError, DataLoadError

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    데이터 수집 파이프라인.

    대용량 틱데이터를 메모리 효율적으로 처리합니다.
    """

    def __init__(
        self,
        parser: Optional[TickDataParser] = None,
        batch_size: int = 10000,
        error_threshold: float = 0.1
    ):
        """
        Args:
            parser: 데이터 파서 (None이면 기본 파서 사용)
            batch_size: 배치 처리 크기
            error_threshold: 허용 가능한 에러 비율 (0.1 = 10%)
        """
        self.parser = parser or TickDataParser(strict_validation=False)
        self.batch_size = batch_size
        self.error_threshold = error_threshold
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "error_rate": 0.0
        }

    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        data_type: str = "order_book"
    ) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """
        파일 배치를 처리합니다.

        Args:
            file_paths: 처리할 파일 경로 리스트
            data_type: 데이터 타입 ('order_book' 또는 'trade')

        Returns:
            파싱된 데이터 리스트

        Raises:
            DataIngestionError: 처리 실패 시
        """
        all_results = []

        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                results = self._process_single_file(file_path, data_type)
                all_results.extend(results)

            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
                self.stats["failed"] += 1

                # 에러율이 임계값을 초과하면 중단
                if self._check_error_threshold():
                    raise DataIngestionError(
                        f"Error rate exceeded threshold: {self.stats['error_rate']:.2%}"
                    )

        logger.info(
            f"Batch processing complete: {len(all_results)} records processed"
        )
        return all_results

    def handle_streaming_data(
        self,
        stream: Iterator[dict],
        data_type: str = "order_book",
        callback: Optional[Callable] = None
    ) -> Iterator[Union[OrderBookSnapshot, TradeRecord]]:
        """
        스트리밍 데이터를 처리합니다 (제너레이터 패턴).

        Args:
            stream: 데이터 스트림 이터레이터
            data_type: 데이터 타입
            callback: 각 데이터 처리 후 호출할 콜백 함수

        Yields:
            파싱된 데이터 객체
        """
        buffer = []

        for raw_data in stream:
            try:
                # 데이터 파싱
                if data_type == "order_book":
                    parsed = self.parser.parse_order_book(raw_data)
                elif data_type == "trade":
                    parsed = self.parser.parse_trade_data(raw_data)
                else:
                    raise DataIngestionError(f"Unknown data type: {data_type}")

                buffer.append(parsed)
                self.stats["successful"] += 1

                # 배치 크기에 도달하면 yield
                if len(buffer) >= self.batch_size:
                    for item in buffer:
                        if callback:
                            callback(item)
                        yield item
                    buffer.clear()

            except Exception as e:
                logger.warning(f"Failed to parse streaming data: {e}")
                self.stats["failed"] += 1
                continue

        # 남은 버퍼 데이터 처리
        for item in buffer:
            if callback:
                callback(item)
            yield item

    def process_with_chunking(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 50000
    ) -> Iterator[pd.DataFrame]:
        """
        대용량 파일을 청크 단위로 처리합니다.

        Args:
            file_path: 파일 경로
            chunk_size: 청크 크기 (행 수)

        Yields:
            처리된 데이터프레임 청크
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        try:
            # 파일 형식에 따라 청크 읽기
            if file_path.suffix == ".csv":
                chunks = pd.read_csv(file_path, chunksize=chunk_size)
            elif file_path.suffix == ".parquet":
                # Parquet는 전체 읽기 후 청크 분할
                df = pd.read_parquet(file_path)
                chunks = [
                    df[i:i + chunk_size]
                    for i in range(0, len(df), chunk_size)
                ]
            else:
                raise DataLoadError(f"Unsupported file format: {file_path.suffix}")

            for chunk_idx, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {chunk_idx}, size: {len(chunk)}")
                self.stats["total_processed"] += len(chunk)
                yield chunk

        except Exception as e:
            raise DataIngestionError(f"Failed to process file with chunking: {e}")

    def _process_single_file(
        self,
        file_path: Union[str, Path],
        data_type: str
    ) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """단일 파일을 처리합니다."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        # 파일 형식에 따라 처리
        if file_path.suffix == ".json":
            return self._process_json_file(file_path, data_type)
        elif file_path.suffix == ".csv":
            return self._process_csv_file(file_path, data_type)
        elif file_path.suffix == ".parquet":
            return self._process_parquet_file(file_path, data_type)
        else:
            raise DataLoadError(f"Unsupported file format: {file_path.suffix}")

    def _process_json_file(
        self,
        file_path: Path,
        data_type: str
    ) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """JSON 파일 처리."""
        import json

        with open(file_path, 'r') as f:
            data_list = json.load(f)

        if not isinstance(data_list, list):
            data_list = [data_list]

        return self.parser.parse_batch(data_list, data_type, "json")

    def _process_csv_file(
        self,
        file_path: Path,
        data_type: str
    ) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """CSV 파일 처리."""
        df = pd.read_csv(file_path)
        data_list = df.to_dict('records')

        return self.parser.parse_batch(data_list, data_type, "json")

    def _process_parquet_file(
        self,
        file_path: Path,
        data_type: str
    ) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """Parquet 파일 처리."""
        df = pd.read_parquet(file_path)
        data_list = df.to_dict('records')

        return self.parser.parse_batch(data_list, data_type, "json")

    def _check_error_threshold(self) -> bool:
        """에러율이 임계값을 초과했는지 확인."""
        total = self.stats["successful"] + self.stats["failed"]
        if total == 0:
            return False

        self.stats["error_rate"] = self.stats["failed"] / total
        return self.stats["error_rate"] > self.error_threshold

    def get_statistics(self) -> dict:
        """처리 통계를 반환합니다."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """처리 통계를 초기화합니다."""
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "error_rate": 0.0
        }


class RealTimeDataBuffer:
    """
    실시간 데이터 버퍼.

    슬라이딩 윈도우 방식으로 최근 데이터를 유지합니다.
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 버퍼 최대 크기
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, data: Union[OrderBookSnapshot, TradeRecord]) -> None:
        """버퍼에 데이터 추가."""
        self.buffer.append(data)

    def get_recent(self, n: int) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """
        최근 N개의 데이터를 반환합니다.

        Args:
            n: 가져올 데이터 개수

        Returns:
            최근 데이터 리스트
        """
        return list(self.buffer)[-n:]

    def get_all(self) -> List[Union[OrderBookSnapshot, TradeRecord]]:
        """버퍼의 모든 데이터를 반환합니다."""
        return list(self.buffer)

    def clear(self) -> None:
        """버퍼를 비웁니다."""
        self.buffer.clear()

    def is_full(self) -> bool:
        """버퍼가 가득 찼는지 확인합니다."""
        return len(self.buffer) >= self.max_size

    def __len__(self) -> int:
        """버퍼의 현재 크기를 반환합니다."""
        return len(self.buffer)
