"""
Parquet 데이터 스토어.

대용량 시계열 데이터를 효율적으로 저장하고 조회합니다.
"""

import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime, date
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..exceptions import DataSaveError, DataLoadError, StorageError

logger = logging.getLogger(__name__)


class ParquetDataStore:
    """
    Parquet 기반 데이터 스토어.

    날짜 기반 파티셔닝을 지원하며 대용량 데이터를 효율적으로 저장합니다.
    """

    def __init__(self, base_path: str):
        """
        Args:
            base_path: 데이터 저장 기본 경로
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_daily_data(
        self,
        data: pd.DataFrame,
        data_date: date,
        symbol: str,
        data_type: str = "tick_data"
    ) -> None:
        """
        일별 데이터를 저장합니다.

        Args:
            data: 저장할 데이터프레임
            data_date: 데이터 날짜
            symbol: 종목 코드
            data_type: 데이터 타입 ('tick_data', 'features', 'predictions')

        Raises:
            DataSaveError: 저장 실패 시
        """
        try:
            # 파티션 경로 생성
            partition_path = self._get_partition_path(data_date, symbol, data_type)
            partition_path.parent.mkdir(parents=True, exist_ok=True)

            # Parquet 형식으로 저장
            table = pa.Table.from_pandas(data)
            pq.write_table(
                table,
                partition_path,
                compression='snappy',
                use_dictionary=True
            )

            logger.info(f"Saved {len(data)} records to {partition_path}")

        except Exception as e:
            raise DataSaveError(f"Failed to save data: {e}")

    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        symbol: str,
        data_type: str = "tick_data"
    ) -> pd.DataFrame:
        """
        날짜 범위의 데이터를 로드합니다.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            symbol: 종목 코드
            data_type: 데이터 타입

        Returns:
            결합된 데이터프레임

        Raises:
            DataLoadError: 로드 실패 시
        """
        try:
            dataframes = []
            current_date = start_date

            while current_date <= end_date:
                partition_path = self._get_partition_path(current_date, symbol, data_type)

                if partition_path.exists():
                    df = pd.read_parquet(partition_path)
                    dataframes.append(df)
                    logger.debug(f"Loaded {len(df)} records from {partition_path}")
                else:
                    logger.warning(f"File not found: {partition_path}")

                # 다음 날로 이동
                current_date = date.fromordinal(current_date.toordinal() + 1)

            if not dataframes:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()

            # 모든 데이터프레임 결합
            result = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded total {len(result)} records")

            return result

        except Exception as e:
            raise DataLoadError(f"Failed to load data: {e}")

    def load_single_day(
        self,
        data_date: date,
        symbol: str,
        data_type: str = "tick_data"
    ) -> pd.DataFrame:
        """
        특정 날짜의 데이터를 로드합니다.

        Args:
            data_date: 데이터 날짜
            symbol: 종목 코드
            data_type: 데이터 타입

        Returns:
            데이터프레임
        """
        return self.load_date_range(data_date, data_date, symbol, data_type)

    def create_partitions(
        self,
        data: pd.DataFrame,
        date_column: str = "timestamp"
    ) -> None:
        """
        데이터를 날짜별로 파티셔닝하여 저장합니다.

        Args:
            data: 전체 데이터프레임
            date_column: 날짜 컬럼명
        """
        if date_column not in data.columns:
            raise StorageError(f"Column '{date_column}' not found in data")

        # 날짜별로 그룹화
        data[date_column] = pd.to_datetime(data[date_column])
        data['date'] = data[date_column].dt.date

        for date_val, group_df in data.groupby('date'):
            # symbol 컬럼이 있으면 symbol별로도 분리
            if 'symbol' in group_df.columns:
                for symbol, symbol_df in group_df.groupby('symbol'):
                    self.save_daily_data(
                        symbol_df.drop('date', axis=1),
                        date_val,
                        symbol
                    )
            else:
                self.save_daily_data(
                    group_df.drop('date', axis=1),
                    date_val,
                    "default"
                )

    def list_available_dates(
        self,
        symbol: str,
        data_type: str = "tick_data"
    ) -> List[date]:
        """
        사용 가능한 날짜 리스트를 반환합니다.

        Args:
            symbol: 종목 코드
            data_type: 데이터 타입

        Returns:
            날짜 리스트
        """
        dates = []
        type_path = self.base_path / data_type

        if not type_path.exists():
            return dates

        # year/month/day 구조 탐색
        for year_dir in sorted(type_path.glob("year=*")):
            year = int(year_dir.name.split("=")[1])

            for month_dir in sorted(year_dir.glob("month=*")):
                month = int(month_dir.name.split("=")[1])

                for day_dir in sorted(month_dir.glob("day=*")):
                    day = int(day_dir.name.split("=")[1])

                    symbol_path = day_dir / f"symbol={symbol}" / "data.parquet"
                    if symbol_path.exists():
                        dates.append(date(year, month, day))

        return dates

    def _get_partition_path(
        self,
        data_date: date,
        symbol: str,
        data_type: str
    ) -> Path:
        """파티션 경로를 생성합니다."""
        return (
            self.base_path /
            data_type /
            f"year={data_date.year}" /
            f"month={data_date.month:02d}" /
            f"day={data_date.day:02d}" /
            f"symbol={symbol}" /
            "data.parquet"
        )

    def optimize_storage(
        self,
        data_date: date,
        symbol: str,
        data_type: str = "tick_data"
    ) -> None:
        """
        저장된 데이터를 최적화합니다 (재압축).

        Args:
            data_date: 데이터 날짜
            symbol: 종목 코드
            data_type: 데이터 타입
        """
        try:
            # 데이터 로드
            df = self.load_single_day(data_date, symbol, data_type)

            if df.empty:
                logger.warning("No data to optimize")
                return

            # 재저장 (더 나은 압축)
            partition_path = self._get_partition_path(data_date, symbol, data_type)

            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                partition_path,
                compression='zstd',  # 더 강력한 압축
                compression_level=9,
                use_dictionary=True
            )

            logger.info(f"Optimized storage for {partition_path}")

        except Exception as e:
            logger.error(f"Failed to optimize storage: {e}")

    def get_storage_stats(self, data_type: str = "tick_data") -> dict:
        """
        저장소 통계를 반환합니다.

        Args:
            data_type: 데이터 타입

        Returns:
            통계 딕셔너리
        """
        type_path = self.base_path / data_type

        if not type_path.exists():
            return {"total_files": 0, "total_size_mb": 0}

        files = list(type_path.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "base_path": str(type_path)
        }
