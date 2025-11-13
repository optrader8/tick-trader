"""
예외 계층 정의.

틱 트레이딩 시스템의 모든 사용자 정의 예외를 정의합니다.
"""


class TickAnalysisException(Exception):
    """틱 분석 시스템의 기본 예외 클래스."""

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class DataIngestionError(TickAnalysisException):
    """데이터 수집 및 파싱 중 발생하는 예외."""

    pass


class DataValidationError(DataIngestionError):
    """데이터 유효성 검증 실패 시 발생하는 예외."""

    pass


class DataParsingError(DataIngestionError):
    """데이터 파싱 중 발생하는 예외."""

    pass


class FeatureEngineeringError(TickAnalysisException):
    """특징 엔지니어링 중 발생하는 예외."""

    pass


class FeatureCalculationError(FeatureEngineeringError):
    """특징 계산 중 발생하는 예외."""

    pass


class FeatureMissingError(FeatureEngineeringError):
    """필수 특징이 누락된 경우 발생하는 예외."""

    pass


class ModelTrainingError(TickAnalysisException):
    """모델 학습 중 발생하는 예외."""

    pass


class ModelConvergenceError(ModelTrainingError):
    """모델이 수렴하지 않을 때 발생하는 예외."""

    pass


class HyperparameterError(ModelTrainingError):
    """하이퍼파라미터 설정 오류 시 발생하는 예외."""

    pass


class PredictionError(TickAnalysisException):
    """예측 수행 중 발생하는 예외."""

    pass


class ModelNotFoundError(PredictionError):
    """모델을 찾을 수 없을 때 발생하는 예외."""

    pass


class InsufficientDataError(PredictionError):
    """예측을 위한 데이터가 부족한 경우 발생하는 예외."""

    pass


class StorageError(TickAnalysisException):
    """데이터 저장 및 조회 중 발생하는 예외."""

    pass


class DataLoadError(StorageError):
    """데이터 로드 실패 시 발생하는 예외."""

    pass


class DataSaveError(StorageError):
    """데이터 저장 실패 시 발생하는 예외."""

    pass


class CacheError(StorageError):
    """캐시 관련 작업 실패 시 발생하는 예외."""

    pass


class BacktestError(TickAnalysisException):
    """백테스팅 중 발생하는 예외."""

    pass


class PerformanceCalculationError(BacktestError):
    """성능 지표 계산 중 발생하는 예외."""

    pass


class ConfigurationError(TickAnalysisException):
    """시스템 설정 오류 시 발생하는 예외."""

    pass


class StreamProcessingError(TickAnalysisException):
    """실시간 스트림 처리 중 발생하는 예외."""

    pass


class BufferOverflowError(StreamProcessingError):
    """버퍼 오버플로우 발생 시 예외."""

    pass


class LatencyExceededError(StreamProcessingError):
    """레이턴시 임계값을 초과한 경우 발생하는 예외."""

    pass
