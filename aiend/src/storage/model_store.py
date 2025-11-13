"""
모델 아티팩트 저장소.

학습된 모델과 메타데이터를 버전 관리합니다.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import shutil

from ..exceptions import StorageError, ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelArtifactStore:
    """
    모델 아티팩트 저장소.

    학습된 모델을 버전별로 저장하고 관리합니다.
    """

    def __init__(self, base_path: str):
        """
        Args:
            base_path: 모델 저장 기본 경로
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_path / "registry.json"

        # 레지스트리 초기화
        self._load_registry()

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        모델을 저장합니다.

        Args:
            model: 저장할 모델 객체
            model_name: 모델 이름
            version: 모델 버전
            metadata: 모델 메타데이터

        Returns:
            모델 ID (model_name:version)
        """
        try:
            model_id = f"{model_name}:{version}"
            model_dir = self.base_path / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)

            # 모델 저장
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # 메타데이터 저장
            if metadata is None:
                metadata = {}

            metadata.update({
                "model_name": model_name,
                "version": version,
                "saved_at": datetime.now().isoformat(),
                "model_path": str(model_path)
            })

            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # 레지스트리 업데이트
            self._update_registry(model_id, metadata)

            logger.info(f"Saved model: {model_id}")
            return model_id

        except Exception as e:
            raise StorageError(f"Failed to save model: {e}")

    def load_model(self, model_id: str) -> Any:
        """
        모델을 로드합니다.

        Args:
            model_id: 모델 ID (model_name:version)

        Returns:
            모델 객체

        Raises:
            ModelNotFoundError: 모델을 찾을 수 없는 경우
        """
        try:
            if model_id not in self.registry:
                raise ModelNotFoundError(f"Model not found: {model_id}")

            model_path = Path(self.registry[model_id]["model_path"])

            if not model_path.exists():
                raise ModelNotFoundError(f"Model file not found: {model_path}")

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            logger.info(f"Loaded model: {model_id}")
            return model

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to load model: {e}")

    def load_latest_model(self, model_name: str) -> tuple[Any, str]:
        """
        특정 모델의 최신 버전을 로드합니다.

        Args:
            model_name: 모델 이름

        Returns:
            (모델 객체, 버전) 튜플
        """
        versions = self.list_model_versions(model_name)

        if not versions:
            raise ModelNotFoundError(f"No versions found for model: {model_name}")

        # 최신 버전 선택 (저장 시간 기준)
        latest = max(versions, key=lambda x: x["saved_at"])
        model_id = f"{model_name}:{latest['version']}"

        model = self.load_model(model_id)
        return model, latest["version"]

    def list_model_versions(self, model_name: str) -> List[Dict]:
        """
        특정 모델의 모든 버전을 나열합니다.

        Args:
            model_name: 모델 이름

        Returns:
            버전 정보 리스트
        """
        versions = []

        for model_id, metadata in self.registry.items():
            if metadata["model_name"] == model_name:
                versions.append(metadata)

        return sorted(versions, key=lambda x: x["saved_at"], reverse=True)

    def list_all_models(self) -> List[str]:
        """
        모든 모델 이름을 나열합니다.

        Returns:
            모델 이름 리스트
        """
        model_names = set()

        for metadata in self.registry.values():
            model_names.add(metadata["model_name"])

        return sorted(list(model_names))

    def get_model_metadata(self, model_id: str) -> Dict:
        """
        모델 메타데이터를 조회합니다.

        Args:
            model_id: 모델 ID

        Returns:
            메타데이터 딕셔너리
        """
        if model_id not in self.registry:
            raise ModelNotFoundError(f"Model not found: {model_id}")

        return self.registry[model_id].copy()

    def delete_model(self, model_id: str) -> None:
        """
        모델을 삭제합니다.

        Args:
            model_id: 모델 ID
        """
        try:
            if model_id not in self.registry:
                raise ModelNotFoundError(f"Model not found: {model_id}")

            # 모델 파일 삭제
            model_path = Path(self.registry[model_id]["model_path"])
            model_dir = model_path.parent

            if model_dir.exists():
                shutil.rmtree(model_dir)

            # 레지스트리에서 제거
            del self.registry[model_id]
            self._save_registry()

            logger.info(f"Deleted model: {model_id}")

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete model: {e}")

    def _load_registry(self) -> None:
        """레지스트리를 로드합니다."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self) -> None:
        """레지스트리를 저장합니다."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def _update_registry(self, model_id: str, metadata: Dict) -> None:
        """레지스트리를 업데이트합니다."""
        self.registry[model_id] = metadata
        self._save_registry()

    def get_storage_stats(self) -> Dict:
        """
        저장소 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        total_models = len(self.registry)
        total_size = 0

        for model_path in self.base_path.rglob("model.pkl"):
            total_size += model_path.stat().st_size

        return {
            "total_models": total_models,
            "total_size_mb": total_size / (1024 * 1024),
            "model_names": self.list_all_models()
        }
