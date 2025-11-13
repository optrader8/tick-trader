"""
AI Engine service for managing aiend integration.
"""

import asyncio
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from app.core.config import settings
from app.services.redis import redis_service

logger = logging.getLogger(__name__)


class AIEngineService:
    """Service for managing AI engine (aiend) integration."""

    def __init__(self):
        self.aiend_path = settings.aiend_directory
        self.python_path = self._get_python_path()
        self.running_jobs: Dict[str, asyncio.subprocess.Process] = {}

    def _get_python_path(self) -> str:
        """Get Python executable path."""
        venv_python = Path(settings.PYTHON_VENV_PATH) / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        return "python3"

    async def start_analysis(
        self,
        dataset_path: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Start analysis job."""
        job_id = str(uuid4())

        # Create job configuration
        job_config = {
            'job_id': job_id,
            'dataset_path': dataset_path,
            'model_type': model_type,
            'config': config,
            'started_at': datetime.now().isoformat()
        }

        # Enqueue job
        await redis_service.enqueue_job(job_config)

        # Start job processing
        asyncio.create_task(self._process_job(job_config))

        logger.info(f"Analysis job {job_id} started")
        return job_id

    async def _process_job(self, job_config: Dict[str, Any]):
        """Process analysis job."""
        job_id = job_config['job_id']

        try:
            # Update status to running
            await redis_service.set_job_status(job_id, {
                'status': 'running',
                'progress': 0,
                'current_step': 'Initializing'
            })

            # Create config file for aiend
            config_path = self._create_config_file(job_config)

            # Build command
            script_path = self.aiend_path / "scripts" / "run_analysis.py"
            cmd = [
                self.python_path,
                str(script_path),
                "--config", str(config_path),
                "--job-id", job_id
            ]

            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.aiend_path)
            )

            self.running_jobs[job_id] = process

            # Monitor process output
            await self._monitor_process(job_id, process)

            # Wait for completion
            await process.wait()

            if process.returncode == 0:
                await redis_service.set_job_status(job_id, {
                    'status': 'completed',
                    'progress': 100,
                    'current_step': 'Done'
                })
                logger.info(f"Job {job_id} completed successfully")
            else:
                stderr = await process.stderr.read()
                error_msg = stderr.decode()
                await redis_service.set_job_status(job_id, {
                    'status': 'failed',
                    'progress': 0,
                    'error': error_msg
                })
                logger.error(f"Job {job_id} failed: {error_msg}")

        except Exception as e:
            logger.error(f"Job {job_id} error: {e}")
            await redis_service.set_job_status(job_id, {
                'status': 'failed',
                'progress': 0,
                'error': str(e)
            })

        finally:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

    async def _monitor_process(
        self,
        job_id: str,
        process: asyncio.subprocess.Process
    ):
        """Monitor process output and update status."""
        async for line in process.stdout:
            line_str = line.decode().strip()
            if line_str:
                # Log output
                await redis_service.add_job_log(job_id, line_str)

                # Parse progress updates
                if line_str.startswith('[PROGRESS]'):
                    try:
                        progress_data = json.loads(line_str[10:])
                        await redis_service.set_job_status(job_id, progress_data)
                    except json.JSONDecodeError:
                        pass

    def _create_config_file(self, job_config: Dict[str, Any]) -> Path:
        """Create configuration file for aiend."""
        config_dir = self.aiend_path / "config" / "jobs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / f"{job_config['job_id']}.yaml"

        # Convert to aiend config format
        aiend_config = {
            'data': {
                'raw_path': job_config['dataset_path']
            },
            'training': {
                job_config['model_type']: job_config['config']
            }
        }

        # Write config file
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(aiend_config, f)

        return config_path

    async def cancel_job(self, job_id: str):
        """Cancel running job."""
        if job_id in self.running_jobs:
            process = self.running_jobs[job_id]
            process.terminate()
            await asyncio.sleep(1)
            if process.returncode is None:
                process.kill()

            await redis_service.set_job_status(job_id, {
                'status': 'cancelled',
                'progress': 0
            })
            logger.info(f"Job {job_id} cancelled")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        return await redis_service.get_job_status(job_id)


# Global instance
ai_engine_service = AIEngineService()
