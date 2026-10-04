import threading
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from src.ingestion.delta_worker import DeltaIngestionWorker

logger = logging.getLogger("SchedulerRunner")

class IngestionScheduler:
    def __init__(self, delta_worker: DeltaIngestionWorker, interval_hours: int = 1):
        self.delta_worker = delta_worker
        self.interval_hours = interval_hours
        self.scheduler = BackgroundScheduler()
        self.lock = threading.Lock()
        
        self.last_run_time: Optional[datetime] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.is_running_job: bool = False

    def _job_wrapper(self):
        """Wrapper ensuring mutually exclusive single-job execution."""
        if not self.lock.acquire(blocking=False):
            logger.warning("[Scheduler] Ingestion job already in progress. Skipping this cycle.")
            return

        try:
            self.is_running_job = True
            logger.info(f"[Scheduler] Starting scheduled delta ingestion at {datetime.now(timezone.utc).isoformat()}...")
            result = self.delta_worker.run_sync()
            self.last_run_time = datetime.now(timezone.utc)
            self.last_result = result
            logger.info(f"[Scheduler] Ingestion finished: {result}")
        except Exception as e:
            logger.exception(f"[Scheduler] Ingestion job error: {e}")
        finally:
            self.is_running_job = False
            self.lock.release()

    def start(self):
        """Starts background periodic execution on an hourly interval."""
        if not self.scheduler.running:
            trigger = IntervalTrigger(hours=self.interval_hours)
            self.scheduler.add_job(
                self._job_wrapper,
                trigger=trigger,
                id="delta_ingestion_job",
                name="Hourly Delta Document Ingestion",
                replace_existing=True
            )
            self.scheduler.start()
            logger.info(f"[Scheduler] Ingestion scheduler started with an interval of {self.interval_hours} hour(s).")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("[Scheduler] Ingestion scheduler stopped.")

    def trigger_now(self) -> Dict[str, Any]:
        """Manually triggers an immediate ingestion cycle in the calling thread or worker."""
        if not self.lock.acquire(blocking=False):
            return {"status": "in_progress", "message": "An ingestion job is already running."}

        try:
            self.is_running_job = True
            result = self.delta_worker.run_sync()
            self.last_run_time = datetime.utcnow()
            self.last_result = result
            return result
        finally:
            self.is_running_job = False
            self.lock.release()

    def get_status(self) -> Dict[str, Any]:
        job = self.scheduler.get_job("delta_ingestion_job")
        next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
        return {
            "scheduler_active": self.scheduler.running,
            "is_running_job": self.is_running_job,
            "interval_hours": self.interval_hours,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "next_run_time": next_run,
            "last_result": self.last_result
        }
