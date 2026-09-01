from celery import Celery

from .config import get_settings
from .service import execute_analysis

settings = get_settings()
celery_app = Celery("loreguard", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, task_acks_late=True)


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def analyze_project(self, run_id: str) -> None:
    execute_analysis(run_id)
