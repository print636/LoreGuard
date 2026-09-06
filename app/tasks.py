from celery import Celery

from .config import get_settings
from .domain import AnalysisCancelled
from .service import WorkerLeaseBusy, execute_analysis

settings = get_settings()
celery_app = Celery("loreguard", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, task_acks_late=True)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    dont_autoretry_for=(WorkerLeaseBusy, AnalysisCancelled),
    retry_backoff=True,
    max_retries=3,
)
def analyze_project(self, run_id: str) -> None:
    # Keep intermediate attempt failures non-terminal so Celery can perform its
    # bounded autoretry.  Only the last configured attempt produces a terminal
    # failure event; every attempt reuses the run's frozen input snapshot.
    try:
        execute_analysis(
            run_id,
            raise_on_failure=True,
            finalize_failure=self.request.retries >= self.max_retries,
            worker_token=self.request.id,
            raise_on_busy=True,
        )
    except AnalysisCancelled:
        # Cancellation is a requested terminal outcome, never a transient
        # worker failure.  Keep this guard even though the service normally
        # consumes the exception while finalizing the run.
        return
    except WorkerLeaseBusy as exc:
        # A late-ack redelivery can arrive while the previous worker still owns
        # a live lease.  Delay this delivery past that lease instead of
        # acknowledging it as successful or running the analysis concurrently.
        raise self.retry(exc=exc, countdown=exc.retry_after_seconds)
