from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID


class ScheduleService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_schedules(self, project_id: UUID | None = None) -> list[Schedule]:
        query = select(Schedule)
        if project_id is not None:
            query = query.where(Schedule.project_id == project_id)
        return list(self.session.exec(query).all())

    def get_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self.session.get(Schedule, schedule_id)
        if schedule is None:
            raise ValueError("Schedule not found")
        return schedule

    def create_schedule(
        self,
        title: str,
        prompt: str,
        cadence: str,
        next_run_at: datetime,
        model: str,
        timezone: str = "UTC",
        project_id: UUID | None = None,
        enabled: bool = True,
        missed_run_policy: str = "skip",
    ) -> Schedule:
        schedule = Schedule(
            title=title,
            prompt=prompt,
            cadence=cadence,
            next_run_at=next_run_at,
            model=model,
            timezone=timezone,
            project_id=project_id or GENERAL_PROJECT_ID,
            enabled=enabled,
            missed_run_policy=missed_run_policy,
        )
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def update_schedule(self, schedule_id: UUID, **kwargs) -> Schedule:
        schedule = self.get_schedule(schedule_id)
        for field, value in kwargs.items():
            if value is not None and hasattr(schedule, field):
                setattr(schedule, field, value)
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def delete_schedule(self, schedule_id: UUID) -> bool:
        schedule = self.session.get(Schedule, schedule_id)
        if schedule is None:
            return False
        for run in self.session.exec(
            select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)
        ).all():
            self.session.delete(run)
        self.session.delete(schedule)
        self.session.commit()
        return True

    def pause_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self.get_schedule(schedule_id)
        schedule.enabled = False
        schedule.health = "paused"
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def resume_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self.get_schedule(schedule_id)
        schedule.enabled = True
        # Give an auto-paused (or manually paused) task a clean slate so a
        # stale failure streak doesn't immediately re-pause it.
        schedule.consecutive_failures = 0
        schedule.last_error = None
        schedule.health = "ok"
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule


class ScheduleRunService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def run_exists(self, schedule_id: UUID, idempotency_key: str) -> bool:
        """True if a run already satisfies this occurrence (idempotency guard)."""
        return (
            self.session.exec(
                select(ScheduleRun).where(
                    ScheduleRun.schedule_id == schedule_id,
                    ScheduleRun.idempotency_key == idempotency_key,
                )
            ).first()
            is not None
        )

    def create_run(
        self,
        schedule_id: UUID,
        is_manual: bool = False,
        idempotency_key: str | None = None,
    ) -> ScheduleRun:
        run = ScheduleRun(
            schedule_id=schedule_id,
            started_at=datetime.now(timezone.utc),
            status=RunStatus.running,
            is_manual=is_manual,
            idempotency_key=idempotency_key,
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def finish_run(
        self,
        run_id: UUID,
        conversation_id: UUID | None = None,
        error: str | None = None,
        attempts: int = 1,
    ) -> ScheduleRun:
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            raise ValueError("ScheduleRun not found")
        now = datetime.now(timezone.utc)
        run.finished_at = now
        started_at = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=timezone.utc)
        run.duration_ms = int((now - started_at).total_seconds() * 1000)
        run.status = RunStatus.failed if error else RunStatus.success
        run.error = error
        run.attempts = attempts
        run.conversation_id = conversation_id
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def list_runs(self, schedule_id: UUID, limit: int = 100) -> list[ScheduleRun]:
        return list(
            self.session.exec(
                select(ScheduleRun)
                .where(ScheduleRun.schedule_id == schedule_id)
                .order_by(ScheduleRun.started_at.desc())  # type: ignore[union-attr]
                .limit(limit)
            ).all()
        )
