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
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def resume_schedule(self, schedule_id: UUID) -> Schedule:
        schedule = self.get_schedule(schedule_id)
        schedule.enabled = True
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule


class ScheduleRunService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(self, schedule_id: UUID, is_manual: bool = False) -> ScheduleRun:
        run = ScheduleRun(
            schedule_id=schedule_id,
            started_at=datetime.now(timezone.utc),
            status=RunStatus.running,
            is_manual=is_manual,
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def has_running_run(self, schedule_id: UUID) -> bool:
        """
        Check if the schedule has a running non-manual (cron) run.
        The scheduler's due-check gates on has_active_run instead, so a
        manual run in flight also defers the slot (PR #181 review).
        """
        run = self.session.exec(
            select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.running)
            .where(ScheduleRun.is_manual == False)
            .limit(1)
        ).first()
        return run is not None

    def has_active_run(self, schedule_id: UUID) -> bool:
        """Any in-flight run, manual or cron. Drives the UI "running" state
        (unlike has_running_run, which only guards cron overlap)."""
        run = self.session.exec(
            select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.running)
            .limit(1)
        ).first()
        return run is not None

    def last_successful_finish(self, schedule_id: UUID) -> datetime | None:
        """When the schedule's most recent successful run (manual or cron)
        finished, or None if it never succeeded. Used by the scheduler's
        freshness guard to skip a due slot right after e.g. a manual run.
        """
        run = self.session.exec(
            select(ScheduleRun)
            .where(ScheduleRun.schedule_id == schedule_id)
            .where(ScheduleRun.status == RunStatus.success)
            .order_by(ScheduleRun.finished_at.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()
        if run is None or run.finished_at is None:
            return None
        finished = run.finished_at
        return finished if finished.tzinfo else finished.replace(tzinfo=timezone.utc)

    def set_run_conversation(self, run_id: UUID, conversation_id: UUID) -> None:
        """Attach the run's conversation as soon as it is known — before the
        turn executes — so the UI can open a run that is still in flight."""
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            return
        run.conversation_id = conversation_id
        self.session.add(run)
        self.session.commit()

    def finish_run(
        self,
        run_id: UUID,
        conversation_id: UUID | None = None,
        error: str | None = None,
        status: RunStatus | None = None,
    ) -> ScheduleRun:
        run = self.session.get(ScheduleRun, run_id)
        if run is None:
            raise ValueError("ScheduleRun not found")
        now = datetime.now(timezone.utc)
        run.finished_at = now
        started_at = run.started_at if run.started_at.tzinfo else run.started_at.replace(tzinfo=timezone.utc)
        run.duration_ms = int((now - started_at).total_seconds() * 1000)
        run.status = status or (RunStatus.failed if error else RunStatus.success)
        run.error = error
        if conversation_id is not None:
            run.conversation_id = conversation_id
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def reap_orphaned_runs(
        self,
        error: str = "Run orphaned by a server restart before it completed.",
    ) -> int:
        """Mark every run still in ``running`` as ``failed``.

        A crash/restart while a run is in flight leaves its ``ScheduleRun`` in
        ``running`` forever. Because the scheduler's due-check skips schedules
        with a running run, a single stale row wedges that schedule
        permanently. Called once on boot to release those runs. Returns the
        number of runs reaped.
        """
        runs = self.session.exec(
            select(ScheduleRun).where(ScheduleRun.status == RunStatus.running)
        ).all()
        now = datetime.now(timezone.utc)
        for run in runs:
            started_at = (
                run.started_at
                if run.started_at.tzinfo
                else run.started_at.replace(tzinfo=timezone.utc)
            )
            run.finished_at = now
            run.duration_ms = int((now - started_at).total_seconds() * 1000)
            run.status = RunStatus.failed
            run.error = error
            self.session.add(run)
        self.session.commit()
        return len(runs)

    def list_runs(self, schedule_id: UUID, limit: int = 100) -> list[ScheduleRun]:
        return list(
            self.session.exec(
                select(ScheduleRun)
                .where(ScheduleRun.schedule_id == schedule_id)
                .order_by(ScheduleRun.started_at.desc())  # type: ignore[union-attr]
                .limit(limit)
            ).all()
        )
