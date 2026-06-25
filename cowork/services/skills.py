from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from cowork.models.skill import Skill


class SkillService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_skills(self) -> list[Skill]:
        return list(self.session.exec(select(Skill).order_by(Skill.created_at.desc())).all())

    def get_skill(self, skill_id: UUID) -> Skill:
        skill = self.session.get(Skill, skill_id)
        if skill is None:
            raise ValueError(f"Skill {skill_id} not found.")
        return skill

    def record_use(self, label: str) -> Skill | None:
        """Bump the `used` counter for the skill with this label.

        Driven by Anton's `recall_skill` tool firing (surfaced as the
        `response.skill_recalled` stream event). Matching is by label —
        the same identifier Anton recalls by and the cowork skill's
        unique key. Unknown labels are a no-op (the recall may have hit a
        skill that isn't cowork-managed, or a closest-match fallback).
        Returns the updated Skill, or None if no row matched.
        """
        skill = self.session.exec(select(Skill).where(Skill.label == label)).first()
        if skill is None:
            return None
        skill.used = (skill.used or 0) + 1
        self.session.add(skill)
        self.session.commit()
        self.session.refresh(skill)
        return skill

    def create_skill(
        self,
        label: str,
        name: str,
        instructions: str,
        description: str | None = None,
        when_to_use: str | None = None,
    ) -> Skill:
        if self.session.exec(select(Skill).where(Skill.label == label)).first():
            raise ValueError(f"A skill with label '{label}' already exists.")
        if self.session.exec(select(Skill).where(Skill.name == name)).first():
            raise ValueError(f"A skill named '{name}' already exists.")
        skill = Skill(label=label, name=name, instructions=instructions, description=description, when_to_use=when_to_use)
        self.session.add(skill)
        self.session.commit()
        self.session.refresh(skill)
        return skill

    def update_skill(
        self,
        skill_id: UUID,
        label: str | None = None,
        name: str | None = None,
        description: str | None = None,
        when_to_use: str | None = None,
        instructions: str | None = None,
    ) -> Skill:
        skill = self.get_skill(skill_id)
        if label is not None:
            skill.label = label
        if name is not None:
            skill.name = name
        if description is not None:
            skill.description = description
        if when_to_use is not None:
            skill.when_to_use = when_to_use
        if instructions is not None:
            skill.instructions = instructions
        self.session.add(skill)
        self.session.commit()
        self.session.refresh(skill)
        return skill

    def delete_skill(self, skill_id: UUID) -> bool:
        skill = self.session.get(Skill, skill_id)
        if skill is None:
            return False
        self.session.delete(skill)
        self.session.commit()
        return True

    def delete_skill_by_name(self, name: str) -> bool:
        skill = self.session.exec(
            select(Skill).where((Skill.name == name) | (Skill.label == name))
        ).first()
        if skill is None:
            return False
        self.session.delete(skill)
        self.session.commit()
        return True
