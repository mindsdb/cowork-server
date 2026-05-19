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

    def create_skill(
        self,
        name: str,
        instructions: str,
        description: str | None = None,
        when_to_use: str | None = None,
    ) -> Skill:
        existing = self.session.exec(select(Skill).where(Skill.name == name)).first()
        if existing:
            raise ValueError(f"A skill named '{name}' already exists.")
        skill = Skill(name=name, instructions=instructions, description=description, when_to_use=when_to_use)
        self.session.add(skill)
        self.session.commit()
        self.session.refresh(skill)
        return skill

    def update_skill(
        self,
        skill_id: UUID,
        name: str | None = None,
        description: str | None = None,
        when_to_use: str | None = None,
        instructions: str | None = None,
    ) -> Skill:
        skill = self.get_skill(skill_id)
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
