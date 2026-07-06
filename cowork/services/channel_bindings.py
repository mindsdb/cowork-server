from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cowork.channels.registry import PluginRegistry, get_registry
from cowork.models.channel import ChannelBinding, ChannelSession
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.schemas.channels import (
    BindingCreateRequest,
    BindingResponse,
    BindingUpdateRequest,
)

_DEFAULT_THREAD_KEY = "__default__"


class BindingNotFoundError(Exception):
    """Binding id does not exist (→ 404)."""


class BindingConflictError(Exception):
    """A binding for the same channel/group/thread already exists (→ 409)."""


class ChannelBindingService:
    def __init__(self, session: Session, registry: PluginRegistry | None = None) -> None:
        self.session = session
        self.registry = registry if registry is not None else get_registry()

    def list(self, channel_type: str | None = None) -> list[BindingResponse]:
        stmt = select(ChannelBinding)
        if channel_type:
            stmt = stmt.where(ChannelBinding.channel_type == channel_type)
        return [self._dto(row) for row in self.session.exec(stmt).all()]

    def create(self, req: BindingCreateRequest) -> BindingResponse:
        self._validate_channel(req.channel_type)
        self._validate_trigger(req.trigger_rule.value, req.trigger_pattern)
        self._validate_links(req.anton_project_id, req.anton_conversation_id)

        thread_key = req.external_thread_id or _DEFAULT_THREAD_KEY
        if self._find(req.channel_type, req.external_group_id, thread_key) is not None:
            raise BindingConflictError(
                f"binding already exists for {req.channel_type}/{req.external_group_id}"
                + (f" thread {req.external_thread_id}" if req.external_thread_id else "")
            )

        binding = ChannelBinding(
            channel_type=req.channel_type,
            external_group_id=req.external_group_id,
            external_thread_id=req.external_thread_id,
            external_thread_key=thread_key,
            display_name=req.display_name,
            trigger_rule=req.trigger_rule.value,
            trigger_pattern=req.trigger_pattern,
            anton_project_id=req.anton_project_id,
            anton_conversation_id=req.anton_conversation_id,
        )
        self.session.add(binding)
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            raise BindingConflictError("binding already exists") from exc
        self.session.refresh(binding)
        return self._dto(binding)

    def update(self, binding_id: UUID, req: BindingUpdateRequest) -> BindingResponse:
        binding = self.session.get(ChannelBinding, binding_id)
        if binding is None:
            raise BindingNotFoundError(str(binding_id))

        provided = req.model_fields_set
        eff_rule = (
            req.trigger_rule.value
            if "trigger_rule" in provided and req.trigger_rule is not None
            else binding.trigger_rule
        )
        eff_pattern = req.trigger_pattern if "trigger_pattern" in provided else binding.trigger_pattern
        self._validate_trigger(eff_rule, eff_pattern)
        eff_project = req.anton_project_id if "anton_project_id" in provided else binding.anton_project_id
        eff_conversation = (
            req.anton_conversation_id if "anton_conversation_id" in provided else binding.anton_conversation_id
        )
        self._validate_links(eff_project, eff_conversation)

        if "display_name" in provided:
            binding.display_name = req.display_name
        if "trigger_rule" in provided and req.trigger_rule is not None:
            binding.trigger_rule = req.trigger_rule.value
        if "trigger_pattern" in provided:
            binding.trigger_pattern = req.trigger_pattern
        if "anton_project_id" in provided:
            # Detach the pinned conversation on project change, else the runtime keeps serving the old project's conversation.
            project_changed = req.anton_project_id != binding.anton_project_id
            binding.anton_project_id = req.anton_project_id
            if project_changed and "anton_conversation_id" not in provided:
                binding.anton_conversation_id = None
                self._drop_sessions(binding.id)
        if "anton_conversation_id" in provided:
            binding.anton_conversation_id = req.anton_conversation_id

        self.session.add(binding)
        self.session.commit()
        self.session.refresh(binding)
        return self._dto(binding)

    def detach_conversation(self, binding: ChannelBinding) -> None:
        """Unpin the binding's conversation so the next inbound message starts a fresh one."""
        binding.anton_conversation_id = None
        self.session.add(binding)
        self._drop_sessions(binding.id)
        self.session.commit()

    def reset_conversations(self, channel_type: str | None = None) -> int:
        """Detach bindings from their current conversation so the next inbound
        message starts a fresh one — used when the channel agent changes, so
        existing chats pick up the new agent instead of staying pinned to the
        old one. Past conversations are preserved (just no longer the active
        thread). Returns how many bindings had an active conversation.
        """
        stmt = select(ChannelBinding)
        if channel_type:
            stmt = stmt.where(ChannelBinding.channel_type == channel_type)
        reset = 0
        for binding in self.session.exec(stmt).all():
            if binding.anton_conversation_id is None:
                continue
            binding.anton_conversation_id = None
            self.session.add(binding)
            # Drop the session rows too so a fresh one is recorded against the
            # new conversation on the next message.
            self._drop_sessions(binding.id)
            reset += 1
        if reset:
            self.session.commit()
        return reset

    def delete(self, binding_id: UUID) -> bool:
        binding = self.session.get(ChannelBinding, binding_id)
        if binding is None:
            return False

        self._drop_sessions(binding_id)
        self.session.delete(binding)
        self.session.commit()
        return True

    def _drop_sessions(self, binding_id: UUID) -> None:
        for sess in self.session.exec(
            select(ChannelSession).where(ChannelSession.binding_id == binding_id)
        ).all():
            self.session.delete(sess)

    def _find(self, channel_type: str, group_id: str, thread_key: str) -> ChannelBinding | None:
        return self.session.exec(
            select(ChannelBinding).where(
                ChannelBinding.channel_type == channel_type,
                ChannelBinding.external_group_id == group_id,
                ChannelBinding.external_thread_key == thread_key,
            )
        ).first()

    def _validate_channel(self, channel_type: str) -> None:
        if self.registry.get(channel_type) is None:
            raise ValueError(f"unknown channel_type: {channel_type}")

    @staticmethod
    def _validate_trigger(rule: str, pattern: str | None) -> None:
        if rule == "regex":
            if not pattern:
                raise ValueError("trigger_pattern is required when trigger_rule is 'regex'")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid trigger_pattern regex: {exc}")

    def _validate_links(self, project_id: UUID | None, conversation_id: UUID | None) -> None:
        if project_id is not None and self.session.get(Project, project_id) is None:
            raise ValueError(f"project not found: {project_id}")
        if conversation_id is not None and self.session.get(Conversation, conversation_id) is None:
            raise ValueError(f"conversation not found: {conversation_id}")

    @staticmethod
    def _dto(binding: ChannelBinding) -> BindingResponse:
        return BindingResponse.model_validate(binding, from_attributes=True)
