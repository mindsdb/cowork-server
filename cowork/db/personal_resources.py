"""Which resources a personal partition keeps per person.

A local deployment with COWORK_TRUST_IDENTITY_HEADERS on gives each signed-in
person a partition (cowork.principal.personal_partition_id).
COWORK_PERSONAL_RESOURCES names the resources that live in it. A resource left
off stays shared across the deployment exactly as it is with the setting off:
its rows keep a NULL org_id and its files stay under the base storage roots.
Memory left off is one store every person's turns learn into and read back,
profile included, and the server logs a warning saying so at start
(cowork.harnesses.memory.policy).
Hosted org tenancy partitions every resource and never reads the list.

``MODEL_RESOURCE`` and ``STORE_RESOURCE`` are how cowork.db.scoped finds the
resource behind a table or a filesystem store. tests/test_personal_resources.py
fails when a table with an ``org_id`` or ``created_by`` column, or a store name
passed to a scoped storage root, has no entry. It also fails when a store root
is read straight from settings (``*.root_dir``, ``*.vault_dir``,
``StreamSettings().dir`` or ``cowork_home() / ...``) without passing through a
scoped storage root and without an allowlist entry, so a new per-person store
can't ship built from the unkeyed root.
"""

from __future__ import annotations

from enum import StrEnum


class PersonalResource(StrEnum):
    CONVERSATIONS = "conversations"  # conversations, messages, turn events, running answers, conversation folders
    PROJECTS = "projects"  # projects, `general`, project files, artifacts and their publish state
    FILES = "files"  # uploads and attachments
    SKILLS = "skills"  # skill store, built-in copies, skill drafts
    CONNECTIONS = "connections"  # data vault, custom connectors, connection labels
    SCHEDULES = "schedules"  # schedules and schedule runs; a personal schedule runs as its creator
    PINS = "pins"
    MEMORY = "memory"  # global and project memory tiers
    SETTINGS = "settings"  # personal and appearance keys, and org keys the proxy allows


ALL_PERSONAL_RESOURCES: frozenset[PersonalResource] = frozenset(PersonalResource)

# A personal resource needs the resources beside it to be personal too.
# Conversations live inside projects, and attachments, pins and scheduled runs
# hang off conversations, so sharing any of those would show the rows of the
# personal one to everyone.
PERSONAL_RESOURCE_REQUIRES: dict[PersonalResource, frozenset[PersonalResource]] = {
    PersonalResource.PROJECTS: frozenset({PersonalResource.CONVERSATIONS}),
    PersonalResource.CONVERSATIONS: frozenset(
        {PersonalResource.FILES, PersonalResource.PINS, PersonalResource.SCHEDULES}
    ),
}


class PersonalResourceDependencyError(ValueError):
    """COWORK_PERSONAL_RESOURCES lists a resource without one it requires."""


def check_personal_resource_dependencies(resources: frozenset[PersonalResource]) -> None:
    """Raise PersonalResourceDependencyError naming the first missing resource."""
    for resource in sorted(resources):
        missing = PERSONAL_RESOURCE_REQUIRES.get(resource, frozenset()) - resources
        if missing:
            raise PersonalResourceDependencyError(
                f"COWORK_PERSONAL_RESOURCES lists {resource!s} without "
                f"{', '.join(sorted(missing))}, which {resource!s} requires"
            )


# Keyed by table name, not model class, so this module imports no models and
# app settings can import the enum above without a cycle.
MODEL_RESOURCE: dict[str, PersonalResource] = {
    "conversations": PersonalResource.CONVERSATIONS,
    # created_by only: a message's partition is its conversation's, and the
    # flush hook stamps its author only where conversations are personal.
    "messages": PersonalResource.CONVERSATIONS,
    # A channel installation and its bindings and events route an external chat
    # into conversations, so they follow conversations.
    "channel_installations": PersonalResource.CONVERSATIONS,
    "channel_bindings": PersonalResource.CONVERSATIONS,
    "channel_events": PersonalResource.CONVERSATIONS,
    "projects": PersonalResource.PROJECTS,
    "artifact_identities": PersonalResource.PROJECTS,
    # Authorship claims for skills, project memory and project instructions.
    # SharedResourceAccess reads and writes them only in org mode, where every
    # resource is partitioned, so this entry decides nothing in a personal scope.
    "shared_resource_attributions": PersonalResource.PROJECTS,
    "shared_resource_mutations": PersonalResource.PROJECTS,
    "files": PersonalResource.FILES,
    "schedules": PersonalResource.SCHEDULES,
    "pins": PersonalResource.PINS,
    # SettingService routes its own rows; ScopedSession leaves this table alone.
    "settings": PersonalResource.SETTINGS,
}

# None keeps a store in each person's partition in every personal scope,
# whatever COWORK_PERSONAL_RESOURCES lists.
STORE_RESOURCE: dict[str, PersonalResource | None] = {
    "projects": PersonalResource.PROJECTS,
    # Publish history (cowork.services.publish), part of an artifact's
    # publish state.
    "publish": PersonalResource.PROJECTS,
    "files": PersonalResource.FILES,
    "skills": PersonalResource.SKILLS,
    "code-skills": PersonalResource.SKILLS,
    "data-vault": PersonalResource.CONNECTIONS,
    "memory": PersonalResource.MEMORY,
    # The HOME and TMPDIR each person's agent code runs with, and the
    # folders of their credential probes (cowork.code_sandbox). Every
    # person's own whenever identity headers are on, whatever the list leaves
    # shared, memory included: one person's pip cache, temporary files or
    # candidate credentials never reach another person's cells. The operator
    # view keeps the base store.
    "sandbox": None,
}
