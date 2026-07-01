# Builtin skills

Skills shipped with cowork and seeded into the user's canonical skills store
(`COWORK_SKILLS_DIR`, default `~/.cowork/skills`) on startup.

Layout — one folder per skill, each holding a `SKILL.md`:

    skills_builtin/
      my-skill/
        SKILL.md

Seeding is versioned (`cowork/migrations.py::seed_builtin_skills`):

- A `Setting` sentinel `_builtin_skills_set` stores the seeded version.
- Seeding runs only when the stored version < `BUILTIN_SKILLS_VERSION`.
- Existing folders in the store are never overwritten, so a user's edits or
  deletions survive. Add new skills here and bump `BUILTIN_SKILLS_VERSION` to
  ship them in a later release.

Once seeded, builtin skills are ordinary file-backed skills — the user can edit
or delete them like any other.
