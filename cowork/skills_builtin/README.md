# Builtin skills

Skills shipped with cowork and seeded into the canonical skills store. Desktop
uses `COWORK_SKILLS_DIR` (default `~/.cowork/skills`); org mode lazily seeds each
organization's `<shared>/<org_id>/skills` store on its first read or turn.

Layout — one folder per skill, each holding a `SKILL.md`:

    skills_builtin/
      my-skill/
        SKILL.md

Seeding is versioned by `BUILTIN_SKILLS_VERSION`:

- Desktop stores the version in the `Setting` sentinel `_builtin_skills_set`.
- Org mode stores it beside that organization's skills in `.builtins_seeded`,
  so a lost durable volume reseeds instead of being masked by a surviving DB
  row.
- Seeding runs only when the stored version < `BUILTIN_SKILLS_VERSION`.
- A builtin the store lacks is copied only if it is new since the store's
  version, so a builtin someone deleted stays deleted.
- A builtin the store has is replaced only when its files are exactly a version
  we shipped (`shipped.json`; anton's `stats.json` sidecar is ignored and kept).
  A copy someone edited, or one holding a link, is left as it is.
- To ship a new or changed builtin: edit it here, run
  `python scripts/builtin_skills_shipped.py` in a full clone (it reads git
  history), and bump `BUILTIN_SKILLS_VERSION`.

On desktop, seeded copies remain ordinary editable/deletable file-backed skills.
In org mode, packaged slugs are reserved and immutable for members, admins, and
owners. This is enforced from the packaged slug catalogue rather than mutable
frontmatter, so editing a copied `SKILL.md` cannot turn a builtin into a custom
skill. Project-rename link repair is the sole system consistency path that may
rewrite a packaged copy; it does not transfer authorship or record a skill edit.
