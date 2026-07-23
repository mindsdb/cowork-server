---
name: skill-creator
description: >
  Use this skill when the user wants to create a new skill, modify or improve
  an existing skill, convert a repeated workflow into a reusable skill, clean
  up a vendor-specific skill, test whether a skill works, compare outputs with
  and without a skill, or improve a skill's trigger description.
---

# Skill Creator

Create practical standalone skills and improve existing ones without stripping
away the useful operational knowledge.

## Core principle

Do not compress a good skill into a tiny checklist. Preserve the parts that
make it work:

- trigger conditions
- task classification rules
- step-by-step workflows
- edge cases
- quality checks
- safety rules
- examples
- final verification gates

Remove only details that are actually broken for the target app: product names,
missing helper files, hidden integrations, machine-specific paths, or workflow
steps that depend on tools the app does not provide.

## Creation loop

Use this loop:

1. Decide what the skill should enable.
2. Capture when the skill should trigger.
3. Draft the skill.
4. Create realistic test prompts.
5. Run or reason through the test prompts.
6. Compare outputs against the expected behavior.
7. Rewrite the skill based on failures, gaps, and user feedback.
8. Repeat until the skill is strong enough to upload.
9. To create or edit a skill, FIRST call `create_skill_draft(name)` and write
   SKILL.md to the returned `skill_file`. Editing an existing skill: the same
   call pre-fills the folder from its saved version, so start from that. Every
   edit — new skill or iteration — goes to the draft; NEVER write into the
   project `skills/` directory (the live store).

If the user already has a draft skill, start from the draft and improve it. Do
not restart from a generic template unless the draft is unusable.

## Capture intent

Extract as much as possible from the conversation and source material:

- What should the skill help the assistant do?
- What should trigger it?
- What inputs does it expect?
- What output format or behavior should it produce?
- What tools, connections, or files does it assume?
- Which assumptions need to be removed or generalized?
- What edge cases matter?
- What quality checks prove the job is complete?

Ask only the minimum questions needed to proceed.

## Interview and research

When details are unclear, ask about:

- example user requests
- expected deliverables
- input file types
- output file types
- required tone or style
- safety constraints
- app capabilities
- success criteria
- examples of good and bad outputs

For technical or domain skills, keep domain-specific rules. Do not replace them
with vague advice.

## Skill structure

Use this standalone upload format:

```markdown
---
name: skill-name
description: >
  Specific trigger description with realistic phrases and tasks.
---

# Skill Name

## Start by classifying the task
## Workflow
## Important rules
## Quality checks
## Final checks
```

The uploadable `.skill` archive should contain exactly one root `SKILL.md`.
Keep a plain `.md` copy beside it for easy inspection.

## Description writing

The description is the trigger. Make it direct and specific.

Include:

- what the skill does
- when it should be used
- common user phrases
- file types or domains
- important exclusions when needed

Avoid descriptions so broad that the skill triggers everywhere.

## Body writing

The body should guide the assistant after the skill triggers.

Good skill bodies include:

- task routing
- concrete workflows
- best practices
- failure modes
- verification steps
- output conventions
- examples when helpful

Prefer imperative, practical instructions. Avoid long theory unless it changes
behavior.

## Progressive detail without missing files

If the target app supports only a single uploaded skill file, fold the important
reference material into `SKILL.md` instead of pointing to missing files.

If a source skill referenced external files:

- preserve the important rules from those references when available
- remove the dead link
- rewrite the dependency as a direct instruction

## Testing

Create 2 to 5 realistic test prompts, depending on skill complexity.

Good test prompts:

- sound like a real user
- cover the main workflow
- include at least one edge case
- have a clear expected behavior

Example test set:

```json
{
  "skill_name": "example-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "Turn this messy meeting transcript into a client-ready summary.",
      "expected_output": "A structured summary with decisions, open questions, and action items."
    }
  ]
}
```

If the environment does not support automated evals, reason through the prompts
manually and revise the skill where it would fail.

## Evaluation

Evaluate both qualitative and objective behavior.

Objective checks can include:

- required file exists
- required sections are present
- forbidden app names are absent
- output follows requested format
- key facts were preserved
- final QA checklist was followed

Qualitative checks can include:

- clarity
- usefulness
- tone
- completeness
- good judgment

Do not invent fake benchmark precision when only manual review was done.

## Cleaning vendor-specific skills

When adapting a skill from another app:

1. Preserve the operational knowledge.
2. Remove product names that do not belong in the target app.
3. Remove missing scripts, hidden paths, and unavailable tools.
4. Convert tool-specific commands into app-neutral workflows.
5. Keep detailed domain rules.
6. Rebuild the upload file.
7. Verify the result stands alone.

## Final checks

Before finishing a skill:

1. The description triggers on realistic user requests.
2. The body has enough detail to be useful.
3. Important source rules were preserved.
4. Broken product names, paths, integrations, scripts, and dead references were removed.
5. The skill can stand alone if uploaded by itself.
6. The `.skill` archive contains exactly one root `SKILL.md`.
7. A plain `.md` copy matches the uploaded `SKILL.md`.
8. Every edit went through `create_skill_draft` — nothing was written into the project `skills/` directory.
