---
name: scheduled-artifact-refresh
description: >
  Use this skill when the user wants an existing artifact, such as a dashboard,
  report or chart, kept up to date on a schedule: "keep this dashboard up to
  date every hour", "refresh this report every morning", "update these numbers
  daily". It sets up a scheduled task whose runs update the artifact's data in
  place and leave its layout alone.
---

# Keep an artifact up to date

Use this skill to turn "keep this up to date" into a scheduled task the user
doesn't have to write. Each run of the task starts a new conversation with none
of this chat's context, so everything a run needs goes into the task's prompt.

## 1. Pin down the artifact

- Find the exact artifact: its title and its slug (`list_artifacts` shows
  both). If the user says "this dashboard" and the chat has more than one
  artifact, ask which one.
- The task runs in this chat's project, and a run can only open artifacts in
  its own project. If the artifact is in another project, tell the user to ask
  from a chat in that project.
- If the artifact is a live app with its own backend that reads its data when
  someone opens it, it is already current. Say so instead of scheduling a
  refresh.

## 2. Find where its data comes from

Read the artifact and how this chat built it. For each table, chart or figure,
note:

- the connection it reads, by its exact name
- the query, table, endpoint or file, with its filters
- the time window, relative to the run: "the previous 7 days", "month to date"

If some data can't be fetched again, because it was pasted into the chat or
came from a one-off upload, tell the user which parts can't refresh. Ask
whether to schedule the rest, or to connect a source for those parts first.

## 3. Decide what a run may change

A run changes the artifact's data and nothing else:

- It may change values, table rows, chart series, totals and a "last updated"
  time if the artifact shows one.
- It keeps the layout, styling, sections, titles, chart types and any text the
  user wrote.
- It formats new values the way the artifact does now. Read the format off
  the artifact while setting up, and write it into the prompt with an example
  ("USD with two decimals, e.g. $80,621.82"). Runs that copy whatever the last
  run wrote drift.
- When the data has a new category or column, it fits it into the existing
  tables and charts rather than redesigning them.

## 4. Pick the schedule

Use the cadence the user asked for. Ask only if they gave none. Don't refresh
faster than the source changes: a daily export needs a daily task. The fastest
cadence is every 15 minutes.

## 5. Write the task prompt

Fill in this shape. Name everything explicitly.

```
Update the artifact "<title>" (slug: <slug>) with fresh data. Change only its
data. Keep its layout, styling, sections, titles, chart types and text as they
are, and don't create a new artifact.

Data sources:
- <section or chart>: <connection>, <query or table and filters>, <time window>
- ...

Formats:
- <value>: <format, with an example from the artifact>
- ...

Steps:
1. Open the artifact and read how it is built before changing anything.
2. Fetch each data source above.
3. Replace the old values with the new ones in place, in the formats above,
   and update its "last updated" time if it shows one.
4. If a source fails or returns nothing, keep that section's previous data and
   say which source failed. Don't remove sections or leave charts empty.
```

If the artifact is published, add a step: "5. Publish it again with
`publish_or_preview` and action `publish`, leaving its access as it is." That
keeps the shared link current.

## 6. Create the task

- If `manage_scheduled_task` is available, call it with action `create`, a
  title such as "Refresh <title>", the prompt, the cadence, the first run time
  and the timezone. The user confirms on the card it shows.
- If it isn't available, give the user the prompt in a code block. Tell them to
  open Scheduled Tasks, create a task, paste the prompt, pick the schedule, and
  choose this artifact's project.

Once the task exists, suggest Run now on its page to check one refresh before
the first scheduled run. Refreshes don't crowd the artifact's version history:
each one replaces the task's previous refresh.
