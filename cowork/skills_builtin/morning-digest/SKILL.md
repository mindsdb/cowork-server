---
name: morning-digest
description: >
  Use this skill for the scheduled morning digest: a once-a-day briefing
  gathered from the user's live tabs and connectors (inbox subjects, Linear
  changes, calendar) that ends with ONE parked approval for anything that
  goes out. This is a read-mostly workflow — gather, draft, park, stop.
---

# Morning Digest

Produce one calm morning briefing: what came in overnight, what moved, what
needs the user — delivered as a single draft artifact with exactly one
approval parked at the end. The digest is the product ritual: it must be
boring in the best way.

## Non-negotiables

- **Read-mostly.** Gathering is always safe: read tabs, list messages, pull
  calendar events, query connectors. Never modify anything while gathering.
- **One draft, one approval.** Everything you produce lands in ONE draft
  artifact. Exactly one `request_approval` call — the digest itself if it
  gets sent/posted anywhere, or the single most consequential action it
  surfaced. Never park a queue of proposals; the user has one click, not ten.
- **Sign-in walls are data, not failure.** A `needsAuth` tab (or a PAUSED
  auth card) means that source is simply absent today: note "Gmail needs you
  to sign in" in the digest and move on. Never attempt logins.
- **End with the appointment.** After the schedule exists, close with
  "See you at 9:00." (or the user's chosen time) — the promise matters.

## Gathering

1. `browser_tabs` — what's open, what needs sign-in.
2. From each available source, read LIGHTLY: inbox subjects + senders
   (browser_read on the mail tab or the gmail connector), Linear/project
   changes, today's calendar events. Skim, don't spelunk — a digest cites,
   it doesn't reproduce.
3. If the browser is unavailable, degrade to connectors and say so in the
   digest ("no live tabs this morning").

## Draft

One artifact (markdown or HTML per the user's artifacts conventions):
- **Needs you** — approvals waiting, sign-ins needed, anything blocked.
- **Overnight** — new mail worth reading (subject + sender, one line each),
  Linear/project movements, mentions.
- **Today** — calendar shape: meetings, gaps, the one thing to start with.
Keep it under a screen. No invented urgency, no filler.

## Approval

Park exactly one `request_approval`:
- If the digest gets delivered anywhere (posted to Slack, sent by mail):
  title "Send morning digest", draft = the digest body, action = the exact
  send. The user edits freely — their edit wins.
- If it's read-only today (nothing to send), skip the approval entirely and
  just leave the artifact — an unnecessary card is approval-fatigue
  training, and it is how rituals die.
