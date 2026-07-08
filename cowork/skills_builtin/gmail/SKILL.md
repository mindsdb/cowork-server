---
name: gmail
description: >
  Use this skill when the user wants to work with Gmail. This includes inbox
  triage, mailbox search, thread summaries, extracting decisions and follow-ups,
  drafting replies, drafting forwards, finding attachments, and organizing
  messages with explicit confirmation before sending, archiving, deleting,
  moving, or changing labels.
---

# Gmail

Use this skill to turn noisy email threads into clear summaries, action lists,
triage buckets, and ready-to-send drafts while avoiding unwanted mailbox
changes.

## Preferred deliverables

- Thread briefs with latest status, decisions, open questions, and next actions.
- Reply or forward drafts that are ready to review, paste, or send.
- Inbox triage lists grouped by urgency, waiting state, or follow-up need.
- Search results with enough sender, subject, and timestamp context to identify
  the right thread.
- Action lists with owner, deadline, and source thread when available.

## Task routing

Classify the request:

- SEARCH: find messages, threads, senders, dates, attachments, or subjects.
- SUMMARIZE: explain a thread or mailbox slice.
- TRIAGE: group messages by urgency, waiting, FYI, newsletters, or needs reply.
- DRAFT: write a reply, reply-all, follow-up, or forward.
- EXTRACT: pull decisions, commitments, deadlines, links, attachments, or tasks.
- ORGANIZE: archive, label, delete, star, move, or mark messages.

## Search and mailbox analysis pattern

For mailbox analysis requests such as triage, follow-up detection, topic
summaries, cleanup, thread understanding, or "what matters here":

1. Prefer a Gmail-native search or connected mailbox search first.
2. Use Gmail-style query thinking for dates, senders, unread state,
   attachments, subject terms, labels, exclusions, and mailbox scope.
3. Treat search results as message-level clues until full thread context is
   read.
4. Expand specific threads when several messages appear related or when reply
   context matters.
5. Read multiple shortlisted emails when comparing urgency or extracting tasks.
6. Use IDs or message handles only when an action requires them.
7. Summarize before writing if the user's intent is ambiguous.
8. Keep analysis separate from actions such as send, archive, trash, or label
   changes unless the user explicitly asked for them.

Useful Gmail query patterns:

- `from:name@example.com`
- `to:name@example.com`
- `subject:(invoice OR contract)`
- `has:attachment`
- `is:unread`
- `older_than:30d`
- `newer_than:7d`
- `label:important`
- `in:anywhere`

Use only the query features supported by the app's Gmail connection.

## Thread summaries

Summaries should lead with the latest status, then list:

- decisions
- open questions
- action items
- deadlines
- owners
- relevant links or attachments
- what the user still owes, if anything

When a thread is long, separate confirmed facts from inferred next steps.

## Reply drafting

When drafting:

- preserve exact recipients, subject lines, names, dates, links, and commitments
  from the source thread
- match the tone of the thread unless the user asks for a different tone
- keep the draft concise unless detail is needed
- include greeting, body, and closing when appropriate
- call out missing facts separately instead of hiding uncertainty in the draft
- identify whether reply, reply-all, or forward is more appropriate when it
  matters

Do not send unless the user clearly asks to send and the recipient, subject,
and body are clear.

## Forwarding

For forwarded emails:

- include a short framing note explaining why the recipient is receiving it
- preserve the important context from the original thread
- do not over-share unrelated thread details
- include attachments only when requested or clearly needed

## Inbox triage

Use practical buckets:

- urgent or needs reply
- waiting on someone else
- scheduled or date-based
- FYI or low priority
- newsletters and automated mail
- unclear, needs user decision

When ranking urgency:

- state the search scope and coverage
- avoid saying "the only urgent email" unless the mailbox scan was broad enough
- treat read/unread status as a weak signal, not proof of attention
- mention what was excluded when the result comes from a narrowed search

## Organizing and mailbox actions

Treat send, archive, delete, trash, move, mark read/unread, star, and label
changes as explicit actions.

Before applying mailbox changes:

- confirm target messages or thread
- confirm the exact action
- confirm destructive actions especially carefully
- avoid acting when multiple threads could match

## Output conventions

- Keep summaries skimmable.
- Use sender and timestamp when multiple emails are involved.
- Keep draft replies ready to paste or send.
- If provenance matters, summarize the evidence used.
- If mailbox access is incomplete or scoped to the wrong account, say so
  clearly and ask for the right mailbox or thread.

## Example requests

- "Summarize the latest thread with Acme and tell me what I still owe them."
- "Draft a reply that confirms Tuesday works and asks for the final agenda."
- "Go through my unread inbox and group emails into urgent, waiting, and low priority."
- "Prepare a polite follow-up to the recruiter thread if I have not replied yet."

## Final checks

Before finishing:

1. Confirm the mailbox, query, or thread scope used.
2. Verify summaries and drafts are grounded in the messages.
3. Confirm no send/archive/delete/label action was taken without clear intent.
4. Mention uncertainty when search scope, account access, or thread identity is unclear.
