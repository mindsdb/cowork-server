"""Tests for the ranked search over the raw turns a summary replaced (ENG-735).

Behaviour under test — `cowork.services.history_recall`:
  * turns are grouped so the agent's own tool loop stays inside its turn
  * a multi-word query is ranked by word rarity, not word count
  * a query split between the question and the answer still matches
  * word forms match, output is capped, no match is an empty result
"""
from __future__ import annotations

from cowork.services.history_recall import (
    format_turns,
    group_turns,
    search_turns,
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _tool_use(name: str, payload: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "name": name, "input": payload}],
    }


def _tool_result(text: str) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "content": text}]}


class TestGroupTurns:
    def test_user_message_starts_a_turn(self):
        turns = group_turns([_user("one"), _assistant("a"), _user("two"), _assistant("b")])

        assert [t.number for t in turns] == [1, 2]
        assert [len(t.entries) for t in turns] == [2, 2]

    def test_tool_loop_stays_inside_its_turn(self):
        # A user row carrying tool_result blocks is the agent's own loop —
        # counting it as a new turn would split one exchange in two.
        turns = group_turns([
            _user("run the export"),
            _tool_use("scratchpad", "export()"),
            _tool_result("wrote 12 rows"),
            _assistant("done"),
        ])

        assert len(turns) == 1
        assert len(turns[0].entries) == 4

    def test_archive_starting_mid_exchange(self):
        # The archive is a slice of a conversation, so it can begin on an
        # assistant message; that must not be dropped.
        turns = group_turns([_assistant("continuing"), _user("next"), _assistant("ok")])

        assert len(turns) == 2
        assert turns[0].entries[0].text == "continuing"


class TestRanking:
    def test_rare_word_outranks_frequent_one(self):
        # Both candidates match exactly one word of the query, so coverage
        # cannot break the tie — only rarity can. "file" is in three turns and
        # cannot tell them apart; "staging" is in one and identifies it.
        messages = [
            _user("open the file please"), _assistant("opened"),
            _user("rename the file"), _assistant("renamed"),
            _user("delete the file"), _assistant("deleted"),
            _user("staging notes"), _assistant("noted"),
        ]

        found = search_turns(messages, "file staging", limit=1)

        assert [t.number for t in found] == [4]

    def test_matching_every_word_beats_matching_one(self):
        messages = [
            _user("upload the file"), _assistant("uploaded"),
            _user("upload the file to staging"), _assistant("uploaded"),
        ]

        found = search_turns(messages, "file staging", limit=1)

        assert [t.number for t in found] == [2]

    def test_query_split_between_question_and_answer(self):
        messages = [
            _user("what is the staging password"),
            _assistant("it is hunter2"),
            _user("thanks"),
            _assistant("welcome"),
        ]

        found = search_turns(messages, "staging password hunter2", limit=1)

        assert [t.number for t in found] == [1]

    def test_word_forms_match(self):
        messages = [
            _user("we finished the deployment"), _assistant("noted"),
            _user("unrelated chatter"), _assistant("noted"),
        ]

        found = search_turns(messages, "deploy", limit=1)

        assert [t.number for t in found] == [1]

    def test_no_match_is_empty(self):
        messages = [_user("hello"), _assistant("hi")]

        assert search_turns(messages, "quarterly revenue forecast") == []

    def test_empty_query_is_empty(self):
        messages = [_user("hello"), _assistant("hi")]

        assert search_turns(messages, "   ") == []

    def test_limit_is_respected(self):
        messages = []
        for i in range(5):
            messages += [_user(f"backup number {i}"), _assistant("ok")]

        assert len(search_turns(messages, "backup", limit=2)) == 2

    def test_tool_output_is_searchable(self):
        # Before compaction the model saw tool results in full, so a fact that
        # only ever appeared in one must stay recoverable.
        messages = [
            _user("check the database"),
            _tool_result("host=db-staging-7 port=5432"),
            _assistant("looks healthy"),
            _user("unrelated"), _assistant("ok"),
        ]

        found = search_turns(messages, "db-staging-7", limit=1)

        assert [t.number for t in found] == [1]


class TestFormatting:
    def test_long_tool_output_is_clipped(self):
        messages = [_user("dump it"), _tool_result("x" * 50_000)]

        rendered = format_turns(search_turns(messages, "dump", limit=1))

        assert "truncated" in rendered
        assert len(rendered) < 10_000

    def test_user_message_is_clipped_harder_than_the_answer(self):
        messages = [_user("question " + "u" * 5_000), _assistant("answer " + "s" * 5_000)]

        rendered = format_turns(search_turns(messages, "question", limit=1))

        assert rendered.count("truncated") == 2
        assert len(rendered) < 4_000

    def test_turn_number_is_reported(self):
        messages = [_user("one"), _assistant("a"), _user("two staging"), _assistant("b")]

        rendered = format_turns(search_turns(messages, "staging", limit=1))

        assert "turn 2" in rendered
