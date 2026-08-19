"""The surface kwarg must never be able to fail a turn (ENG-1459).

cowork-server pins anton to a *branch*, so the installed anton can legitimately
predate anton's half of this change. Passing an unexpected keyword to
`ChatSessionConfig` would then raise on **every** turn — an unacceptable
outcome for a telemetry field, and the exact version-skew shape that has bitten
this seam before (a settings key the server 400s on because the renderer leads
the server).
"""

import dataclasses

from cowork.harnesses.anton_harness.harness import _surface_kwarg


@dataclasses.dataclass
class _NewAnton:
    harness: str | None = None
    surface: str | None = None


@dataclasses.dataclass
class _OldAnton:
    """An anton from before ENG-1459 — no `surface` field."""

    harness: str | None = None


class TestVersionSkew:
    def test_an_old_anton_gets_no_kwarg_at_all(self):
        # The load-bearing case: {} keeps the call signature valid instead of
        # raising TypeError on every turn.
        assert _surface_kwarg(_OldAnton) == {}

    def test_the_kwarg_is_actually_accepted_by_the_config_it_targets(self):
        # Proves the guard is not vacuously returning {} for everything — the
        # result must construct cleanly against a config that has the field.
        kwargs = _surface_kwarg(_NewAnton)
        assert set(kwargs) <= {"surface"}
        _NewAnton(harness="anton", **kwargs)  # must not raise

    def test_a_current_anton_gets_the_resolved_surface(self, monkeypatch):
        monkeypatch.setattr(
            "cowork.build_info.surface", lambda: "web"
        )
        assert _surface_kwarg(_NewAnton) == {"surface": "web"}

    def test_an_unresolvable_surface_is_omitted_rather_than_guessed(self, monkeypatch):
        # Absent reads as honestly unknown; a guess silently joins the
        # population it is being compared against.
        monkeypatch.setattr("cowork.build_info.surface", lambda: None)
        assert _surface_kwarg(_NewAnton) == {}

    def test_a_broken_resolver_never_propagates(self, monkeypatch):
        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr("cowork.build_info.surface", _boom)
        assert _surface_kwarg(_NewAnton) == {}
