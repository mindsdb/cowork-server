"""In-frame preview shim: placement, escaping and the line-offset constant.

The shim must be the document's first script — it substitutes storage APIs that
generated pages read during top-level initialisation, and a page that reads them
before the shim runs is exactly the failure this exists to prevent.
"""

from __future__ import annotations

from cowork.services.preview_shim import SHIM_JS, inject_shim


def test_shim_goes_first_inside_head():
    out = inject_shim("<html><head><script>var a=1;</script></head><body></body></html>")
    assert out.index("anton-preview") < out.index("var a=1;")


def test_shim_survives_attributes_on_head():
    out = inject_shim('<html><head data-x="1"><title>t</title></head></html>')
    assert out.index("anton-preview") < out.index("<title>")


def test_shim_lands_after_html_open_when_there_is_no_head():
    out = inject_shim('<!doctype html><html lang="en"><body>x</body></html>')
    assert out.index("<!doctype html>") < out.index("anton-preview")
    assert out.index('<html lang="en">') < out.index("anton-preview")


def test_shim_never_precedes_the_doctype():
    # A script before the doctype puts the document into quirks mode and moves
    # the page's layout under the user.
    # Explicit escape: a literal BOM in the source is invisible to a reader.
    out = inject_shim("﻿<!DOCTYPE html><body>x</body>")
    assert out.index("<!DOCTYPE html>") < out.index("anton-preview")


def test_shim_prepends_to_a_bare_fragment():
    out = inject_shim("<div>x</div>")
    assert out.index("anton-preview") < out.index("<div>x</div>")


def test_shim_js_has_no_script_terminator():
    # A literal </script> in the payload would break out of the injected tag.
    # Same rule and same precedent as comments_layer.LAYER_JS.
    assert "</script>" not in SHIM_JS


def test_a_script_terminator_would_be_escaped_if_one_ever_appeared():
    # The guard above is a review rule; this is the mechanism that holds when a
    # future edit slips one in.
    import cowork.services.preview_shim as module

    original = module.SHIM_JS
    try:
        module.SHIM_JS = "var s = '</script>';"
        out = module.inject_shim("<html><head></head></html>")
    finally:
        module.SHIM_JS = original
    assert "<\\/script>" in out
    assert out.count("</script>") == 1


def test_line_offset_matches_the_lines_the_injection_adds():
    # Every inline script below the injection shifts by exactly this many lines;
    # the shim subtracts the same number before reporting an error's position.
    out = inject_shim("<html><head></head><body></body></html>")
    added = out.count("\n")
    assert f"LINE_OFFSET = {added}" in out


def test_reporter_is_installed_before_the_storage_guard():
    # Diagnostics must survive a real origin, where the storage half returns early.
    assert SHIM_JS.index("addEventListener('error'") < SHIM_JS.index("nativeStorageWorks")
