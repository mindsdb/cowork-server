from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from cowork.services import screenshot_diff
from cowork.services.screenshot_diff import (
    ScreenshotDiffUnavailable,
    compare_png_files,
    render_static_html_screenshot_diff,
)


def _png(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (2, 2), color).save(path)


def test_compare_png_files_reports_changed_pixels(tmp_path: Path):
    base = tmp_path / "base.png"
    compare = tmp_path / "compare.png"
    diff = tmp_path / "diff.png"
    _png(base, (255, 255, 255, 255))
    _png(compare, (255, 0, 0, 255))

    result = compare_png_files(base, compare, diff, threshold=16)

    assert result["changedPixels"] == 4
    assert result["totalPixels"] == 4
    assert result["ratio"] == 1
    assert result["threshold"] == 16
    assert diff.is_file()


def test_compare_png_files_reports_identical_images(tmp_path: Path):
    base = tmp_path / "base.png"
    compare = tmp_path / "compare.png"
    diff = tmp_path / "diff.png"
    _png(base, (25, 50, 75, 255))
    _png(compare, (25, 50, 75, 255))

    result = compare_png_files(base, compare, diff, threshold=16)

    assert result["changedPixels"] == 0
    assert result["ratio"] == 0
    assert diff.is_file()


def test_static_html_renderer_surfaces_typed_unavailable_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path / "base.html"
    compare = tmp_path / "compare.html"
    base.write_text("<h1>Base</h1>", encoding="utf-8")
    compare.write_text("<h1>Compare</h1>", encoding="utf-8")

    async def fail_render(*args, **kwargs):
        raise ScreenshotDiffUnavailable("playwright-unavailable")

    monkeypatch.setattr(screenshot_diff, "_render_pair", fail_render)

    with pytest.raises(ScreenshotDiffUnavailable) as exc:
        render_static_html_screenshot_diff(base, compare, tmp_path / "out")

    assert exc.value.reason == "playwright-unavailable"


@pytest.mark.skipif(os.getenv("COWORK_VISUAL_DIFF_REAL") != "1", reason="real browser screenshot smoke is opt-in")
def test_real_static_html_screenshot_diff_renders_pngs(tmp_path: Path):
    base = tmp_path / "base.html"
    compare = tmp_path / "compare.html"
    base.write_text(
        """
        <!doctype html>
        <html><body style="margin:0;background:white">
        <main style="width:240px;height:140px;background:#0057b8;color:white;font:24px sans-serif">Base</main>
        </body></html>
        """,
        encoding="utf-8",
    )
    compare.write_text(
        """
        <!doctype html>
        <html><body style="margin:0;background:white">
        <main style="width:240px;height:140px;background:#c2410c;color:white;font:24px sans-serif">Compare</main>
        </body></html>
        """,
        encoding="utf-8",
    )

    result = render_static_html_screenshot_diff(
        base,
        compare,
        tmp_path / "out",
        viewport={"width": 320, "height": 200},
    )

    assert result["changedPixels"] > 0
    for key in ("basePath", "comparePath", "diffPath"):
        image_path = Path(result[key])
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            assert image.size == (320, 200)
            assert image.getbbox() is not None
