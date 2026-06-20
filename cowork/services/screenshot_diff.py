from __future__ import annotations

import asyncio
import threading
from pathlib import Path


class ScreenshotDiffUnavailable(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def compare_png_files(
    base_png: str | Path,
    compare_png: str | Path,
    diff_png: str | Path,
    *,
    threshold: int = 16,
) -> dict:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency-specific fallback
        raise ScreenshotDiffUnavailable("pillow-unavailable") from exc

    base = Image.open(base_png).convert("RGBA")
    compare = Image.open(compare_png).convert("RGBA")
    width = max(base.width, compare.width)
    height = max(base.height, compare.height)
    base_canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    compare_canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    base_canvas.paste(base, (0, 0))
    compare_canvas.paste(compare, (0, 0))

    changed = 0
    diff = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    diff_pixels = []
    base_bytes = base_canvas.tobytes()
    compare_bytes = compare_canvas.tobytes()
    for offset in range(0, len(base_bytes), 4):
        delta = max(abs(base_bytes[offset + index] - compare_bytes[offset + index]) for index in range(4))
        if delta > threshold:
            changed += 1
            diff_pixels.append((255, 38, 0, 190))
        else:
            diff_pixels.append((255, 255, 255, 0))
    diff.putdata(diff_pixels)
    diff_png = Path(diff_png)
    diff_png.parent.mkdir(parents=True, exist_ok=True)
    diff.save(diff_png)
    total = width * height
    return {
        "changedPixels": changed,
        "totalPixels": total,
        "ratio": (changed / total) if total else 0,
        "threshold": threshold,
        "width": width,
        "height": height,
    }


def render_static_html_screenshot_diff(
    base_html: str | Path,
    compare_html: str | Path,
    output_dir: str | Path,
    *,
    viewport: dict | None = None,
    threshold: int = 16,
    timeout_ms: int = 8000,
) -> dict:
    base_html = Path(base_html).expanduser().resolve(strict=True)
    compare_html = Path(compare_html).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    viewport = viewport or {"width": 1440, "height": 900}
    base_png = output_dir / "base.png"
    compare_png = output_dir / "compare.png"
    diff_png = output_dir / "diff.png"

    _run_async_in_thread(
        _render_pair(
            base_html,
            compare_html,
            base_png,
            compare_png,
            viewport=viewport,
            timeout_ms=timeout_ms,
        )
    )
    metrics = compare_png_files(base_png, compare_png, diff_png, threshold=threshold)
    return {
        **metrics,
        "basePath": str(base_png),
        "comparePath": str(compare_png),
        "diffPath": str(diff_png),
        "viewport": viewport,
    }


def render_url_screenshot_diff(
    base_url: str,
    compare_url: str,
    output_dir: str | Path,
    *,
    viewport: dict | None = None,
    threshold: int = 16,
    timeout_ms: int = 8000,
) -> dict:
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    viewport = viewport or {"width": 1440, "height": 900}
    base_png = output_dir / "base.png"
    compare_png = output_dir / "compare.png"
    diff_png = output_dir / "diff.png"

    _run_async_in_thread(
        _render_url_pair(
            str(base_url),
            str(compare_url),
            base_png,
            compare_png,
            viewport=viewport,
            timeout_ms=timeout_ms,
        )
    )
    metrics = compare_png_files(base_png, compare_png, diff_png, threshold=threshold)
    return {
        **metrics,
        "basePath": str(base_png),
        "comparePath": str(compare_png),
        "diffPath": str(diff_png),
        "viewport": viewport,
    }


async def _render_pair(
    base_html: Path,
    compare_html: Path,
    base_png: Path,
    compare_png: Path,
    *,
    viewport: dict,
    timeout_ms: int,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover - dependency-specific fallback
        raise ScreenshotDiffUnavailable("playwright-unavailable") from exc

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(viewport=viewport)
                await _render_page(page, base_html, base_png, timeout_ms=timeout_ms)
                await _render_page(page, compare_html, compare_png, timeout_ms=timeout_ms)
            finally:
                await browser.close()
    except ScreenshotDiffUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - browser-runtime-specific fallback
        raise ScreenshotDiffUnavailable("screenshot-render-failed") from exc


async def _render_url_pair(
    base_url: str,
    compare_url: str,
    base_png: Path,
    compare_png: Path,
    *,
    viewport: dict,
    timeout_ms: int,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover - dependency-specific fallback
        raise ScreenshotDiffUnavailable("playwright-unavailable") from exc

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page(viewport=viewport)
                await _render_url_page(page, base_url, base_png, timeout_ms=timeout_ms)
                await _render_url_page(page, compare_url, compare_png, timeout_ms=timeout_ms)
            finally:
                await browser.close()
    except ScreenshotDiffUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - browser-runtime-specific fallback
        raise ScreenshotDiffUnavailable("screenshot-render-failed") from exc


async def _render_page(page, html_path: Path, png_path: Path, *, timeout_ms: int) -> None:
    await page.goto(html_path.as_uri(), wait_until="networkidle", timeout=timeout_ms)
    await page.screenshot(path=str(png_path), full_page=True)


async def _render_url_page(page, url: str, png_path: Path, *, timeout_ms: int) -> None:
    await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
    await page.screenshot(path=str(png_path), full_page=True)


def _run_async_in_thread(coro) -> None:
    result: dict = {}

    def runner() -> None:
        try:
            asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
