r"""The first danger zone must be visible while it is being drawn.

A real operator opens the app, presses Start camera, picks Box, and drags. The
shape has to be on the picture from the first pixel of the drag, then stay there
after the release and after Save zone - not only turn up once a second zone has
been started. Every zone also carries its own name, the same name as the chip
under the video, and the machine is named once in the card header.

Nothing is mocked: the real Vite server, the real React UI and Playwright, with
a video built from ultralytics' bundled bus.jpg played by the real pipeline. The
backend runs on a spare port with a throwaway database, and the check starts
from zero saved zones so the "first zone" case is the real case.

Each tool is tried as the very first zone, with the zones cleared in between,
and every moment gets a screenshot:

    1  Box          the rubber band while dragging, then the shape and its tag
    2  Polygon      the dashed outline while the corners go down
    3  Circle       the rubber band while dragging, then the shape and its tag
    4  Two zones    a tag on each zone, in that zone's own corner

Usage (needs Node.js for Vite, and Playwright with a Chromium build):

    py -3.13 tests/e2e/first_zone_visible.py --video path\to\clip.mp4 --shots out\
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_workflow import (  # noqa: E402
    Backend,
    DevServer,
    Report,
    Ui,
    build_clip,
    free_port,
)

# where each tool is drawn, in frame coordinates: three different parts of the
# picture, so a tag sitting over one zone cannot be mistaken for another's
BOX_ZONE = ((90, 70), (300, 250))
POLYGON_ZONE = ((60, 300), (250, 300), (250, 430), (60, 430))
CIRCLE_ZONE = ((400, 300), (560, 430))

# what the overlay is painting right now
OVERLAY = """
() => ({
  svg: document.querySelectorAll('[data-testid="zone-stage"]').length,
  shapes: document.querySelectorAll('[data-testid^="zone-shape-"]').length,
  band: document.querySelectorAll('.rubber-band').length,
  band_shape: (document.querySelector('.rubber-band') || {}).tagName || '',
  band_box: (() => {
    const node = document.querySelector('.rubber-band');
    if (!node) return null;
    const box = node.getBoundingClientRect();
    return { w: Math.round(box.width), h: Math.round(box.height) };
  })(),
  preview: document.querySelectorAll('.zone-preview').length,
  preview_points: ((document.querySelector('.zone-preview') || {}).getAttribute
    ? document.querySelector('.zone-preview').getAttribute('points') : '') || '',
  tags: Array.from(document.querySelectorAll('[data-testid^="zone-tag-"]'))
    .map((node) => node.getAttribute('data-testid')),
  tag_text: Array.from(document.querySelectorAll('.zone-tag-text')).map((n) => n.textContent),
  tag_at: Array.from(document.querySelectorAll('[data-testid^="zone-tag-"]'))
    .map((n) => n.getAttribute('transform')),
  chips: Array.from(document.querySelectorAll('.zone-chips .chip')).map((n) => n.textContent.trim()),
  machine: Array.from(document.querySelectorAll('[data-testid="machine-name"]'))
    .map((n) => n.textContent.trim()),
})
"""


def overlay(page) -> dict:
    return page.evaluate(OVERLAY)


# where every tag and every shape actually ended up on screen, so "the label is
# on its own zone" is a measurement and not an opinion
PLACEMENT = """
() => {
  const rect = (node) => {
    const box = node.getBoundingClientRect();
    return { x: box.x, y: box.y, w: box.width, h: box.height };
  };
  // a zone is drawn with a 2px stroke and the stage carries a 1px border, so the
  // two boxes are allowed to be a pixel or two apart; a tag on the wrong zone is
  // hundreds of pixels out
  const SLOP = 2;
  const inside = (a, b) => a.x >= b.x - SLOP && a.y >= b.y - SLOP
    && a.x + a.w <= b.x + b.w + SLOP && a.y + a.h <= b.y + b.h + SLOP;
  const shapes = [...document.querySelectorAll('[data-testid^="zone-shape-"]')].map(rect);
  return [...document.querySelectorAll('[data-testid^="zone-tag-"]')].map((tag, index) => {
    const box = rect(tag);
    const shape = shapes[index];
    return {
      index,
      text: tag.querySelector('.zone-tag-text')?.textContent ?? '',
      tag_box: [box.x, box.y, box.w, box.h].map(Math.round),
      shape_box: shape ? [shape.x, shape.y, shape.w, shape.h].map(Math.round) : null,
      in_own_zone: shape ? inside(box, shape) : false,
      over_other_zones: shapes
        .map((shape, other) => other !== index && box.x < shape.x + shape.w
          && box.x + box.w > shape.x && box.y < shape.y + shape.h && box.y + box.h > shape.y)
        .some(Boolean),
    };
  });
}
"""


def placement(page) -> list[dict]:
    return page.evaluate(PLACEMENT)


def open_toolbar(ui: Ui) -> None:
    """Save zone folds the toolbar away; Edit zone brings it back"""
    if ui.page.get_by_test_id("toolbar").count() == 0:
        ui.page.locator("button.primary").click()
        ui.page.wait_for_selector('[data-testid="toolbar"]')
        time.sleep(0.3)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="a 640x480 clip to play")
    parser.add_argument("--shots", type=Path, default=Path("e2e-shots"),
                        help="where the screenshots go")
    parser.add_argument("--api-port", type=int, default=8098)
    parser.add_argument("--ui-port", type=int, default=5178)
    parser.add_argument("--keep", action="store_true",
                        help="leave the backend and Vite running afterwards")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="dz-first-zone-"))
    args.shots.mkdir(parents=True, exist_ok=True)
    video = args.video or build_clip(work / "bus_12s.mp4")

    backend = Backend(free_port(args.api_port), work / "trinetra.db",
                      work / "workers.json", work / "backend.log")
    dev = DevServer(free_port(args.ui_port), backend.port, work / "vite.log")

    report = Report()
    browser = None

    def drag(ui: Ui, start, end, hold_at: float = 0.45):
        """press, move, hold (so a screenshot can be taken), move on, release"""
        first = ui.frame_to_screen(*start)
        last = ui.frame_to_screen(*end)
        page.mouse.move(first[0], first[1])
        page.mouse.down()
        for step in range(1, 6):
            page.mouse.move(
                first[0] + (last[0] - first[0]) * step / 5,
                first[1] + (last[1] - first[1]) * step / 5,
            )
        time.sleep(hold_at)
        middle = overlay(page)
        page.mouse.move(last[0], last[1])
        time.sleep(0.15)
        page.mouse.up()
        time.sleep(0.6)
        return middle

    try:
        backend.start()
        dev.start()
        print(f"backend {backend.url}  ui {dev.url}  video {video}", flush=True)
        print(f"logs and throwaway database: {work}", flush=True)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1680, "height": 1050})
            page = context.new_page()
            page.set_default_timeout(30_000)
            ui = Ui(page, backend, args.shots)

            page.goto(dev.url, wait_until="domcontentloaded")
            page.wait_for_selector("button.primary")

            # -- 0: the case under test -----------------------------------
            step = report.step(0, "Zero zones")
            report.check(step, backend.config()["zones"] == [],
                         "the throwaway database starts with no saved zone")
            page.locator("details.advanced summary").click()
            page.get_by_label("Camera source (0 = webcam, or a video file path)").fill(str(video))
            page.locator("button.primary").click()
            ui.wait_state(lambda s: s["running"] and s["frame_seq"] > 2, 60, "the camera to run")
            page.wait_for_selector('[data-testid="toolbar"]')
            empty = overlay(page)
            report.check(step, empty["svg"] == 1,
                         "the overlay is on the picture even with nothing drawn")
            report.check(step, empty["shapes"] == 0 and empty["tag_text"] == [],
                         "nothing is painted into it yet")
            shot = ui.shot("0-no-zones-yet.png")
            step.shots.append(shot)

            # -- 1: Box as the very first zone ----------------------------
            step = report.step(1, "Box")
            ui.tool("Rectangle").click()
            mid = drag(ui, *BOX_ZONE)
            report.check(step, mid["svg"] == 1, "the overlay is there mid-drag")
            report.check(step, mid["band"] == 1 and mid["band_shape"] == "rect",
                         f"the rubber band is being drawn ({mid['band_shape']})")
            report.check(step, (mid["band_box"] or {}).get("w", 0) > 20,
                         f"the band already has size on screen: {mid['band_box']}")
            report.check(step, mid["shapes"] == 0,
                         "no shape is committed yet, the band is the live preview")
            shot = ui.shot("1-box-while-dragging.png")
            step.shots.append(shot)

            after = overlay(page)
            report.check(step, after["shapes"] == 1,
                         f"the box is on the picture after the release ({after['shapes']} shape)")
            report.check(step, after["tag_text"] == ["Zone 1"],
                         f"and it is labelled {after['tag_text']}")
            shot = ui.shot("2-box-after-release.png")
            step.shots.append(shot)

            page.locator("button.primary").click()
            ui.wait_state(lambda s: len(s["zone_polygons"]) == 1, 30, "the box to be stored")
            time.sleep(1.0)
            saved = overlay(page)
            report.check(step, saved["shapes"] == 1 and saved["tag_text"] == ["Zone 1"],
                         f"the box and its tag survive Save zone ({saved['tag_text']})")
            shot = ui.shot("3-box-after-save.png")
            step.shots.append(shot)

            # -- 2: Polygon as the very first zone ------------------------
            step = report.step(2, "Polygon")
            open_toolbar(ui)
            ui.tool("Clear").click()
            time.sleep(0.4)
            report.check(step, overlay(page)["shapes"] == 0,
                         "cleared back to zero zones, so this one is the first again")
            ui.tool("Polygon").click()
            corners = POLYGON_ZONE
            for index in range(len(corners) - 1):
                ui.page.mouse.click(*ui.frame_to_screen(*corners[index]))
                time.sleep(0.25)
                live = overlay(page)
                report.check(step, live["preview"] == 1 and live["preview_points"] != "",
                             f"the dashed outline is drawn after click {index + 1}")
            shot = ui.shot("4-polygon-while-clicking.png")
            step.shots.append(shot)
            # clicking the first corner again closes the polygon
            ui.page.mouse.click(*ui.frame_to_screen(*corners[0]))
            time.sleep(0.6)
            closed = overlay(page)
            report.check(step, closed["shapes"] == 1 and closed["preview"] == 0,
                         f"closing it leaves one shape and no preview ({closed['shapes']})")
            report.check(step, closed["tag_text"] == ["Zone 1"],
                         f"and it is labelled {closed['tag_text']}")
            page.locator("button.primary").click()
            ui.wait_state(lambda s: len(s["zone_polygons"]) == 1, 30, "the polygon to be stored")
            time.sleep(1.0)
            report.check(step, overlay(page)["tag_text"] == ["Zone 1"],
                         "the polygon and its tag survive Save zone")
            shot = ui.shot("5-polygon-after-save.png")
            step.shots.append(shot)

            # -- 3: Circle as the very first zone -------------------------
            step = report.step(3, "Circle")
            open_toolbar(ui)
            ui.tool("Clear").click()
            time.sleep(0.4)
            report.check(step, overlay(page)["shapes"] == 0,
                         "cleared back to zero zones for the circle")
            ui.tool("Circle").click()
            mid = drag(ui, *CIRCLE_ZONE)
            report.check(step, mid["svg"] == 1, "the overlay is there mid-drag")
            report.check(step, mid["band"] == 1 and mid["band_shape"] == "ellipse",
                         f"the rubber band is an ellipse ({mid['band_shape']})")
            report.check(step, (mid["band_box"] or {}).get("w", 0) > 20,
                         f"the band already has size on screen: {mid['band_box']}")
            shot = ui.shot("6-circle-while-dragging.png")
            step.shots.append(shot)
            after = overlay(page)
            report.check(step, after["shapes"] == 1 and after["tag_text"] == ["Zone 1"],
                         f"the circle is on the picture and labelled {after['tag_text']}")
            shot = ui.shot("7-circle-after-release.png")
            step.shots.append(shot)

            # -- 4: two zones, two tags -----------------------------------
            step = report.step(4, "Two zones")
            ui.tool("Rectangle").click()
            drag(ui, (400, 60), (600, 200))
            time.sleep(0.5)
            both = overlay(page)
            report.check(step, both["shapes"] == 2, "the second zone is drawn as well")
            report.check(step, both["tag_text"] == ["Zone 1", "Zone 2"],
                         f"each zone has its own name: {both['tag_text']}")
            report.check(step, both["chips"] == ["Zone 1", "Zone 2"],
                         f"and the chips under the video agree: {both['chips']}")
            report.check(step, len(set(both["tag_at"])) == 2,
                         f"the two tags sit in two different corners: {both['tag_at']}")
            where = placement(page)
            report.check(step, [w["in_own_zone"] for w in where] == [True, True],
                         f"each tag is measured inside its own zone: {where}")
            report.check(step, not any(w["over_other_zones"] for w in where),
                         "no tag overlaps the other zone")
            report.check(step, len(both["machine"]) == 1 and bool(both["machine"][0]),
                         f"the machine is named once, in the card header: {both['machine']}")
            page.locator("button.primary").click()
            ui.wait_state(lambda s: len(s["zone_polygons"]) == 2, 30, "both zones to be stored")
            time.sleep(1.0)
            saved = overlay(page)
            report.check(step, saved["tag_text"] == ["Zone 1", "Zone 2"],
                         f"both tags are still there after Save zone: {saved['tag_text']}")
            shot = ui.shot("8-two-zones-after-save.png")
            step.shots.append(shot)

            # -- 5: the point list is folded away until it is asked for ---
            step = report.step(5, "Coordinates")
            page.locator("button.primary").click()  # Edit zone
            ui.wait_state(lambda s: True, 5, "the editor to open")
            page.wait_for_selector('[data-testid="toolbar"]', timeout=15_000)

            # no zone picked, so there is no point list at all
            report.check(step, page.locator('[data-testid="coord-details"]').count() == 0,
                         "no point list while nothing is picked")

            page.get_by_label("Select zone 2").click()
            time.sleep(0.4)
            panel = page.locator('[data-testid="coord-details"]')
            report.check(step, panel.count() == 1, "picking a zone brings up its point list")
            report.check(step, panel.evaluate("el => el.open") is False,
                         "the point list is COLLAPSED by default when a zone is picked")
            report.check(step, "Zone 2 coordinates" in panel.locator("summary").inner_text(),
                         f"the summary names the zone: "
                         f"{panel.locator('summary').inner_text().strip()!r}")
            report.check(step, not panel.locator("tbody tr").first.is_visible(),
                         "the point numbers are not on screen while it is collapsed "
                         "(the rows are in the page but display:none)")
            shot = ui.shot("9-coordinates-collapsed.png")
            step.shots.append(shot)

            panel.locator("summary").click()
            time.sleep(0.4)
            report.check(step, panel.evaluate("el => el.open") is True,
                         "opening it shows the numbers")
            rows = panel.locator("tbody tr").count()
            report.check(step, rows == 4, f"Zone 2 has its own four points ({rows})")
            report.check(step, panel.locator("summary").inner_text().strip() == "Zone 2 coordinates",
                         "and only that zone: the heading names Zone 2, never Zone 1")
            shot = ui.shot("10-coordinates-open.png")
            step.shots.append(shot)

            context.close()
    finally:
        pending = sys.exc_info()[1]
        if pending is not None:
            report.fail(f"the run stopped here: {type(pending).__name__}: {pending}")
        report.finish()
        (args.shots / "first-zone-result.json").write_text(
            json.dumps([vars(entry) for entry in report.steps], indent=2), encoding="utf-8"
        )
        if browser is not None:
            try:
                browser.close()
            except Exception:  # noqa: BLE001 - the driver may already be gone
                pass
        if not args.keep:
            dev.stop()
            backend.stop()

    print(f"\nscreenshots: {args.shots}", flush=True)
    print(f"logs: {work}", flush=True)
    return 0 if all(entry.ok for entry in report.steps) else 1


if __name__ == "__main__":
    sys.exit(main())