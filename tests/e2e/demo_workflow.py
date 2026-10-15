r"""End-to-end check of the demo workflow, against the real UI and a real video.

Nothing here is mocked: a video built from ultralytics' bundled bus.jpg is
played by the real pipeline, the real detector finds real people in it, and the
real React UI is driven with Playwright. The backend runs on a spare port with a
throwaway database, so the demo the developer is using is never touched.

The seven steps of the demo are checked in order and each one reports PASS or
FAIL with the numbers behind it:

    1  Start camera      live video appears and people get boxes
    2  Draw              the danger zone is drawn on the live video
    3  Save zone         the zone is stored AND monitoring is live at once
    4  Alert             the banner, the red zone and the beep, then it clears
    5  Edit zone         the toolbar comes back and changes apply live
    6  Persistence       the zone is still there after the app restarts
    7  Stop camera       the feed stops

Usage (needs Node.js for Vite, and Playwright with a Chromium build):

    py -3.13 tests/e2e/demo_workflow.py --video path\to\clip.mp4 --shots out\

The video is not kept in the repository: make one from bus.jpg with

    python -c "import cv2,os,ultralytics;src=os.path.join(os.path.dirname(ultralytics.__file__),'assets','bus.jpg');w=cv2.VideoWriter(r'clip.mp4',cv2.VideoWriter_fourcc(*'mp4v'),12.0,(640,480));[w.write(cv2.resize(cv2.imread(src),(640,480))) for _ in range(144)];w.release()"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The people in bus.jpg, once it has been resized to 640x480. Both are real
# detections from yolo11n, not guesses, so the zone is drawn where a person
# actually stands.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PERSON_ZONE = ((500, 150), (639, 420))
EMPTY_ZONE = ((0, 0), (120, 120))


# --- tiny helpers -----------------------------------------------------------


def free_port(preferred: int) -> int:
    """The first free port at or above `preferred`."""
    for port in range(preferred, preferred + 40):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port near {preferred}")


def get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, body: dict | None = None, timeout: float = 90.0):
    data = json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for(url: str, seconds: float, what: str) -> None:
    deadline = time.monotonic() + seconds
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
                last = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except Exception as exc:  # not up yet
            last = str(exc)
        time.sleep(0.4)
    raise TimeoutError(f"{what} never answered on {url} ({last})")


@dataclass
class Step:
    number: int
    name: str
    ok: bool = True
    checks: int = 0
    evidence: list[str] = field(default_factory=list)
    shots: list[str] = field(default_factory=list)
    failed: str = ""


class Report:
    def __init__(self) -> None:
        self.steps: list[Step] = []

    def step(self, number: int, name: str) -> Step:
        entry = Step(number=number, name=name)
        self.steps.append(entry)
        print(f"\n=== step {number}: {name} ===", flush=True)
        return entry

    def check(self, entry: Step, ok: bool, evidence: str) -> bool:
        entry.checks += 1
        entry.ok = entry.ok and ok
        if not ok and not entry.failed:
            entry.failed = evidence
        entry.evidence.append(f"{'PASS' if ok else 'FAIL'} {evidence}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {evidence}", flush=True)
        return ok

    def fail(self, message: str) -> None:
        """Note an unexpected crash against whichever step was running."""
        entry = self.steps[-1] if self.steps else self.step(0, "Setup")
        entry.checks += 1
        entry.ok = False
        if not entry.failed:
            entry.failed = message
        entry.evidence.append(f"FAIL {message}")
        print(f"  [FAIL] {message}", flush=True)

    def finish(self) -> None:
        print("\n================ RESULT ================")
        for entry in self.steps:
            mark = "PASS" if entry.ok and entry.checks else "FAIL"
            print(f"step {entry.number} {entry.name:<14} {mark}")
            for line in entry.evidence:
                print(f"    {line}")
            if entry.shots:
                print(f"    screenshots: {', '.join(entry.shots)}")
        print("========================================")


# --- the processes the check runs against -----------------------------------


class Backend:
    def __init__(self, port: int, db: Path, snapshot: Path, log: Path) -> None:
        self.port = port
        self.db = db
        self.snapshot = snapshot
        self.log = log
        self.process: subprocess.Popen | None = None
        self.python = os.environ.get("TRINETRA_PYTHON") or shutil.which("python") or "python"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        env = os.environ.copy()
        # a database of its own, so a developer's own zones are never touched
        env["TRINETRA_DB"] = str(self.db)
        env["TRINETRA_SNAPSHOT"] = str(self.snapshot)
        env["PYTHONUNBUFFERED"] = "1"
        handle = self.log.open("ab")
        self.process = subprocess.Popen(
            [
                self.python,
                "-m",
                "uvicorn",
                "api.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=handle,
        )
        wait_for(f"{self.url}/health", 90, "the backend")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.process = None

    def state(self) -> dict:
        # the pipeline holds a lock while it publishes, and the machine may be
        # busy running the detector, so one slow answer is not a failure
        last: Exception | None = None
        for _ in range(3):
            try:
                return get_json(f"{self.url}/state", timeout=15.0)
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5)
        raise last if last else RuntimeError("the backend never answered /state")

    def config(self) -> dict:
        return get_json(f"{self.url}/config", timeout=15.0)

    def frame_seq(self) -> int:
        """The published frame counter, wrapped around by the clip length.

        `frame_seq` is the index inside the source file, so a looping video
        restarts it at 0 every time the file ends. A caller that needs to know
        whether frames kept arriving has to count the wraps itself, which
        `frames_advanced` does.
        """
        return int(self.state()["frame_seq"])

    #: how many laps the clip is assumed to be long enough for between two samples
    #: (144 frames for the test clip), so a wrap can be turned back into the
    #: number of frames that were really played. Anything above the real clip
    #: length only means two samples straddled more than one loop.
    WRAP = 1000

    def frames_advanced(self, seconds: float, sample: float = 0.25) -> tuple[int, int]:
        """`(frames, loops)` published during `seconds`, counting the loops.

        The clip is replayed when it ends, so the raw counter can drop from 143
        to 2 while the video is obviously still running. That is not a stall, it
        is the file boundary, so each drop is added back in as a whole lap. The
        result only goes up, which is what "the video never stopped" means.

        Returning the loop count separately keeps the log readable: a raw total
        that jumped by `WRAP` would print as a five billion frame jump.
        """
        started = self.frame_seq()
        current = started
        loops = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(sample)
            seen = self.frame_seq()
            if seen < current:
                loops += 1  # the clip ended and was replayed from the top
            current = seen
        return (current + loops * self.WRAP) - started, loops


class DevServer:
    """The real Vite dev server, moved off port 5173 for the check."""

    def __init__(self, port: int, api_port: int, log: Path) -> None:
        self.port = port
        self.api_port = api_port
        self.log = log
        self.process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        npm = shutil.which("npm") or "npm"
        env = os.environ.copy()
        env["TRINETRA_API"] = f"http://127.0.0.1:{self.api_port}"
        handle = self.log.open("ab")
        self.process = subprocess.Popen(
            [npm, "--prefix", "frontend", "run", "dev", "--", "--port", str(self.port),
             "--strictPort"],
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=handle,
        )
        wait_for(self.url, 120, "the Vite dev server")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.process = None


# --- driving the page -------------------------------------------------------

# Counts the alert beep instead of listening for it: a headless browser has no
# audio device, but the oscillator the app creates is still observable.
BEEP_SPY = """
window.__beeps = 0;
window.__audioContexts = 0;
const RealAudioContext = window.AudioContext || window.webkitAudioContext;
if (RealAudioContext) {
  class SpyAudioContext extends RealAudioContext {
    constructor(...args) {
      super(...args);
      window.__audioContexts += 1;
    }
    createOscillator() {
      const oscillator = super.createOscillator();
      const start = oscillator.start.bind(oscillator);
      oscillator.start = (...args) => {
        window.__beeps += 1;
        return start(...args);
      };
      return oscillator;
    }
  }
  window.AudioContext = SpyAudioContext;
  window.webkitAudioContext = SpyAudioContext;
}
"""


class Ui:
    """The handful of things this check does to the page."""

    def __init__(self, page, backend: Backend, shots: Path) -> None:
        self.page = page
        self.backend = backend
        self.shots = shots

    def shot(self, name: str) -> str:
        path = self.shots / name
        self.page.screenshot(path=str(path))
        return str(path)

    def tool(self, name: str):
        """A drawing tool, looked up inside the toolbar only.

        The zone chips are named "Select zone 1" and friends, so a page-wide
        lookup by "Select" finds two buttons and refuses to click either.
        """
        return self.page.get_by_test_id("toolbar").get_by_role("button", name=name, exact=True)

    def video_box(self) -> dict:
        box = self.page.locator("img.video").bounding_box()
        if not box:
            raise RuntimeError("the video element has no size on screen")
        return box

    def frame_to_screen(self, x: int, y: int) -> tuple[float, float]:
        box = self.video_box()
        return (
            box["x"] + (x / FRAME_WIDTH) * box["width"],
            box["y"] + (y / FRAME_HEIGHT) * box["height"],
        )

    def drag(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Press, drag and release, the way a person does it.

        The two ends are nudged a few screen pixels towards the middle of the
        picture: the stage draws a 1px border over the edge of the frame, and a
        press that lands on the border never reaches the video at all. A human
        never hits that, so neither does the check.
        """
        box = self.video_box()
        inset = 4.0

        def clamp_x(value: float) -> float:
            return min(max(value, box["x"] + inset), box["x"] + box["width"] - inset)

        def clamp_y(value: float) -> float:
            return min(max(value, box["y"] + inset), box["y"] + box["height"] - inset)

        first = self.frame_to_screen(*start)
        last = self.frame_to_screen(*end)
        first = (clamp_x(first[0]), clamp_y(first[1]))
        last = (clamp_x(last[0]), clamp_y(last[1]))
        self.page.mouse.move(first[0], first[1])
        self.page.mouse.down()
        # a few steps, so the app sees a real drag and not a click
        for step in range(1, 6):
            self.page.mouse.move(
                first[0] + (last[0] - first[0]) * step / 5,
                first[1] + (last[1] - first[1]) * step / 5,
            )
        self.page.mouse.up()

    def wait_state(self, predicate, seconds: float, what: str):
        deadline = time.monotonic() + seconds
        state = self.backend.state()
        while time.monotonic() < deadline:
            state = self.backend.state()
            if predicate(state):
                return state
            time.sleep(0.2)
        raise TimeoutError(f"{what} within {seconds:.0f}s (last state: {json.dumps(state)[:200]})")


def build_clip(target: Path, seconds: float = 12.0, fps: float = 12.0) -> Path:
    """A short, looping clip of ultralytics' bundled bus.jpg."""
    import cv2
    import ultralytics

    source = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    frame = cv2.resize(cv2.imread(str(source)), (FRAME_WIDTH, FRAME_HEIGHT))
    if frame is None:
        raise RuntimeError(f"could not read {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (FRAME_WIDTH, FRAME_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not write the test clip")
    for _ in range(int(seconds * fps)):
        writer.write(frame)
    writer.release()
    return target


def main() -> int:
    # the toolbar labels carry arrow glyphs, which a cp1252 console cannot print
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="a 640x480 clip to play")
    parser.add_argument("--shots", type=Path, default=Path("e2e-shots"),
                        help="where the screenshots go")
    parser.add_argument("--api-port", type=int, default=8095)
    parser.add_argument("--ui-port", type=int, default=5175)
    parser.add_argument("--keep", action="store_true",
                        help="leave the backend and Vite running afterwards")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="dz-e2e-"))
    args.shots.mkdir(parents=True, exist_ok=True)
    video = args.video or build_clip(work / "bus_12s.mp4")
    log = (work / "servers.log").open("ab")

    backend = Backend(free_port(args.api_port), work / "trinetra.db",
                      work / "workers.json", work / "backend.log")
    dev = DevServer(free_port(args.ui_port), backend.port, work / "vite.log")

    report = Report()
    browser = None
    try:
        backend.start()
        dev.start()
        print(f"backend {backend.url}  ui {dev.url}  video {video}", flush=True)
        print(f"logs and throwaway database: {work}", flush=True)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                args=["--autoplay-policy=no-user-gesture-required"]
            )
            context = browser.new_context(viewport={"width": 1680, "height": 1050})
            context.add_init_script(BEEP_SPY)
            page = context.new_page()
            page.set_default_timeout(30_000)
            ui = Ui(page, backend, args.shots)

            # -- 1: start the camera ------------------------------------
            step = report.step(1, "Start camera")
            page.goto(dev.url, wait_until="domcontentloaded")
            page.wait_for_selector("button.primary")
            report.check(step, page.locator("button.primary").inner_text().strip() == "Start camera",
                         'the one primary button reads "Start camera"')

            # there is no webcam on a build machine, so the clip goes in the
            # source box under Advanced - the same box a real operator uses
            page.locator("details.advanced summary").click()
            source_box = page.get_by_label("Camera source (0 = webcam, or a video file path)")
            source_box.fill(str(video))
            report.check(step, source_box.input_value() == str(video),
                         f"the camera source points at the test clip ({video.name})")

            page.locator("button.primary").click()
            state = ui.wait_state(lambda s: s["running"] and s["frame_seq"] > 0, 60,
                                  "the camera to produce frames")
            first_seq = state["frame_seq"]
            state = ui.wait_state(lambda s: len(s["workers"]) > 0, 60,
                                  "people to be detected")
            report.check(step, True,
                         f"running={state['running']} frame_seq={first_seq} "
                         f"people_boxed={len(state['workers'])} "
                         f"ids={[w['worker_id'] for w in state['workers']]}")
            time.sleep(1.0)
            later = backend.state()
            report.check(step, later["frame_seq"] > first_seq,
                         f"the feed is live: frame_seq {first_seq} -> {later['frame_seq']}")
            report.check(step, later["frame_width"] == FRAME_WIDTH and later["frame_height"] == FRAME_HEIGHT,
                         f"the picture is {later['frame_width']}x{later['frame_height']}")
            shot = ui.shot("1-camera-started.png")
            step.shots.append(shot)
            report.check(step, bool(page.locator("img.video").is_visible()),
                         "the video element is on screen")

            # -- 2: draw the zone on the live video ----------------------
            step = report.step(2, "Draw the zone")
            report.check(step, page.get_by_test_id("toolbar").is_visible(),
                         "the drawing toolbar is showing")
            tools = [ui.tool(name).get_attribute("aria-label")
                     for name in ["Select", "Rectangle", "Polygon", "Circle", "Freehand"]]
            report.check(step, tools == ["Select", "Rectangle", "Polygon", "Circle", "Freehand"],
                         f"the toolbar has {tools} plus Undo / Delete / Clear")
            ui.tool("Rectangle").click()
            ui.drag(PERSON_ZONE[0], PERSON_ZONE[1])
            page.wait_for_selector('[data-testid="zone-shape-0"]', timeout=15_000)
            points = page.locator('[data-testid="zone-shape-0"]').get_attribute("points")
            report.check(step, bool(points) and len(points.split()) == 4,
                         f"a 4 corner zone is drawn on the live video ({points})")
            report.check(step, page.get_by_label("Select zone 1").is_visible(),
                         "the zone shows up in the zone list")
            shot = ui.shot("2-zone-drawn.png")
            step.shots.append(shot)

            # -- 3: save the zone, which also starts monitoring -----------
            step = report.step(3, "Save zone and monitor")
            report.check(step, page.locator("button.primary").inner_text().strip() == "Save zone",
                         'the primary button is now "Save zone"')
            seq_before_save = backend.state()["frame_seq"]
            save_at = time.monotonic()
            page.locator("button.primary").click()

            # the backend is watched from here on, so the alert timings are
            # measured on the pipeline and not on how often React repaints
            inside_at = None
            alert_at = None
            while time.monotonic() - save_at < 25:
                state = backend.state()
                if inside_at is None and state["inside"] > 0:
                    inside_at = time.monotonic() - save_at
                if alert_at is None and state["status"] == "ALERT":
                    alert_at = time.monotonic() - save_at
                    break
                time.sleep(0.05)

            config = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                config = backend.config()
                if config["zones"]:
                    break
                time.sleep(0.2)
            report.check(step, bool(config and config["zones"]),
                         f"the zone is stored: {len(config['zones']) if config else 0} zone(s)")
            if config and config["zones"]:
                report.check(step, len(config["zones"][0]) == 4,
                             f"its corners are {config['zones'][0]}")
            state = ui.wait_state(
                lambda s: s["running"] and bool(s.get("zone_polygons")), 30,
                "monitoring with the new zone")
            report.check(step, state["running"] and bool(state["zone_polygons"]),
                         f"monitoring is live with {len(state['zone_polygons'])} zone(s) - "
                         "no second button was needed")
            report.check(step, page.locator("button.primary").inner_text().strip() == "Edit zone",
                         'the primary button becomes "Edit zone"')
            report.check(step, state["running"],
                         "the capture stayed open while saving: the camera is still running")
            # the clip is 12s and is replayed, so the raw counter wraps; counting
            # the loops is what actually proves the capture was never reopened
            saved_frames, saved_loops = backend.frames_advanced(1.0)
            report.check(step, saved_frames > 0,
                         f"frames kept arriving while saving: {saved_frames} frames in 1s "
                         f"({saved_loops} clip loop(s), frame_seq {seq_before_save} -> "
                         f"{backend.frame_seq()})")
            shot = ui.shot("3-zone-saved.png")
            step.shots.append(shot)

            # -- 4: the alert, and the clearing of it ---------------------
            step = report.step(4, "Alert and clear")
            try:
                page.wait_for_selector(".alert-banner", timeout=20_000)
                banner_seconds = time.monotonic() - save_at
            except Exception:
                banner_seconds = None
            dwell = float(backend.config()["rules"]["dwell_seconds"])
            report.check(step, banner_seconds is not None,
                         'the "DANGER - Person in danger zone" banner appeared'
                         + (f" {banner_seconds:.2f}s after Save zone was pressed" if banner_seconds
                            else " - it never appeared"))
            report.check(step, inside_at is not None,
                         f"a person was counted inside {inside_at:.2f}s after the save"
                         if inside_at is not None else "nobody was ever counted inside the zone")
            report.check(step, alert_at is not None,
                         f"the backend raised ALERT {alert_at:.2f}s after the save"
                         if alert_at is not None else "the backend never reached ALERT")
            state = ui.wait_state(lambda s: s["status"] == "ALERT", 20, "an ALERT state")
            report.check(step, state["status"] == "ALERT" and state["inside"] >= 1,
                         f"the backend says status={state['status']} with {state['inside']} inside")
            report.check(step, bool(state["capacity_alert"]) and state["inside"] > state["allowed_people"],
                         f"it is the people rule talking: {state['inside']} inside, "
                         f"{state['allowed_people']} allowed")
            banner_line = page.locator(".alert-banner").inner_text().splitlines()[0].strip()
            report.check(step, banner_line == "DANGER - Person in danger zone",
                         f'banner text: "{banner_line}"')
            status_text = page.locator(".status-big").inner_text().strip()
            report.check(step, status_text.startswith("ALERT"),
                         f'the Status card reads "{status_text}"')
            classes = page.locator('[data-testid="zone-shape-0"]').get_attribute("class") or ""
            report.check(step, "alert" in classes.split(),
                         f"the zone itself turned red (class=\"{classes}\")")
            beeps = page.evaluate("window.__beeps")
            contexts = page.evaluate("window.__audioContexts")
            report.check(step, contexts >= 1,
                         f"the AudioContext was opened on the Start camera click ({contexts})")
            report.check(step, beeps >= 1, f"the alert beeped {beeps} time(s)")
            report.check(step, page.get_by_label("Mute alert sound").is_visible()
                         or page.get_by_label("Unmute alert sound").is_visible(),
                         "the banner carries a mute toggle")
            shot = ui.shot("4-alert.png")
            step.shots.append(shot)

            # the alarm has to go away once the area is empty. Removing every
            # zone asks first, and monitoring keeps running without one.
            page.locator("button.danger-action").click()
            page.wait_for_selector(".confirm-row", timeout=15_000)
            report.check(step, "Remove all" in page.locator(".confirm-row").inner_text(),
                         'removing the zones asks first: "'
                         + page.locator(".confirm-row").inner_text().split("\n")[0].strip() + '"')
            page.locator(".confirm-row").get_by_role("button", name="Remove all zones").click()
            state = ui.wait_state(lambda s: not s["zone_polygons"], 30, "every zone to be removed")
            ui.wait_state(lambda s: s["inside"] == 0 and s["status"] != "ALERT", 30,
                          "the alert to clear")
            time.sleep(1.0)
            report.check(step, not backend.state()["zone_polygons"],
                         "no danger zone is left on the camera")
            report.check(step, page.locator(".alert-banner").count() == 0,
                         "the banner is gone once the area is empty")
            report.check(step, page.locator(".status-big").inner_text().strip() == "SAFE",
                         'the Status card is back to "SAFE"')
            report.check(step, page.evaluate("window.__beeps") == beeps,
                         f"no extra beep while clearing (still {beeps})")
            cleared_frames, cleared_loops = backend.frames_advanced(1.0)
            report.check(step, cleared_frames > 0,
                         f"monitoring kept running without a zone: {cleared_frames} frames "
                         f"in 1s ({cleared_loops} clip loop(s))")
            shot = ui.shot("4-alert-cleared.png")
            step.shots.append(shot)

            # the dwell rule on its own: allow more people than are inside, so the
            # people rule cannot talk and only the timer can raise the alarm
            page.locator("button.primary").click()  # Edit zone
            page.wait_for_selector('[data-testid="toolbar"]', timeout=15_000)
            page.get_by_label("People allowed in the zone").fill("3")
            ui.tool("Rectangle").click()
            ui.drag(PERSON_ZONE[0], PERSON_ZONE[1])
            page.wait_for_selector('[data-testid="zone-shape-0"]', timeout=15_000)
            dwell_saved_at = time.monotonic()
            page.locator("button.primary").click()  # Save zone
            # watched from the click: the save is two requests, so the first
            # samples still show the old rules and that is not a failure
            inside_after_save = None
            inside_status = None
            worker_alert_at = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                probe = backend.state()
                if inside_after_save is None and probe["allowed_people"] == 3 and probe["inside"] > 0:
                    inside_after_save = time.monotonic() - dwell_saved_at
                    inside_status = probe["status"]
                if worker_alert_at is None and probe["worker_alert"]:
                    worker_alert_at = time.monotonic() - dwell_saved_at
                    break
                time.sleep(0.05)
            report.check(step, inside_after_save is not None and inside_status != "ALERT",
                         f"with {3} allowed the people rule stays quiet: the person is inside "
                         f"{inside_after_save:.2f}s after the save and the status is still "
                         f"{inside_status}"
                         if inside_after_save is not None
                         else "the person was never counted inside the new zone")
            report.check(step, worker_alert_at is not None and inside_after_save is not None
                         and worker_alert_at - inside_after_save >= dwell - 0.35,
                         f"the dwell timer waited its turn: {dwell:g}s rule, "
                         f"{worker_alert_at - inside_after_save:.2f}s from first inside to the alarm"
                         if worker_alert_at is not None and inside_after_save is not None
                         else "the dwell alarm never fired")
            state = backend.state()
            alerting = [w for w in state["workers"] if w["state"] == "ALERT"]
            report.check(step, bool(alerting) and alerting[0]["dwell_seconds"] >= dwell - 0.35,
                         f"the worker is ALERT after {alerting[0]['dwell_seconds']}s inside"
                         if alerting else "no worker ever reached the ALERT state")
            report.check(step, not state["capacity_alert"],
                         "it was the dwell rule that raised it, not the people rule")
            shot = ui.shot("4-dwell-alert.png")
            step.shots.append(shot)

            # a second, empty zone must not make the app call itself safe: the
            # person is inside the other one, and inside any zone counts
            page.locator("button.primary").click()  # Edit zone
            page.wait_for_selector('[data-testid="toolbar"]', timeout=15_000)
            ui.tool("Rectangle").click()
            ui.drag(EMPTY_ZONE[0], EMPTY_ZONE[1])
            page.wait_for_selector('[data-testid="zone-shape-1"]', timeout=15_000)
            page.locator("button.primary").click()  # Save zone
            state = ui.wait_state(lambda s: len(s["zone_polygons"]) == 2, 30,
                                  "both zones to be stored")
            time.sleep(1.5)
            state = backend.state()
            report.check(step, len(state["zone_polygons"]) == 2 and state["status"] == "ALERT",
                         f"both zones are live ({len(state['zone_polygons'])}) and the alarm is still "
                         f"up: a person inside zone 1 counts although zone 2 is empty")
            shot = ui.shot("4-two-zones.png")
            step.shots.append(shot)

            # back to the shipped defaults, and leave one zone over a person so
            # the steps that follow start from the demo's normal state
            page.locator("button.danger-action").click()
            page.locator(".confirm-row").get_by_role("button", name="Remove all zones").click()
            ui.wait_state(lambda s: not s["zone_polygons"], 30, "every zone to be removed")
            page.locator("button.primary").click()  # Edit zone
            page.wait_for_selector('[data-testid="toolbar"]', timeout=15_000)
            page.get_by_label("People allowed in the zone").fill("0")
            ui.tool("Rectangle").click()
            ui.drag(PERSON_ZONE[0], PERSON_ZONE[1])
            page.locator("button.primary").click()  # Save zone
            state = ui.wait_state(
                lambda s: len(s["zone_polygons"]) == 1 and s["allowed_people"] == 0
                and s["status"] == "ALERT",
                30, "one zone over a person with the default limit")
            report.check(step, state["allowed_people"] == 0 and state["capacity_alert"],
                         f"the demo defaults are back: {state['allowed_people']} allowed, one zone "
                         f"over a person, and the people rule is what alarms (status={state['status']})")

            # -- 5: edit the zone while the camera keeps running ---------
            step = report.step(5, "Edit zone")
            page.locator("button.primary").click()  # Edit zone
            page.wait_for_selector('[data-testid="toolbar"]', timeout=15_000)
            report.check(step, backend.state()["running"],
                         "the camera is still running after Edit zone")
            report.check(step, page.locator(".zone-chips .chip").count() == 1,
                         "the zone list is showing the saved zone")
            # the clip is 12s and is replayed, so the raw counter wraps; counting the
            # loops is what actually proves the capture was never reopened
            edit_frames, edit_loops = backend.frames_advanced(1.0)
            report.check(step, edit_frames > 0,
                         f"the video never stopped: {edit_frames} frames in 1s "
                         f"({edit_loops} clip loop(s), frame_seq {backend.frame_seq()})")
            shot = ui.shot("5-edit-zone.png")
            step.shots.append(shot)

            # move a corner and save it again
            ui.tool("Select").click()
            page.get_by_label("Select zone 1").click()
            corner = json.loads(json.dumps(backend.config()["zones"][0]))[0]
            page.wait_for_timeout(200)
            page.mouse.move(*ui.frame_to_screen(*corner))
            page.mouse.down()
            page.mouse.move(*ui.frame_to_screen(corner[0] - 40, corner[1] + 20), steps=5)
            page.mouse.up()
            page.wait_for_timeout(400)
            page.locator("button.primary").click()
            ui.wait_state(
                lambda s: bool(s["zone_polygons"]) and s["zone_polygons"][0][0] != corner, 30,
                "the moved corner to be applied")
            saved = backend.config()["zones"][0][0]
            report.check(step, saved != corner,
                         f"the changed zone was stored live: corner {corner} -> {saved}")
            report.check(step, backend.state()["running"],
                         "monitoring never restarted while the zone was edited")
            shot = ui.shot("5-zone-edited.png")
            step.shots.append(shot)

            # -- 6: the zone survives a restart of the app ----------------
            step = report.step(6, "Zones persist")
            zones_before = backend.config()["zones"]
            dev.stop()
            backend.stop()
            backend.start()
            dev.start()
            after = backend.config()["zones"]
            report.check(step, after == zones_before,
                         f"the same {len(after)} zone(s) came back after a restart "
                         f"({after[0] if after else None})")
            page.goto(dev.url, wait_until="domcontentloaded")
            page.wait_for_selector("button.primary")
            report.check(step, page.locator("button.primary").inner_text().strip() == "Start camera",
                         'the app opens on "Start camera" with the zone already stored')
            page.locator("button.primary").click()
            ui.wait_state(lambda s: s["running"] and s["frame_seq"] > 0, 60,
                          "the camera after the restart")
            page.wait_for_selector('[data-testid="zone-shape-0"]', timeout=20_000)
            report.check(step, page.get_by_label("Select zone 1").is_visible(),
                         "the saved zone is drawn again, with no drawing at all")
            shot = ui.shot("6-after-restart.png")
            step.shots.append(shot)

            # -- 7: stop the camera --------------------------------------
            step = report.step(7, "Stop camera")
            page.get_by_role("button", name="Stop camera").click()
            state = ui.wait_state(lambda s: not s["running"], 30, "the camera to stop")
            report.check(step, not state["running"], "the backend reports running=False")
            # the card follows the poll, so give React a moment to repaint
            try:
                page.wait_for_function(
                    "() => (document.querySelector('.status-big') || {}).textContent"
                    "?.trim() === 'NOT MONITORING'",
                    timeout=15_000)
            except Exception:  # noqa: BLE001 - the text is read either way
                pass
            card_text = page.locator(".status-big").inner_text().strip()
            report.check(step, card_text == "NOT MONITORING",
                         f'the Status card reads "{card_text}"')
            # counted once the pipeline is already down: the frames between the
            # click and the stop landing are the last ones it will ever produce
            stopped_at = backend.state()["frame_seq"]
            time.sleep(2.0)
            frozen = backend.state()["frame_seq"]
            report.check(step, frozen == stopped_at and not backend.state()["running"],
                         f"no more frames are produced (frame_seq stayed at {frozen} for 2s)")
            report.check(step, page.get_by_test_id("toolbar").count() == 0,
                         "the drawing toolbar is gone while the camera is stopped")
            shot = ui.shot("7-stopped.png")
            step.shots.append(shot)

            context.close()
    finally:
        # an unexpected crash is worth more in the report than on its own: the
        # step it hit, and everything that step had already proved, are both here
        pending = sys.exc_info()[1]
        if pending is not None:
            report.fail(f"the run stopped here: {type(pending).__name__}: {pending}")
        report.finish()
        (args.shots / "result.json").write_text(
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