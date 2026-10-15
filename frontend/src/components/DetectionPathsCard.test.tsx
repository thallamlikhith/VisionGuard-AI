import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DetectionPathsCard from "./DetectionPathsCard";
import type { DetectionPathsModel } from "./DetectionPathsCard";

function model(overrides: Partial<DetectionPathsModel> = {}): DetectionPathsModel {
  return {
    rows: [
      { key: "hazard", title: "Hazard zone", blurb: "hazard blurb", enabled: true },
      { key: "fire", title: "Fire", blurb: "fire blurb", enabled: false, fps: 2.0, fpsCaption: "shared with Smoke" },
      { key: "smoke", title: "Smoke", blurb: "smoke blurb", enabled: false },
      { key: "crowd", title: "Crowd", blurb: "crowd blurb", enabled: true, fps: 2.0 },
      { key: "ppe", title: "PPE", blurb: "ppe blurb", enabled: false, fps: 1.0 },
    ],
    ready: true,
    onToggle: vi.fn(),
    onFps: vi.fn(),
    ...overrides,
  };
}

function fireRow(overrides = {}): DetectionPathsModel["rows"][number] {
  return {
    key: "fire",
    title: "Fire",
    blurb: "fire blurb",
    enabled: true,
    fps: 2.0,
    fpsCaption: "shared with Smoke",
    gateSeconds: { fire: 1.5, smoke: 2.5 },
    ...overrides,
  };
}

function field(label: string): HTMLInputElement {
  const el = screen.getByLabelText(label);
  if (!(el instanceof HTMLInputElement)) throw new Error(`${label} is not an input`);
  return el;
}

function resetButton(): HTMLButtonElement {
  const el = screen.getByTestId("paths-reset");
  if (!(el instanceof HTMLButtonElement)) throw new Error("paths-reset is not a button");
  return el;
}

describe("DetectionPathsCard", () => {
  it("renders one row per system with switches reflecting the model", () => {
    render(<DetectionPathsCard model={model()} />);
    expect(field("Hazard zone enabled").checked).toBe(true);
    expect(field("Fire enabled").checked).toBe(false);
    expect(field("Smoke enabled").checked).toBe(false);
    expect(field("Crowd enabled").checked).toBe(true);
    expect(field("PPE enabled").checked).toBe(false);
  });

  it("toggling a switch calls back with the system key and value", () => {
    const m = model();
    render(<DetectionPathsCard model={m} />);
    fireEvent.click(field("Smoke enabled"));
    expect(m.onToggle).toHaveBeenCalledWith("smoke", true);
    fireEvent.click(field("Hazard zone enabled"));
    expect(m.onToggle).toHaveBeenCalledWith("hazard", false);
  });

  it("puts the shared cadence on the Fire row and none on Smoke or Hazard", () => {
    render(<DetectionPathsCard model={model()} />);
    expect(field("Fire target FPS").getAttribute("value")).toBe("2");
    expect(screen.getByText(/shared with Smoke/)).toBeTruthy();
    expect(screen.queryByLabelText("Smoke target FPS")).toBeNull();
    expect(screen.queryByLabelText("Hazard zone target FPS")).toBeNull();
  });

  it("disables the fps field of a switched-off path", () => {
    render(<DetectionPathsCard model={model()} />);
    expect(field("Fire target FPS").disabled).toBe(true);
    expect(field("Crowd target FPS").disabled).toBe(false);
  });

  it("edits the cadence through the callback", () => {
    const m = model();
    render(<DetectionPathsCard model={m} />);
    fireEvent.change(field("Crowd target FPS"), {
      target: { value: "3" },
    });
    expect(m.onFps).toHaveBeenCalledWith("crowd", 3);
  });

  it("shows the CPU load note", () => {
    render(<DetectionPathsCard model={model()} />);
    expect(screen.getByTestId("paths-cpu-note").textContent).toMatch(/ms \/ frame/);
  });

  it("disables everything before the config has loaded", () => {
    render(<DetectionPathsCard model={model({ ready: false })} />);
    expect(field("Crowd enabled").disabled).toBe(true);
  });

  it("disables the reset button before the config has loaded", () => {
    render(<DetectionPathsCard model={model({ ready: false })} />);
    expect(resetButton().disabled).toBe(true);
  });

  it("offers a reset to defaults that calls back", () => {
    const m = model({ onReset: vi.fn() });
    render(<DetectionPathsCard model={m} />);
    fireEvent.click(resetButton());
    expect(m.onReset).toHaveBeenCalledTimes(1);
  });

  it("sets the fire/smoke minimum to 0.5 FPS and keeps 0.1 elsewhere", () => {
    render(<DetectionPathsCard model={model()} />);
    expect(field("Fire target FPS").getAttribute("min")).toBe("0.5");
    expect(field("Crowd target FPS").getAttribute("min")).toBe("0.1");
  });

  it("states the gate time in plain words at the current cadence", () => {
    const m = model();
    m.rows = m.rows.map((row) => (row.key === "fire" ? fireRow() : row));
    render(<DetectionPathsCard model={m} />);
    expect(screen.getByTestId("path-gate-time").textContent).toBe(
      "At 2 FPS: fire confirms in about 1.5 s, smoke in about 2.5 s"
    );
    expect(screen.queryByTestId("path-fps-low-warning")).toBeNull();
    expect(screen.queryByTestId("path-gate-warning")).toBeNull();
  });

  it("shows a stored value below the minimum as-is with a warning", () => {
    const m = model();
    m.rows = m.rows.map((row) => (row.key === "fire" ? fireRow({ fps: 0.1 }) : row));
    render(<DetectionPathsCard model={m} />);
    // shown, never silently rewritten to the minimum
    expect(field("Fire target FPS").getAttribute("value")).toBe("0.1");
    expect(screen.getByTestId("path-fps-low-warning").textContent).toMatch(/0\.5 FPS minimum/);
    // the gate would need about 10 s per type: longer than an 8 s test clip
    expect(screen.getByTestId("path-gate-time").textContent).toBe(
      "At 0.1 FPS: fire confirms in about 10 s, smoke in about 10 s"
    );
  });

  it("does not clamp a typed low value up on edit", () => {
    const m = model();
    m.rows = m.rows.map((row) => (row.key === "fire" ? fireRow() : row));
    render(<DetectionPathsCard model={m} />);
    fireEvent.change(field("Fire target FPS"), { target: { value: "0.3" } });
    expect(m.onFps).toHaveBeenCalledWith("fire", 0.3);
  });

  it("warns when the gate cannot confirm inside a short clip", () => {
    const m = model();
    m.rows = m.rows.map((row) =>
      row.key === "fire" ? fireRow({ gateSeconds: { fire: 20, smoke: 2.5 } }) : row
    );
    render(<DetectionPathsCard model={m} />);
    expect(screen.getByTestId("path-gate-time").textContent).toBe(
      "At 2 FPS: fire confirms in about 20 s, smoke in about 2.5 s"
    );
    expect(screen.getByTestId("path-gate-warning").textContent).toMatch(/8 s/);
    expect(screen.queryByTestId("path-fps-low-warning")).toBeNull();
  });
});
