import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DetectionSystemsPanel from "./DetectionSystemsPanel";
import type { DetectionSystemsModel } from "./DetectionSystemsPanel";

function model(overrides: Partial<DetectionSystemsModel> = {}): DetectionSystemsModel {
  return {
    rows: [
      { key: "hazard", title: "Hazard zone", enabled: true },
      { key: "fire", title: "Fire", enabled: false },
      { key: "smoke", title: "Smoke", enabled: false },
      { key: "crowd", title: "Crowd", enabled: true },
      { key: "ppe", title: "PPE", enabled: false },
    ],
    ready: true,
    onToggle: vi.fn(),
    ...overrides,
  };
}

function box(label: string): HTMLInputElement {
  const el = screen.getByLabelText(`${label} enabled`);
  if (!(el instanceof HTMLInputElement)) throw new Error(`${label} is not an input`);
  return el;
}

describe("DetectionSystemsPanel", () => {
  it("renders five switches reflecting the model", () => {
    render(<DetectionSystemsPanel model={model()} />);
    expect(box("Hazard zone").checked).toBe(true);
    expect(box("Fire").checked).toBe(false);
    expect(box("Smoke").checked).toBe(false);
    expect(box("Crowd").checked).toBe(true);
    expect(box("PPE").checked).toBe(false);
  });

  it("calls back with the system key and value", () => {
    const m = model();
    render(<DetectionSystemsPanel model={m} />);
    fireEvent.click(box("Smoke"));
    expect(m.onToggle).toHaveBeenCalledWith("smoke", true);
    fireEvent.click(box("Hazard zone"));
    expect(m.onToggle).toHaveBeenCalledWith("hazard", false);
  });

  it("disables everything before the config has loaded", () => {
    render(<DetectionSystemsPanel model={model({ ready: false })} />);
    expect(box("Fire").disabled).toBe(true);
  });
});
