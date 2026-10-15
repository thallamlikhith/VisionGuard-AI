import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import VideoSourcePicker from "./VideoSourcePicker";
import type { VideoSourcePickerModel } from "./VideoSourcePicker";

function model(overrides: Partial<VideoSourcePickerModel> = {}): VideoSourcePickerModel {
  return {
    options: [
      { value: "camera", label: "Webcam 0", zones: 0 },
      { value: "/data/videos/one.mp4", label: "line.mp4", zones: 2 },
      { value: "/data/videos/two.mp4", label: "clip.mp4", zones: null },
    ],
    value: "camera",
    disabled: false,
    onSelect: vi.fn(),
    uploadDisabled: false,
    onPickFile: vi.fn(),
    uploadPct: null,
    pending: null,
    onConfirmSwitch: vi.fn(),
    onCancelSwitch: vi.fn(),
    noZoneHint: true,
    ...overrides,
  };
}

describe("VideoSourcePicker", () => {
  it("lists every source with its saved zone count, or none while unknown", () => {
    render(<VideoSourcePicker model={model()} />);
    const options = [...screen.getByTestId("live-source-select").querySelectorAll("option")].map(
      (node) => node.textContent
    );
    expect(options).toEqual(["Webcam 0 · 0 zones", "line.mp4 · 2 zones", "clip.mp4"]);
  });

  it("uses the singular for exactly one zone", () => {
    render(
      <VideoSourcePicker
        model={model({
          options: [{ value: "camera", label: "Webcam 0", zones: 1 }],
        })}
      />
    );
    expect(screen.getByRole("option").textContent).toBe("Webcam 0 · 1 zone");
  });

  it("reports the picked value and disables everything in edit mode", async () => {
    const user = userEvent.setup();
    const m = model({ disabled: true });
    render(<VideoSourcePicker model={m} />);
    const select = screen.getByTestId("live-source-select") as HTMLSelectElement;
    expect(select.disabled).toBe(true);
    await user.selectOptions(select, "/data/videos/one.mp4");
    // a disabled select takes no input: the callback stays quiet
    expect(m.onSelect).not.toHaveBeenCalled();
  });

  it("shows the running-feed confirm with switch and keep actions", () => {
    const m = model({
      pending: { value: "/data/videos/one.mp4", label: "line.mp4" },
      noZoneHint: false,
    });
    render(<VideoSourcePicker model={m} />);
    expect(screen.getByTestId("live-source-confirm").textContent).toMatch(
      /stops the current feed/i
    );
    fireEvent.click(screen.getByRole("button", { name: /^switch$/i }));
    expect(m.onConfirmSwitch).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /^keep current$/i }));
    expect(m.onCancelSwitch).toHaveBeenCalledTimes(1);
  });

  it("shows the no-zone hint only without a pending confirm", () => {
    const { rerender } = render(<VideoSourcePicker model={model()} />);
    expect(screen.getByTestId("live-no-zone-hint").textContent).toMatch(/no zone saved/i);

    rerender(<VideoSourcePicker model={model({ noZoneHint: false })} />);
    expect(screen.queryByTestId("live-no-zone-hint")).toBeNull();

    rerender(
      <VideoSourcePicker
        model={model({ pending: { value: "x", label: "y" } })}
      />
    );
    expect(screen.queryByTestId("live-no-zone-hint")).toBeNull();
  });

  it("shows upload progress while a file is in flight", () => {
    render(<VideoSourcePicker model={model({ uploadPct: 50 })} />);
    expect(
      (screen.getByTestId("live-upload-progress") as HTMLProgressElement).value
    ).toBe(50);
  });

  it("shows the Loop box only when the model carries it", async () => {
    const user = userEvent.setup();
    // no loopVideo at all (older model): nothing rendered
    const { rerender } = render(<VideoSourcePicker model={model()} />);
    expect(screen.queryByTestId("live-loop-video")).toBeNull();

    const onChange = vi.fn();
    rerender(
      <VideoSourcePicker
        model={model({
          loopVideo: { checked: false, disabled: false, onChange },
        })}
      />
    );
    const box = screen.getByTestId("live-loop-video") as HTMLInputElement;
    expect(box.checked).toBe(false);
    await user.click(box);
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("disables the Loop box with the rest of the picker in edit mode", () => {
    render(
      <VideoSourcePicker
        model={model({
          disabled: true,
          loopVideo: { checked: true, disabled: true, onChange: vi.fn() },
        })}
      />
    );
    expect((screen.getByTestId("live-loop-video") as HTMLInputElement).disabled).toBe(true);
  });
});
