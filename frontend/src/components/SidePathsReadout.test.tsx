import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SidePathsReadout from "./SidePathsReadout";

describe("SidePathsReadout", () => {
  it("shows Off cards when every path is off", () => {
    render(
      <SidePathsReadout
        model={{ crowd: { enabled: false }, ppe: { enabled: false }, running: true }}
      />
    );
    expect(screen.getByTestId("crowd-card").textContent).toContain("Off");
    expect(screen.getByTestId("ppe-card").textContent).toContain("Off");
    expect(screen.queryByTestId("crowd-tile")).toBeNull();
    expect(screen.queryByTestId("ppe-tile")).toBeNull();
  });

  it("shows Off cards while stopped even when ticked", () => {
    render(
      <SidePathsReadout
        model={{
          crowd: { enabled: true, metrics: { person_count: 3, coverage_fraction: 0.1, density_per_megapixel: 1 } },
          ppe: { enabled: true, grade: "high", stats: {} },
          running: false,
        }}
      />
    );
    expect(screen.queryByTestId("crowd-tile")).toBeNull();
    expect(screen.queryByTestId("ppe-tile")).toBeNull();
  });

  it("shows live crowd numbers with neutral styling", () => {
    render(
      <SidePathsReadout
        model={{
          crowd: {
            enabled: true,
            metrics: { person_count: 3, coverage_fraction: 0.125, density_per_megapixel: 2.5 },
          },
          ppe: null,
          running: true,
        }}
      />
    );
    expect(screen.getByTestId("crowd-tile")).toBeDefined();
    expect(screen.getByText("People in view")).toBeTruthy();
    expect(screen.getByText("Not the same as zone capacity")).toBeTruthy();
    expect(screen.getByTestId("crowd-count").textContent).toBe("3");
    expect(screen.getByTestId("crowd-detail").textContent).toMatch(/12\.5% of view/);
    expect(screen.getByTestId("crowd-detail").textContent).toMatch(/2\.5\/MP/);
    expect(screen.queryByTestId("ppe-tile")).toBeNull();
  });

  it("shows the PPE grade with breakdown, falls and the grading-only caption", () => {
    const { rerender } = render(
      <SidePathsReadout
        model={{
          crowd: null,
          ppe: {
            enabled: true,
            grade: "high",
            deferred: true,
            stats: {
              violation_count: 2,
              person_count: 1,
              fall_count: 1,
              violations_by_class: [["NO-Hardhat", 2]],
            },
          },
          running: true,
        }}
      />
    );
    expect(screen.getByTestId("ppe-grade").textContent).toBe("high");
    expect(screen.getByTestId("ppe-detail").textContent).toMatch(/2 violations \/ 1 people/);
    expect(screen.getByTestId("ppe-breakdown").textContent).toMatch(/Hardhat.*2/);
    expect(screen.getByTestId("ppe-falls").textContent).toMatch(/1 fall/);
    expect(screen.getByTestId("ppe-deferred-note").textContent).toMatch(
      /Grading only, alerts not enabled/
    );
    rerender(
      <SidePathsReadout
        model={{
          crowd: null,
          ppe: { enabled: true, grade: "high", deferred: false, stats: {} },
          running: true,
        }}
      />
    );
    expect(screen.queryByTestId("ppe-deferred-note")).toBeNull();
  });
});
