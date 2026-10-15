import { useEffect, useState, type ReactNode } from "react";

/** Browser-clock date/time for the top bar (no backend involved). Ticks once
 *  a second; unmounting clears the timer. */
export default function WallClock(): ReactNode {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const date = now.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
  const time = now.toLocaleTimeString("en-GB", { hour12: false });
  return (
    <span className="wall-clock" data-testid="wall-clock">
      {date} · {time}
    </span>
  );
}
