import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies the API to the FastAPI backend so the browser only
// ever talks to one origin (no CORS setup needed). TRINETRA_API moves the
// backend off port 8090, which is what the end-to-end check uses so it can run
// beside a real instance instead of fighting it for the ports.
const api = process.env.TRINETRA_API ?? "http://127.0.0.1:8090";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    host: "127.0.0.1",
    proxy: {
      // one prefix per backend area: Vite matches prefixes, so /events also
      // covers /events/filters, /events/daily, /events/export.csv,
      // /events/ack-all and /events/:id/ack + /snapshot. Listed explicitly so
      // a future tighter match cannot silently drop a sub-route.
      "/config": { target: api, changeOrigin: true },
      "/start": { target: api, changeOrigin: true },
      "/stop": { target: api, changeOrigin: true },
      "/state": { target: api, changeOrigin: true },
      "/health": { target: api, changeOrigin: true },
      "/stream": { target: api, changeOrigin: true },
      "/events": { target: api, changeOrigin: true },
      "/history": { target: api, changeOrigin: true },
      "/fire-alerts": { target: api, changeOrigin: true },
      "/workers": { target: api, changeOrigin: true },
      "/videos": { target: api, changeOrigin: true },
    },
  },
  preview: {
    port: 5174,
    host: "127.0.0.1",
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
