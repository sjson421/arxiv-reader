import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, API calls go to `arxiv-reader serve`.
const api = "http://127.0.0.1:8732";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/digest": api, "/papers": api } },
});
