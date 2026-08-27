import type { IncomingHttpHeaders } from "node:http";
import { resolve } from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

import { createRunsApi } from "./server/runs";

function requestHeaders(source: IncomingHttpHeaders): Headers {
  const headers = new Headers();
  for (const [name, value] of Object.entries(source)) {
    if (Array.isArray(value)) {
      for (const item of value) headers.append(name, item);
    } else if (value !== undefined) {
      headers.set(name, value);
    }
  }
  return headers;
}

function runsApi(): Plugin {
  const api = createRunsApi(
    process.env.CONTRAST_RUNS_DIR ?? resolve(import.meta.dirname, "..", "runs"),
  );
  return {
    name: "contrast-runs-api",
    configureServer(server) {
      server.config.logger.info("Runs API: " + api.root);
      server.middlewares.use(async (incoming, outgoing, next) => {
        try {
          const url = new URL(
            incoming.url ?? "/",
            "http://" + (incoming.headers.host ?? "localhost"),
          );
          const response = await api.handle(
            new Request(url, {
              method: incoming.method,
              headers: requestHeaders(incoming.headers),
            }),
          );
          if (!response) return next();
          outgoing.statusCode = response.status;
          response.headers.forEach((value, name) => outgoing.setHeader(name, value));
          outgoing.end(Buffer.from(await response.arrayBuffer()));
        } catch (error) {
          next(error);
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), runsApi()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
});
