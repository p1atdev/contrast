import { stat } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { serveStatic } from "hono/bun";

import { createRunsApi } from "./server/runs";

const defaults = {
  runsDirectory: resolve(import.meta.dir, "..", "runs"),
  host: "127.0.0.1",
  port: "8000",
};
const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    "runs-dir": { type: "string", default: defaults.runsDirectory },
    host: { type: "string", default: defaults.host },
    port: { type: "string", default: defaults.port },
  },
  strict: true,
});
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error("invalid port: " + values.port);
}

const distribution = resolve(import.meta.dir, "dist");
const index = resolve(distribution, "index.html");
try {
  if (!(await stat(index)).isFile()) throw new Error();
} catch {
  throw new Error("web/dist is missing; run bun run build first");
}

const api = createRunsApi(values["runs-dir"]);
api.app.use("*", async (context, next) => {
  if (context.req.method !== "GET" && context.req.method !== "HEAD") {
    return context.json({ detail: "method not allowed" }, 405);
  }
  await next();
});
api.app.use("*", serveStatic({ root: distribution }));
api.app.get("*", serveStatic({ root: distribution, path: "index.html" }));

const server = Bun.serve({
  hostname: values.host,
  port,
  fetch: api.app.fetch,
});

console.log("Contrast Lab: " + server.url);
console.log("Runs: " + api.root);
