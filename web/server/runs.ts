import { readdir, readFile, realpath, stat } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

import { Hono } from "hono";

import type { MetricEvent, RunDetail, RunSummary } from "../src/api";

type StoredConfig = RunDetail["config"] & {
  run: RunDetail["config"]["run"] & { experiment: string };
  training: { step_strategy: string };
};

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, "utf8")) as T;
}

function isMissingFile(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

async function readMetrics(path: string): Promise<MetricEvent[]> {
  try {
    const contents = await readFile(path, "utf8");
    return contents
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as MetricEvent);
  } catch (error) {
    if (isMissingFile(error)) return [];
    throw error;
  }
}

async function directories(path: string): Promise<string[]> {
  try {
    return (await readdir(path, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);
  } catch (error) {
    if (isMissingFile(error)) return [];
    throw error;
  }
}

async function listRuns(runsDirectory: string): Promise<RunSummary[]> {
  const found: RunSummary[] = [];
  for (const experiment of await directories(runsDirectory)) {
    const experimentDirectory = join(runsDirectory, experiment);
    for (const run of await directories(experimentDirectory)) {
      const directory = join(experimentDirectory, run);
      const configPath = join(directory, "config.json");
      let config: StoredConfig;
      try {
        config = await readJson<StoredConfig>(configPath);
      } catch (error) {
        if (isMissingFile(error)) continue;
        throw error;
      }
      const metrics = await readMetrics(join(directory, "metrics.jsonl"));
      found.push({
        id: experiment + "/" + run,
        experiment: config.run.experiment,
        seed: config.run.seed,
        objective: config.objective.kind,
        strategy: config.training.step_strategy,
        latest: metrics.at(-1) ?? null,
      });
    }
  }
  return found.sort((left, right) => right.id.localeCompare(left.id));
}

async function safeRunDirectory(runsDirectory: string, runId: string): Promise<string | null> {
  try {
    const root = await realpath(runsDirectory);
    const candidate = await realpath(resolve(root, runId));
    const pathFromRoot = relative(root, candidate);
    if (!pathFromRoot || pathFromRoot.startsWith(".." + sep)) return null;
    if (isAbsolute(pathFromRoot) || !(await stat(join(candidate, "config.json"))).isFile()) {
      return null;
    }
    return candidate;
  } catch (error) {
    if (isMissingFile(error)) return null;
    throw error;
  }
}

async function getRun(runsDirectory: string, runId: string): Promise<RunDetail | null> {
  const directory = await safeRunDirectory(runsDirectory, runId);
  if (!directory) return null;
  const environmentPath = join(directory, "environment.json");
  let environment: Record<string, unknown> = {};
  try {
    environment = await readJson<Record<string, unknown>>(environmentPath);
  } catch (error) {
    if (!isMissingFile(error)) throw error;
  }
  return {
    id: runId,
    config: await readJson<RunDetail["config"]>(join(directory, "config.json")),
    environment,
    metrics: await readMetrics(join(directory, "metrics.jsonl")),
  };
}

export function createRunsApi(runsDirectory: string) {
  const root = resolve(runsDirectory);
  const app = new Hono();

  app.get("/api/runs", async (context) => context.json(await listRuns(root)));
  app.get("/api/run/:experiment/:run", async (context) => {
    const runId = context.req.param("experiment") + "/" + context.req.param("run");
    const run = runId ? await getRun(root, runId) : null;
    return run ? context.json(run) : context.json({ detail: "run not found" }, 404);
  });
  app.all("/api", (context) => context.json({ detail: "not found" }, 404));
  app.all("/api/*", (context) =>
    context.req.method === "GET" || context.req.method === "HEAD"
      ? context.json({ detail: "not found" }, 404)
      : context.json({ detail: "method not allowed" }, 405),
  );

  return {
    root,
    app,
    async handle(request: Request): Promise<Response | null> {
      const pathname = new URL(request.url).pathname;
      if (pathname !== "/api" && !pathname.startsWith("/api/")) return null;
      return app.fetch(request);
    },
  };
}
