import { afterEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createRunsApi } from "./runs";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true })));
});

async function fixture(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "contrast-runs-"));
  temporaryDirectories.push(root);
  const runDirectory = join(root, "experiment", "20260826T000000Z-0003");
  await mkdir(runDirectory, { recursive: true });
  await writeFile(
    join(runDirectory, "config.json"),
    JSON.stringify({
      run: { experiment: "experiment", seed: 3 },
      data: { split_seed: 0 },
      objective: { kind: "supervised_contrastive" },
      optimizer: { kind: "adamw", lr: 0.001 },
      training: { step_strategy: "grad_cache" },
      batch: { global_source_batch_size: 64, grad_cache_chunk_size_per_rank: 16 },
      model: { dim: 192, depth: 12, num_heads: 3 },
      precision: { autocast_dtype: "bfloat16", allow_tf32: true },
    }),
  );
  await writeFile(join(runDirectory, "environment.json"), JSON.stringify({ device: "cpu" }));
  await writeFile(
    join(runDirectory, "metrics.jsonl"),
    JSON.stringify({ type: "train", step: 1, epoch: 0, loss: 2.5 }) + "\n",
  );
  return root;
}

describe("runs API", () => {
  test("lists runs and returns their details", async () => {
    const api = createRunsApi(await fixture());

    const list = await api.handle(new Request("http://localhost/api/runs"));
    expect(list?.status).toBe(200);
    expect(await list?.json()).toEqual([
      {
        id: "experiment/20260826T000000Z-0003",
        experiment: "experiment",
        seed: 3,
        objective: "supervised_contrastive",
        strategy: "grad_cache",
        latest: { type: "train", step: 1, epoch: 0, loss: 2.5 },
      },
    ]);

    const detail = await api.handle(
      new Request("http://localhost/api/run/experiment/20260826T000000Z-0003"),
    );
    expect(detail?.status).toBe(200);
    const body = await detail?.json();
    expect(body.environment).toEqual({ device: "cpu" });
  });

  test("returns an empty list for a missing runs directory", async () => {
    const root = await fixture();
    const api = createRunsApi(join(root, "missing"));
    const response = await api.handle(new Request("http://localhost/api/runs"));

    expect(await response?.json()).toEqual([]);
  });

  test("rejects paths outside the runs directory", async () => {
    const api = createRunsApi(await fixture());
    const response = await api.handle(
      new Request("http://localhost/api/run/%2E%2E%2Fpackage.json"),
    );

    expect(response?.status).toBe(404);
  });

  test("lets Hono handle methods and API misses", async () => {
    const api = createRunsApi(await fixture());
    const method = await api.handle(new Request("http://localhost/api/runs", { method: "POST" }));
    const missing = await api.handle(new Request("http://localhost/api/unknown"));

    expect(method?.status).toBe(405);
    expect(await method?.json()).toEqual({ detail: "method not allowed" });
    expect(missing?.status).toBe(404);
    expect(await api.handle(new Request("http://localhost/dashboard"))).toBeNull();
  });
});
