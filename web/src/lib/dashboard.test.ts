import { describe, expect, test } from "bun:test";

import type { RunDetail, RunSummary } from "../api";
import {
  buildComparisonCsv,
  filterRuns,
  groupRunsByParent,
  hasMetricData,
  latestMetricValue,
  latestStep,
  toggleRunSelection,
  type MetricDefinition,
} from "./dashboard";

const summaries: RunSummary[] = [
  {
    id: "pilot/alpha",
    experiment: "CIFAR Pilot",
    seed: 0,
    objective: "supcon",
    strategy: "direct",
    latest: null,
  },
  {
    id: "core/beta",
    experiment: "Core Sweep",
    seed: 1,
    objective: "ce",
    strategy: "grad_cache",
    latest: null,
  },
];

const detail = (id: string, metrics: RunDetail["metrics"]): RunDetail =>
  ({
    id,
    config: {
      run: { seed: 0 },
      data: { split_seed: 0 },
      objective: { kind: 'sup,con "quoted"' },
      optimizer: { kind: "adamw", lr: 0.001 },
      batch: { global_source_batch_size: 256, grad_cache_chunk_size_per_rank: 128 },
      model: { dim: 192, depth: 12, num_heads: 3 },
      precision: { autocast_dtype: "bfloat16", allow_tf32: true },
    },
    environment: {},
    metrics,
  }) as RunDetail;

describe("dashboard helpers", () => {
  test("filters runs by case-insensitive text and objective", () => {
    expect(filterRuns(summaries, "CIFAR", "supcon").map((run) => run.id)).toEqual(["pilot/alpha"]);
    expect(filterRuns(summaries, "CACHE", "all").map((run) => run.id)).toEqual(["core/beta"]);
  });

  test("keeps the existing selection when the limit is reached", () => {
    const selected = ["a", "b", "c", "d", "e", "f"];
    expect(toggleRunSelection(selected, "g")).toEqual({ selected, limitExceeded: true });
    expect(toggleRunSelection(selected, "c")).toEqual({
      selected: ["a", "b", "d", "e", "f"],
      limitExceeded: false,
    });
  });

  test("groups runs by their parent directory", () => {
    expect(groupRunsByParent(summaries).map((group) => [group.parent, group.runs.length])).toEqual([
      ["pilot", 1],
      ["core", 1],
    ]);
  });

  test("uses the latest finite metric value and maximum step", () => {
    const run = detail("pilot/alpha", [
      { type: "train", step: 10, epoch: 0, loss: 2 },
      { type: "train", step: 20, epoch: 0, loss: 1 },
      { type: "run_end", step: 20, epoch: 0 },
    ]);
    expect(latestMetricValue(run, "loss")).toBe(1);
    expect(latestMetricValue(run, "missing")).toBeNull();
    expect(latestStep(run)).toBe(20);
    expect(hasMetricData([run], "loss")).toBe(true);
  });

  test("quotes CSV cells and leaves missing metrics empty", () => {
    const metric: MetricDefinition = {
      value: "loss",
      label: "Loss",
      group: "Optimization",
      category: "optimization",
      description: "Training loss",
    };
    const csv = buildComparisonCsv(
      [detail("pilot/run,one", [{ type: "train", step: 1, epoch: 0 }])],
      [metric],
    );
    expect(csv).toContain('"pilot/run,one"');
    expect(csv).toContain('"sup,con ""quoted"""');
    expect(csv.endsWith(",\n")).toBe(true);
  });
});
