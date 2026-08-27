import type { RunDetail, RunSummary } from "../api";

export const SELECTION_LIMIT = 6;

export type MetricCategory =
  | "key"
  | "representation"
  | "ema"
  | "optimization"
  | "diagnostics"
  | "all";

export type MetricDefinition = {
  value: string;
  label: string;
  group: string;
  category: Exclude<MetricCategory, "all">;
  description: string;
  format?: "decimal" | "percent" | "integer";
};

export type SelectionResult = {
  selected: string[];
  limitExceeded: boolean;
};

export type RunGroup = {
  parent: string;
  runs: RunSummary[];
};

export function groupRunsByParent(runs: RunSummary[]): RunGroup[] {
  const groups = new Map<string, RunSummary[]>();
  for (const run of runs) {
    const separator = run.id.lastIndexOf("/");
    const parent = separator > 0 ? run.id.slice(0, separator) : run.experiment;
    groups.set(parent, [...(groups.get(parent) ?? []), run]);
  }
  return [...groups].map(([parent, groupedRuns]) => ({ parent, runs: groupedRuns }));
}

export function filterRuns(runs: RunSummary[], query: string, objective: string): RunSummary[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  return runs.filter((run) => {
    if (objective !== "all" && run.objective !== objective) return false;
    if (!normalizedQuery) return true;
    return [run.id, run.experiment, run.objective, run.strategy, String(run.seed)].some((value) =>
      value.toLocaleLowerCase().includes(normalizedQuery),
    );
  });
}

export function toggleRunSelection(
  selected: string[],
  id: string,
  limit = SELECTION_LIMIT,
): SelectionResult {
  if (selected.includes(id)) {
    return { selected: selected.filter((item) => item !== id), limitExceeded: false };
  }
  if (selected.length >= limit) return { selected, limitExceeded: true };
  return { selected: [...selected, id], limitExceeded: false };
}

export function latestMetricValue(run: RunDetail, metric: string): number | null {
  for (let index = run.metrics.length - 1; index >= 0; index -= 1) {
    const value = run.metrics[index]?.[metric];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

export function latestStep(run: RunDetail): number | null {
  let result: number | null = null;
  for (const event of run.metrics) {
    if (Number.isFinite(event.step)) result = Math.max(result ?? event.step, event.step);
  }
  return result;
}

export function hasMetricData(runs: RunDetail[], metric: string): boolean {
  return runs.some((run) => latestMetricValue(run, metric) !== null);
}

function csvCell(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function buildComparisonCsv(runs: RunDetail[], metrics: MetricDefinition[]): string {
  const headers = [
    "run",
    "objective",
    "seed",
    "optimizer",
    "learning_rate",
    "global_batch",
    "model_dim",
    "model_depth",
    "precision",
    "ema",
    ...metrics.map((metric) => metric.value),
  ];
  const rows = runs.map((run) => [
    run.id,
    run.config.objective.kind,
    run.config.run.seed,
    run.config.optimizer.kind,
    run.config.optimizer.lr,
    run.config.batch.global_source_batch_size,
    run.config.model.dim,
    run.config.model.depth,
    run.config.precision.autocast_dtype,
    run.config.ema?.enabled ?? false,
    ...metrics.map((metric) => latestMetricValue(run, metric.value)),
  ]);
  return [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}
