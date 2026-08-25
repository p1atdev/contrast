import type { RunDetail } from "../api";

export type AlignedChartData = {
  x: number[];
  values: Array<Array<number | null>>;
};

export function alignMetric(runs: RunDetail[], metric: string): AlignedChartData {
  const points = runs.map((run) => {
    const byStep = new Map<number, number>();
    for (const event of run.metrics) {
      const value = event[metric];
      if (typeof value === "number" && Number.isFinite(value)) {
        byStep.set(event.step, value);
      }
    }
    return byStep;
  });
  const x = [...new Set(points.flatMap((series) => [...series.keys()]))].sort((a, b) => a - b);
  return {
    x,
    values: points.map((series) => x.map((step) => series.get(step) ?? null)),
  };
}
