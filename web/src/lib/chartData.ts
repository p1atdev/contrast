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

export function smoothAlignedData(data: AlignedChartData, windowSize: number): AlignedChartData {
  const normalizedWindow = Math.max(1, Math.floor(windowSize));
  if (normalizedWindow === 1) return data;

  return {
    x: data.x,
    values: data.values.map((series) => {
      const window: number[] = [];
      let sum = 0;
      return series.map((value) => {
        if (value === null) return null;
        window.push(value);
        sum += value;
        if (window.length > normalizedWindow) sum -= window.shift() ?? 0;
        return sum / window.length;
      });
    }),
  };
}

export function positiveOnly(data: AlignedChartData): AlignedChartData {
  return {
    x: data.x,
    values: data.values.map((series) =>
      series.map((value) => (value !== null && value > 0 ? value : null)),
    ),
  };
}
