import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import type { RunDetail } from "../api";
import { alignMetric } from "../lib/chartData";

const colors = ["#77e0a5", "#f5b85c", "#67b7ff", "#d88cff", "#ff7d7d", "#d0dc73"];

type Props = {
  runs: RunDetail[];
  metric: string;
  height?: number;
  compact?: boolean;
};

export function MetricChart({ runs, metric, height = 360, compact = false }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const aligned = useMemo(() => alignMetric(runs, metric), [metric, runs]);

  useEffect(() => {
    if (!container.current || aligned.x.length === 0) return;
    const chart = new uPlot(
      {
        width: Math.max(260, container.current.clientWidth),
        height,
        scales: { x: { time: false } },
        axes: [
          {
            stroke: "#87929f",
            grid: { stroke: "#242b33" },
            font: compact ? "10px ui-monospace" : "11px ui-monospace",
          },
          {
            stroke: "#87929f",
            grid: { stroke: "#242b33" },
            font: compact ? "10px ui-monospace" : "11px ui-monospace",
          },
        ],
        cursor: { show: !compact },
        series: [
          { label: "step" },
          ...runs.map((run, index) => ({
            label: run.config.objective.kind + " · seed " + run.config.run.seed,
            stroke: colors[index % colors.length],
            width: compact ? 1.5 : 2,
            spanGaps: true,
          })),
        ],
      },
      [aligned.x, ...aligned.values] as uPlot.AlignedData,
      container.current,
    );
    const resize = new ResizeObserver(([entry]) => {
      chart.setSize({ width: Math.max(260, entry.contentRect.width), height });
    });
    resize.observe(container.current);
    return () => {
      resize.disconnect();
      chart.destroy();
    };
  }, [aligned, compact, height, runs]);

  if (runs.length === 0) {
    return (
      <div className="chart-empty" style={{ minHeight: height }}>
        比較する run を選択してください。
      </div>
    );
  }
  if (aligned.x.length === 0) {
    return (
      <div className="chart-empty" style={{ minHeight: height }}>
        このメトリクスはまだ記録されていません。
      </div>
    );
  }
  return <div className={compact ? "chart compact" : "chart"} ref={container} />;
}
