import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import type { RunDetail } from "../api";
import { alignMetric, positiveOnly, smoothAlignedData } from "../lib/chartData";

const colors = ["#77e0a5", "#f5b85c", "#67b7ff", "#d88cff", "#ff7d7d", "#d0dc73"];

type Props = {
  runs: RunDetail[];
  metric: string;
  height?: number;
  compact?: boolean;
  smoothingWindow?: number;
  logScale?: boolean;
};

export function MetricChart({
  runs,
  metric,
  height = 360,
  compact = false,
  smoothingWindow = 1,
  logScale = false,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const rawAligned = useMemo(() => alignMetric(runs, metric), [metric, runs]);
  const aligned = useMemo(() => {
    const source = logScale ? positiveOnly(rawAligned) : rawAligned;
    return smoothAlignedData(source, smoothingWindow);
  }, [logScale, rawAligned, smoothingWindow]);
  const hasRenderableValues = aligned.values.some((series) =>
    series.some((value) => value !== null),
  );

  useEffect(() => {
    if (!container.current || aligned.x.length === 0 || !hasRenderableValues) return;
    const chart = new uPlot(
      {
        width: Math.max(260, container.current.clientWidth),
        height,
        scales: {
          x: { time: false },
          y: logScale ? { distr: 3, log: 10 } : {},
        },
        axes: [
          {
            stroke: "#bdcbc5",
            ticks: { stroke: "#56645f" },
            grid: { stroke: "#2b3734" },
            font: compact ? "10px ui-monospace" : "11px ui-monospace",
          },
          {
            stroke: "#bdcbc5",
            ticks: { stroke: "#56645f" },
            grid: { stroke: "#2b3734" },
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
  }, [aligned, compact, hasRenderableValues, height, logScale, runs]);

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
  if (!hasRenderableValues) {
    return (
      <div className="chart-empty" style={{ minHeight: height }}>
        対数軸で表示できる正の値がありません。
      </div>
    );
  }
  const className = ["chart", compact ? "compact" : "", logScale ? "log-scale" : ""]
    .filter(Boolean)
    .join(" ");
  return <div className={className} ref={container} />;
}
