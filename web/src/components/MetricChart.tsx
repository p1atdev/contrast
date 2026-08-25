import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import type { RunDetail } from "../api";
import { alignMetric } from "../lib/chartData";

const colors = ["#77e0a5", "#f5b85c", "#67b7ff", "#d88cff", "#ff7d7d", "#d0dc73"];

type Props = {
  runs: RunDetail[];
  metric: string;
};

export function MetricChart({ runs, metric }: Props) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current || runs.length === 0) return;
    const aligned = alignMetric(runs, metric);
    const chart = new uPlot(
      {
        width: Math.max(520, container.current.clientWidth),
        height: 360,
        scales: { x: { time: false } },
        axes: [
          { stroke: "#87929f", grid: { stroke: "#242b33" } },
          { stroke: "#87929f", grid: { stroke: "#242b33" } },
        ],
        series: [
          { label: "step" },
          ...runs.map((run, index) => ({
            label: `${run.config.objective.kind} · seed ${run.id.split("-").at(-1)}`,
            stroke: colors[index % colors.length],
            width: 2,
            spanGaps: true,
          })),
        ],
      },
      [aligned.x, ...aligned.values] as uPlot.AlignedData,
      container.current,
    );
    const resize = new ResizeObserver(([entry]) => {
      chart.setSize({ width: Math.max(520, entry.contentRect.width), height: 360 });
    });
    resize.observe(container.current);
    return () => {
      resize.disconnect();
      chart.destroy();
    };
  }, [metric, runs]);

  if (runs.length === 0) {
    return <div className="chart-empty">比較する run を選択してください。</div>;
  }
  return <div className="chart" ref={container} />;
}
