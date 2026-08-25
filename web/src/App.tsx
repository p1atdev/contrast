import { useEffect, useMemo, useState } from "react";

import { getRun, getRuns, type RunDetail, type RunSummary } from "./api";
import { MetricChart } from "./components/MetricChart";

type MetricDefinition = {
  value: string;
  label: string;
  group: string;
  description: string;
};

const metrics: MetricDefinition[] = [
  {
    value: "loss",
    label: "Training loss",
    group: "OPTIMIZATION",
    description: "Objective value over optimizer steps",
  },
  {
    value: "eval/knn_top1",
    label: "k-NN top-1",
    group: "REPRESENTATION",
    description: "Neighborhood accuracy of learned embeddings",
  },
  {
    value: "eval/classifier_top1",
    label: "Classifier top-1",
    group: "CLASSIFICATION HEAD",
    description: "Top-1 accuracy of the jointly trained head",
  },
  {
    value: "optimization/lr",
    label: "Learning rate",
    group: "SCHEDULE",
    description: "Effective learning rate at each optimizer step",
  },
];

type MetricCardProps = {
  definition: MetricDefinition;
  runs: RunDetail[];
  onExpand: (metric: MetricDefinition) => void;
};

function MetricCard({ definition, runs, onExpand }: MetricCardProps) {
  return (
    <article className="metric-card">
      <button
        type="button"
        className="metric-card-open"
        aria-label={definition.label + "を拡大表示"}
        aria-haspopup="dialog"
        onClick={() => onExpand(definition)}
      />
      <div className="metric-card-heading">
        <div>
          <p className="metric-group">{definition.group}</p>
          <h3>{definition.label}</h3>
          <p className="metric-description">{definition.description}</p>
        </div>
        <span className="expand-label" aria-hidden="true">
          拡大 ↗
        </span>
      </div>
      <MetricChart runs={runs} metric={definition.value} height={240} compact />
    </article>
  );
}

type MetricModalProps = {
  definition: MetricDefinition;
  runs: RunDetail[];
  onClose: () => void;
};

function MetricModal({ definition, runs, onClose }: MetricModalProps) {
  return (
    <div className="metric-modal-layer">
      <button
        type="button"
        className="metric-modal-backdrop"
        aria-label="拡大表示を閉じる"
        onClick={onClose}
      />
      <section
        className="metric-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="expanded-metric-title"
      >
        <div className="metric-modal-heading">
          <div>
            <p className="metric-group">{definition.group}</p>
            <h2 id="expanded-metric-title">{definition.label}</h2>
            <p className="metric-description">{definition.description}</p>
          </div>
          <button type="button" className="modal-close" onClick={onClose} autoFocus>
            <span aria-hidden="true">×</span>
            <span className="sr-only">閉じる</span>
          </button>
        </div>
        <MetricChart runs={runs} metric={definition.value} height={540} />
      </section>
    </div>
  );
}

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [details, setDetails] = useState<Record<string, RunDetail>>({});
  const [expandedMetric, setExpandedMetric] = useState<MetricDefinition | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getRuns()
      .then((items) => {
        setRuns(items);
        setSelected(items.slice(0, 2).map((item) => item.id));
      })
      .catch((reason: unknown) => setError(String(reason)));
  }, []);

  useEffect(() => {
    for (const id of selected) {
      if (!(id in details)) {
        getRun(id)
          .then((detail) => setDetails((current) => ({ ...current, [id]: detail })))
          .catch((reason: unknown) => setError(String(reason)));
      }
    }
  }, [details, selected]);

  useEffect(() => {
    if (!expandedMetric) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpandedMetric(null);
    };
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [expandedMetric]);

  const selectedDetails = useMemo(
    () => selected.flatMap((id) => (details[id] ? [details[id]] : [])),
    [details, selected],
  );

  const toggle = (id: string) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(-6),
    );
  };

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">SUPERVISED CONTRASTIVE WORKBENCH</p>
          <h1>Contrast Lab</h1>
        </div>
        <div className="run-count">
          <strong>{runs.length}</strong>
          <span>recorded runs</span>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="layout">
        <aside>
          <div className="section-title">
            <h2>Runs</h2>
            <span>最大6件</span>
          </div>
          <div className="run-list">
            {runs.map((run) => (
              <button
                className={selected.includes(run.id) ? "run-card selected" : "run-card"}
                key={run.id}
                onClick={() => toggle(run.id)}
              >
                <span className="check">{selected.includes(run.id) ? "●" : "○"}</span>
                <span>
                  <strong>{run.objective}</strong>
                  <small>
                    seed {run.seed} · {run.strategy}
                  </small>
                </span>
                <em>{typeof run.latest?.loss === "number" ? run.latest.loss.toFixed(3) : "—"}</em>
              </button>
            ))}
            {runs.length === 0 && <p className="empty">まだ run がありません。</p>}
          </div>
        </aside>

        <div className="workspace">
          <section className="panel metrics-panel">
            <div className="section-title">
              <div>
                <p className="eyebrow">ALIGNED BY OPTIMIZER STEP</p>
                <h2>Metric overview</h2>
              </div>
              <span>グラフをクリックして拡大</span>
            </div>
            <div className="metric-grid">
              {metrics.map((definition) => (
                <MetricCard
                  definition={definition}
                  runs={selectedDetails}
                  onExpand={setExpandedMetric}
                  key={definition.value}
                />
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="section-title">
              <h2>Controlled variables</h2>
              <span>resolved config</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Objective</th>
                    <th>Optimizer</th>
                    <th>Global batch</th>
                    <th>ViT</th>
                    <th>Precision</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedDetails.map((run) => (
                    <tr key={run.id}>
                      <td>{run.id.split("/").at(-1)}</td>
                      <td>{run.config.objective.kind}</td>
                      <td>{run.config.optimizer.kind}</td>
                      <td>
                        {run.config.batch.global_source_batch_size} / chunk{" "}
                        {run.config.batch.grad_cache_chunk_size_per_rank}
                      </td>
                      <td>
                        d{run.config.model.dim} × {run.config.model.depth} / h
                        {run.config.model.num_heads}
                      </td>
                      <td>
                        {run.config.precision.autocast_dtype}
                        {run.config.precision.allow_tf32 ? " + TF32" : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </section>

      {expandedMetric && (
        <MetricModal
          definition={expandedMetric}
          runs={selectedDetails}
          onClose={() => setExpandedMetric(null)}
        />
      )}
    </main>
  );
}
