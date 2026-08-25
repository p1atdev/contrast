import { useEffect, useMemo, useState } from "react";

import { getRun, getRuns, type RunDetail, type RunSummary } from "./api";
import { MetricChart } from "./components/MetricChart";

const metrics = [
  ["loss", "Training loss"],
  ["eval/knn_top1", "k-NN top-1"],
  ["eval/classifier_top1", "Classifier top-1"],
  ["optimization/lr", "Learning rate"],
] as const;

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [details, setDetails] = useState<Record<string, RunDetail>>({});
  const [metric, setMetric] = useState("loss");
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
          <section className="panel graph-panel">
            <div className="section-title">
              <div>
                <p className="eyebrow">ALIGNED BY OPTIMIZER STEP</p>
                <h2>Metric comparison</h2>
              </div>
              <select value={metric} onChange={(event) => setMetric(event.target.value)}>
                {metrics.map(([value, label]) => (
                  <option value={value} key={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <MetricChart runs={selectedDetails} metric={metric} />
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
    </main>
  );
}
