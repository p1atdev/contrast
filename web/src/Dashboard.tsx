import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Check,
  ChevronRight,
  Download,
  Folder,
  Maximize2,
  Search,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  X,
} from "lucide-react";

import { getRun, getRuns, type RunDetail, type RunSummary } from "./api";
import { MetricChart } from "./components/MetricChart";
import { Alert, AlertDescription, AlertTitle } from "./components/ui/alert";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Checkbox } from "./components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "./components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./components/ui/dialog";
import { Input } from "./components/ui/input";
import { ScrollArea } from "./components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Separator } from "./components/ui/separator";
import { Skeleton } from "./components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "./components/ui/tabs";
import { TooltipProvider } from "./components/ui/tooltip";
import {
  buildComparisonCsv,
  filterRuns,
  groupRunsByParent,
  hasMetricData,
  latestMetricValue,
  latestStep,
  SELECTION_LIMIT,
  toggleRunSelection,
  type MetricCategory,
  type MetricDefinition,
} from "./lib/dashboard";

const runColors = ["#5ee6a8", "#f5b85c", "#67b7ff", "#d88cff", "#ff7d7d", "#d0dc73"];

const metrics: MetricDefinition[] = [
  {
    value: "loss",
    label: "Training loss",
    group: "OPTIMIZATION",
    category: "key",
    description: "Within-objective convergence; scales differ across losses",
    format: "decimal",
  },
  {
    value: "eval/backbone_knn_top1",
    label: "Backbone k-NN top-1",
    group: "REPRESENTATION",
    category: "key",
    description: "k-NN on normalized encoder features",
    format: "percent",
  },
  {
    value: "eval/projector_knn_top1",
    label: "Projector k-NN top-1",
    group: "REPRESENTATION",
    category: "representation",
    description: "k-NN on normalized projection-head embeddings",
    format: "percent",
  },
  {
    value: "eval/linear_probe_top1",
    label: "Frozen linear probe",
    group: "REPRESENTATION",
    category: "key",
    description: "Final validation accuracy of the shared frozen-backbone probe",
    format: "percent",
  },
  {
    value: "test/linear_probe_top1",
    label: "Test linear probe",
    group: "FINAL TEST",
    category: "representation",
    description: "Final test accuracy of the same frozen-backbone probe",
    format: "percent",
  },
  {
    value: "eval_ema/backbone_knn_top1",
    label: "EMA backbone k-NN top-1",
    group: "EMA REPRESENTATION",
    category: "ema",
    description: "Validation k-NN on normalized EMA encoder features",
    format: "percent",
  },
  {
    value: "eval_ema/projector_knn_top1",
    label: "EMA projector k-NN top-1",
    group: "EMA REPRESENTATION",
    category: "ema",
    description: "Validation k-NN on normalized EMA projection-head embeddings",
    format: "percent",
  },
  {
    value: "eval_ema/linear_probe_top1",
    label: "EMA frozen linear probe",
    group: "EMA REPRESENTATION",
    category: "ema",
    description: "Final validation accuracy of a probe trained on the frozen EMA backbone",
    format: "percent",
  },
  {
    value: "test_ema/backbone_knn_top1",
    label: "EMA test backbone k-NN",
    group: "EMA FINAL TEST",
    category: "ema",
    description: "Final test k-NN accuracy on normalized EMA encoder features",
    format: "percent",
  },
  {
    value: "test_ema/projector_knn_top1",
    label: "EMA test projector k-NN",
    group: "EMA FINAL TEST",
    category: "ema",
    description: "Final test k-NN accuracy on normalized EMA projection-head embeddings",
    format: "percent",
  },
  {
    value: "test_ema/linear_probe_top1",
    label: "EMA test linear probe",
    group: "EMA FINAL TEST",
    category: "ema",
    description: "Final test accuracy of the frozen EMA-backbone probe",
    format: "percent",
  },
  {
    value: "optimization/gradient_norm",
    label: "Gradient norm",
    group: "OPTIMIZATION",
    category: "optimization",
    description: "Global norm before gradient clipping",
    format: "decimal",
  },
  {
    value: "optimization/gradient_was_clipped",
    label: "Gradient clipped",
    group: "OPTIMIZATION",
    category: "optimization",
    description: "1 when the logged optimizer step exceeded the clip threshold",
    format: "integer",
  },
  {
    value: "optimization/lr",
    label: "Learning rate",
    group: "SCHEDULE",
    category: "optimization",
    description: "Effective learning rate at each optimizer step",
    format: "decimal",
  },
  {
    value: "ema/decay",
    label: "EMA decay",
    group: "EMA SCHEDULE",
    category: "optimization",
    description: "Effective decay used by the latest EMA update",
    format: "decimal",
  },
  {
    value: "ema/updates",
    label: "EMA updates",
    group: "EMA SCHEDULE",
    category: "optimization",
    description: "Number of shadow-model updates completed",
    format: "integer",
  },
  {
    value: "pairs/positive_per_anchor",
    label: "Positives per anchor",
    group: "OBJECTIVE",
    category: "diagnostics",
    description: "Positive-pair count seen by softmax objectives",
    format: "decimal",
  },
  {
    value: "sigmoid/scale",
    label: "Sigmoid scale",
    group: "OBJECTIVE",
    category: "diagnostics",
    description: "Learned Sigmoid-SupCon logit scale",
    format: "decimal",
  },
  {
    value: "sigmoid/bias",
    label: "Sigmoid bias",
    group: "OBJECTIVE",
    category: "diagnostics",
    description: "Learned Sigmoid-SupCon class-prior bias",
    format: "decimal",
  },
  {
    value: "eval/joint_classifier_top1",
    label: "Joint classifier top-1",
    group: "DIAGNOSTIC",
    category: "diagnostics",
    description: "Meaningful only when the objective trains the joint head",
    format: "percent",
  },
  {
    value: "eval/knn_top1",
    label: "Legacy k-NN top-1",
    group: "LEGACY",
    category: "diagnostics",
    description: "Pre-fix projector-space metric from existing runs",
    format: "percent",
  },
  {
    value: "eval/classifier_top1",
    label: "Legacy classifier top-1",
    group: "LEGACY",
    category: "diagnostics",
    description: "Pre-fix joint-head metric from existing runs",
    format: "percent",
  },
];

const categoryLabels: Array<{ value: MetricCategory; label: string }> = [
  { value: "key", label: "主要" },
  { value: "representation", label: "表現" },
  { value: "ema", label: "EMA" },
  { value: "optimization", label: "最適化" },
  { value: "diagnostics", label: "診断" },
  { value: "all", label: "すべて" },
];

function formatMetric(value: number | null, format: MetricDefinition["format"] = "decimal") {
  if (value === null) return "—";
  if (format === "integer") return Math.round(value).toLocaleString();
  if (format === "percent") {
    const normalized = Math.abs(value) <= 1 ? value * 100 : value;
    return `${normalized.toFixed(1)}%`;
  }
  if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
  return value.toFixed(3);
}

type MetricCardProps = {
  definition: MetricDefinition;
  runs: RunDetail[];
  onExpand: (metric: MetricDefinition) => void;
};

function MetricCard({ definition, runs, onExpand }: MetricCardProps) {
  const values = runs
    .map((run) => latestMetricValue(run, definition.value))
    .filter((value) => value !== null);
  const latest = values.at(-1) ?? null;
  return (
    <Card className="metric-card">
      <CardHeader className="metric-card-header">
        <div className="metric-title-row">
          <Badge variant="outline" className="metric-badge">
            {definition.group}
          </Badge>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={`${definition.label}を拡大表示`}
            onClick={() => onExpand(definition)}
          >
            <Maximize2 />
          </Button>
        </div>
        <div>
          <CardTitle>{definition.label}</CardTitle>
          <CardDescription>{definition.description}</CardDescription>
        </div>
        <div className="metric-value-row">
          <strong>{formatMetric(latest, definition.format)}</strong>
          <span>latest selected value</span>
        </div>
      </CardHeader>
      <CardContent className="metric-chart-wrap">
        <button
          type="button"
          className="metric-chart-button"
          aria-label={`${definition.label}を拡大表示`}
          onClick={() => onExpand(definition)}
        >
          <MetricChart runs={runs} metric={definition.value} height={210} compact />
        </button>
      </CardContent>
    </Card>
  );
}

type KpiCardProps = {
  label: string;
  value: string;
  note: string;
  icon: React.ComponentType<{ className?: string }>;
};

function KpiCard({ label, value, note, icon: Icon }: KpiCardProps) {
  return (
    <Card className="kpi-card">
      <CardContent>
        <div className="kpi-icon">
          <Icon />
        </div>
        <div>
          <p>{label}</p>
          <strong>{value}</strong>
          <span>{note}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function AppSkeleton() {
  return (
    <div className="dashboard-shell loading-shell" aria-label="ダッシュボードを読み込み中">
      <div className="dashboard-layout">
        <Skeleton className="h-[620px] w-full" />
        <div className="space-y-4">
          <div className="kpi-grid">
            {Array.from({ length: 4 }, (_, index) => (
              <Skeleton className="h-32" key={index} />
            ))}
          </div>
          <Skeleton className="h-[520px] w-full" />
        </div>
      </div>
    </div>
  );
}

const configRows: Array<{ label: string; value: (run: RunDetail) => string }> = [
  { label: "Objective", value: (run) => run.config.objective.kind },
  {
    label: "Optimizer",
    value: (run) => `${run.config.optimizer.kind} · lr ${run.config.optimizer.lr}`,
  },
  {
    label: "Seeds",
    value: (run) => `run ${run.config.run.seed} / split ${run.config.data.split_seed ?? "legacy"}`,
  },
  {
    label: "Global batch",
    value: (run) =>
      `${run.config.batch.global_source_batch_size} / chunk ${run.config.batch.grad_cache_chunk_size_per_rank}`,
  },
  {
    label: "ViT",
    value: (run) =>
      `d${run.config.model.dim} × ${run.config.model.depth} / h${run.config.model.num_heads}`,
  },
  {
    label: "Precision",
    value: (run) =>
      `${run.config.precision.autocast_dtype}${run.config.precision.allow_tf32 ? " + TF32" : ""}`,
  },
  {
    label: "EMA",
    value: (run) =>
      run.config.ema
        ? `${run.config.ema.enabled ? "on" : "off"} / ${run.config.ema.evaluation_weights} / ${run.config.ema.decay.kind}`
        : "legacy",
  },
];

export default function Dashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [details, setDetails] = useState<Record<string, RunDetail>>({});
  const [expandedMetric, setExpandedMetric] = useState<MetricDefinition | null>(null);
  const [smoothingWindow, setSmoothingWindow] = useState(1);
  const [logScale, setLogScale] = useState(false);
  const [query, setQuery] = useState("");
  const [objective, setObjective] = useState("all");
  const [metricCategory, setMetricCategory] = useState<MetricCategory>("key");
  const [onlyWithData, setOnlyWithData] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectionWarning, setSelectionWarning] = useState(false);

  useEffect(() => {
    getRuns()
      .then((items) => {
        setRuns(items);
        const saved = localStorage.getItem("contrast:selected-runs");
        let restored: string[] = [];
        try {
          const parsed: unknown = saved ? JSON.parse(saved) : [];
          restored = Array.isArray(parsed)
            ? parsed.filter((value): value is string => typeof value === "string")
            : [];
        } catch {
          restored = [];
        }
        const valid = restored
          .filter((id) => items.some((run) => run.id === id))
          .slice(0, SELECTION_LIMIT);
        setSelected(valid.length > 0 ? valid : items.slice(0, 2).map((item) => item.id));
      })
      .catch((reason: unknown) => setError(String(reason)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (runs.length > 0) localStorage.setItem("contrast:selected-runs", JSON.stringify(selected));
  }, [runs.length, selected]);

  useEffect(() => {
    for (const id of selected) {
      if (!(id in details)) {
        void getRun(id)
          .then((detail) => setDetails((current) => ({ ...current, [id]: detail })))
          .catch((reason: unknown) => setError(String(reason)));
      }
    }
  }, [details, selected]);

  const selectedDetails = useMemo(
    () => selected.flatMap((id) => (details[id] ? [details[id]] : [])),
    [details, selected],
  );
  const objectives = useMemo(() => [...new Set(runs.map((run) => run.objective))].sort(), [runs]);
  const filteredRuns = useMemo(() => filterRuns(runs, query, objective), [runs, query, objective]);
  const runGroups = useMemo(() => groupRunsByParent(filteredRuns), [filteredRuns]);
  const visibleMetrics = useMemo(
    () =>
      metrics.filter(
        (metric) =>
          (metricCategory === "all" || metric.category === metricCategory) &&
          (!onlyWithData || hasMetricData(selectedDetails, metric.value)),
      ),
    [metricCategory, onlyWithData, selectedDetails],
  );

  const bestLoss = useMemo(() => {
    const values = selectedDetails
      .map((run) => latestMetricValue(run, "loss"))
      .filter((value) => value !== null);
    return values.length > 0 ? Math.min(...values) : null;
  }, [selectedDetails]);
  const bestBackbone = useMemo(() => {
    const values = selectedDetails
      .map((run) => latestMetricValue(run, "eval/backbone_knn_top1"))
      .filter((value) => value !== null);
    return values.length > 0 ? Math.max(...values) : null;
  }, [selectedDetails]);
  const maxStep = useMemo(() => {
    const values = selectedDetails.map(latestStep).filter((value) => value !== null);
    return values.length > 0 ? Math.max(...values) : null;
  }, [selectedDetails]);
  const comparisonScope = useMemo(() => {
    const objectiveCount = new Set(selectedDetails.map((run) => run.config.objective.kind)).size;
    const parentCount = new Set(
      selectedDetails.map((run) => {
        const separator = run.id.lastIndexOf("/");
        return separator > 0 ? run.id.slice(0, separator) : run.id;
      }),
    ).size;
    const seedCount = new Set(selectedDetails.map((run) => run.config.run.seed)).size;
    return { objectiveCount, parentCount, seedCount };
  }, [selectedDetails]);

  const toggle = (id: string) => {
    const result = toggleRunSelection(selected, id);
    setSelected(result.selected);
    setSelectionWarning(result.limitExceeded);
  };

  const compareGroup = (groupRuns: RunSummary[]) => {
    setSelected(groupRuns.slice(0, SELECTION_LIMIT).map((run) => run.id));
    setSelectionWarning(groupRuns.length > SELECTION_LIMIT);
  };

  const openMetric = (metric: MetricDefinition) => {
    setSmoothingWindow(1);
    setLogScale(false);
    setExpandedMetric(metric);
  };

  const exportCsv = () => {
    if (selectedDetails.length === 0) return;
    const blob = new Blob([buildComparisonCsv(selectedDetails, metrics)], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `contrast-comparison-${new Date().toISOString().slice(0, 10)}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <AppSkeleton />;

  return (
    <TooltipProvider>
      <main className="dashboard-shell">
        {(error || selectionWarning) && (
          <Alert variant={error ? "destructive" : "default"} className="dashboard-alert">
            <Activity />
            <AlertTitle>{error ? "データを読み込めませんでした" : "比較は最大6件です"}</AlertTitle>
            <AlertDescription>
              {error || "現在の選択はそのまま保持しました。別の run を外してから追加してください。"}
            </AlertDescription>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="通知を閉じる"
              onClick={() => {
                setError("");
                setSelectionWarning(false);
              }}
            >
              <X />
            </Button>
          </Alert>
        )}

        <section className="dashboard-layout">
          <Card className="runs-panel">
            <CardHeader>
              <div className="panel-title-row">
                <div>
                  <CardTitle>Runs</CardTitle>
                  <CardDescription>
                    {runs.length} recorded · 最大{SELECTION_LIMIT}件
                  </CardDescription>
                </div>
                <div className="panel-title-actions">
                  <Badge variant="secondary">{selected.length} selected</Badge>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={exportCsv}
                    disabled={selectedDetails.length === 0}
                  >
                    <Download /> CSV
                  </Button>
                </div>
              </div>
              <div className="run-filters">
                <div className="search-field">
                  <Search />
                  <Input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="run・seed・objectiveを検索"
                    aria-label="runを検索"
                  />
                </div>
                <Select value={objective} onValueChange={setObjective}>
                  <SelectTrigger className="w-full">
                    <SlidersHorizontal />
                    <SelectValue placeholder="Objective" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">すべての objective</SelectItem>
                    {objectives.map((item) => (
                      <SelectItem value={item} key={item}>
                        {item}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="selection-actions">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setSelected(filteredRuns.slice(0, SELECTION_LIMIT).map((run) => run.id))
                  }
                  disabled={filteredRuns.length === 0}
                >
                  表示中から選択
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelected([])}
                  disabled={selected.length === 0}
                >
                  クリア
                </Button>
              </div>
            </CardHeader>
            <Separator />
            <CardContent className="run-list-content">
              <ScrollArea className="run-scroll">
                <div className="run-groups">
                  {runGroups.map((group) => (
                    <Collapsible defaultOpen className="run-group" key={group.parent}>
                      <div className="run-group-heading">
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" className="run-group-toggle" title={group.parent}>
                            <ChevronRight className="run-group-chevron" />
                            <Folder />
                            <span>{group.parent}</span>
                            <Badge variant="outline">{group.runs.length}</Badge>
                          </Button>
                        </CollapsibleTrigger>
                        <Button
                          variant="ghost"
                          size="xs"
                          aria-label={`${group.parent}のrunを比較`}
                          onClick={() => compareGroup(group.runs)}
                        >
                          比較
                        </Button>
                      </div>
                      <CollapsibleContent>
                        <div className="run-group-list">
                          {group.runs.map((run) => {
                            const active = selected.includes(run.id);
                            const colorIndex = selected.indexOf(run.id);
                            return (
                              <button
                                type="button"
                                className={active ? "run-card selected" : "run-card"}
                                aria-pressed={active}
                                key={run.id}
                                onClick={() => toggle(run.id)}
                              >
                                <span
                                  className="run-check"
                                  style={active ? { background: runColors[colorIndex] } : undefined}
                                >
                                  {active && <Check />}
                                </span>
                                <span className="run-copy">
                                  <span className="run-title">
                                    <strong>{run.objective}</strong>
                                    <Badge variant="outline">seed {run.seed}</Badge>
                                  </span>
                                  <small className="run-id">
                                    {run.id.split("/").at(-1)} · {run.strategy}
                                  </small>
                                </span>
                                <span className="run-step">
                                  <small>step</small>
                                  <strong>{run.latest?.step.toLocaleString() ?? "—"}</strong>
                                </span>
                              </button>
                            );
                          })}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  ))}
                  {runGroups.length === 0 && (
                    <div className="empty-state">
                      <Search />
                      <p>条件に合う run がありません。</p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          <div className="workspace">
            <section className="kpi-grid" aria-label="比較サマリー">
              <KpiCard
                label="Comparison scope"
                value={
                  comparisonScope.objectiveCount > 0
                    ? `${comparisonScope.objectiveCount} objectives`
                    : "—"
                }
                note={
                  comparisonScope.objectiveCount > 0
                    ? `${comparisonScope.parentCount} experiment · ${comparisonScope.seedCount} seed`
                    : "runを選択してください"
                }
                icon={Target}
              />
              <KpiCard
                label="Latest step"
                value={maxStep?.toLocaleString() ?? "—"}
                note="furthest selected run"
                icon={Activity}
              />
              <KpiCard
                label="Best loss"
                value={formatMetric(bestLoss)}
                note="lower is better within objective"
                icon={TrendingDown}
              />
              <KpiCard
                label="Best backbone k-NN"
                value={formatMetric(bestBackbone, "percent")}
                note="validation top-1"
                icon={TrendingUp}
              />
            </section>

            <Card className="metrics-panel">
              <CardHeader className="metrics-header">
                <div>
                  <p className="eyebrow">ALIGNED BY OPTIMIZER STEP</p>
                  <CardTitle>Metric overview</CardTitle>
                  <CardDescription>
                    目的に合わせて指標を絞り、グラフから詳細へ移動できます。
                  </CardDescription>
                </div>
                <label className="data-toggle" htmlFor="only-with-data">
                  <Checkbox
                    id="only-with-data"
                    checked={onlyWithData}
                    onCheckedChange={(checked) => setOnlyWithData(checked === true)}
                  />
                  データがある指標のみ
                </label>
              </CardHeader>
              <CardContent>
                <Tabs
                  value={metricCategory}
                  onValueChange={(value) => setMetricCategory(value as MetricCategory)}
                >
                  <TabsList className="metric-tabs" variant="line">
                    {categoryLabels.map((category) => (
                      <TabsTrigger value={category.value} key={category.value}>
                        {category.label}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                </Tabs>
                {selected.length === 0 ? (
                  <div className="empty-state metric-empty">
                    <Target />
                    <h3>比較する run を選択してください</h3>
                    <p>左の一覧から最大6件まで選択できます。</p>
                  </div>
                ) : selectedDetails.length < selected.length ? (
                  <div className="metric-grid">
                    {Array.from({ length: 4 }, (_, index) => (
                      <Skeleton className="h-[360px]" key={index} />
                    ))}
                  </div>
                ) : visibleMetrics.length > 0 ? (
                  <div className="metric-grid">
                    {visibleMetrics.map((definition) => (
                      <MetricCard
                        definition={definition}
                        runs={selectedDetails}
                        onExpand={openMetric}
                        key={definition.value}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="empty-state metric-empty">
                    <Sparkles />
                    <h3>このカテゴリには記録がありません</h3>
                    <p>「データがある指標のみ」を外すと、すべての候補を確認できます。</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="config-panel">
              <CardHeader>
                <div className="panel-title-row">
                  <div>
                    <CardTitle>Controlled variables</CardTitle>
                    <CardDescription>列を見比べ、条件差をすぐに確認できます。</CardDescription>
                  </div>
                  <Badge variant="outline">
                    <BarChart3 /> resolved config
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                {selectedDetails.length === 0 ? (
                  <div className="empty-state compact">
                    <p>run を選択すると設定比較が表示されます。</p>
                  </div>
                ) : (
                  <div className="table-wrap">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Variable</TableHead>
                          {selectedDetails.map((run, index) => (
                            <TableHead key={run.id}>
                              <span className="run-column-title">
                                <i style={{ background: runColors[index] }} />
                                {run.config.objective.kind} · s{run.config.run.seed}
                              </span>
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {configRows.map((row) => {
                          const values = selectedDetails.map(row.value);
                          const differs = new Set(values).size > 1;
                          return (
                            <TableRow key={row.label}>
                              <TableCell className="config-label">
                                {row.label}
                                {differs && <Badge variant="secondary">diff</Badge>}
                              </TableCell>
                              {values.map((value, index) => (
                                <TableCell
                                  className={differs ? "config-diff" : undefined}
                                  key={selectedDetails[index].id}
                                >
                                  {value}
                                </TableCell>
                              ))}
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </section>

        <Dialog
          open={expandedMetric !== null}
          onOpenChange={(open) => {
            if (!open) setExpandedMetric(null);
          }}
        >
          {expandedMetric && (
            <DialogContent className="metric-dialog">
              <DialogHeader>
                <Badge variant="outline" className="metric-badge">
                  {expandedMetric.group}
                </Badge>
                <DialogTitle>{expandedMetric.label}</DialogTitle>
                <DialogDescription>{expandedMetric.description}</DialogDescription>
              </DialogHeader>
              <div className="chart-toolbar" role="group" aria-label="グラフ表示オプション">
                <div className="chart-control">
                  <span>スムージング</span>
                  <Select
                    value={String(smoothingWindow)}
                    onValueChange={(value) => setSmoothingWindow(Number(value))}
                  >
                    <SelectTrigger
                      size="sm"
                      className="chart-control-select"
                      aria-label="移動平均の点数"
                    >
                      <Activity />
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">なし</SelectItem>
                      <SelectItem value="5">移動平均 5点</SelectItem>
                      <SelectItem value="20">移動平均 20点</SelectItem>
                      <SelectItem value="50">移動平均 50点</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  type="button"
                  variant={logScale ? "secondary" : "outline"}
                  size="sm"
                  aria-pressed={logScale}
                  title="対数軸では0以下の値を表示しません"
                  onClick={() => setLogScale((current) => !current)}
                >
                  <TrendingUp />
                  Y軸 log10
                </Button>
                <p className="chart-toolbar-note">
                  {logScale ? "0以下の値は対数軸から除外されます" : "Y軸は線形スケールです"}
                </p>
              </div>
              <div className="dialog-chart">
                <MetricChart
                  runs={selectedDetails}
                  metric={expandedMetric.value}
                  height={520}
                  smoothingWindow={smoothingWindow}
                  logScale={logScale}
                />
              </div>
            </DialogContent>
          )}
        </Dialog>
      </main>
    </TooltipProvider>
  );
}
