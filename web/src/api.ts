export type MetricEvent = {
  type: string;
  step: number;
  epoch: number;
  [key: string]: string | number | boolean | null;
};

export type RunSummary = {
  id: string;
  experiment: string;
  seed: number;
  objective: string;
  strategy: string;
  latest: MetricEvent | null;
};

export type RunDetail = {
  id: string;
  config: {
    run: { seed: number };
    data: { split_seed?: number };
    objective: { kind: string };
    optimizer: { kind: string; lr: number };
    batch: { global_source_batch_size: number; grad_cache_chunk_size_per_rank: number };
    model: { dim: number; depth: number; num_heads: number };
    precision: { autocast_dtype: string; allow_tf32: boolean };
    ema?: {
      enabled: boolean;
      evaluation_weights: "raw" | "ema" | "both";
      decay: {
        kind: "constant" | "linear" | "cosine" | "inverse_power";
      };
    };
  };
  environment: Record<string, unknown>;
  metrics: MetricEvent[];
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const getRuns = () => getJson<RunSummary[]>("/api/runs");
export const getRun = (id: string) => getJson<RunDetail>(`/api/run/${id}`);
