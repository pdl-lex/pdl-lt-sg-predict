// Dünner Client über die FastAPI. Eine Origin via Vite-Proxy (/api).

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* keine JSON-Antwort */ }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
}

// ── Typen ────────────────────────────────────────────────────────────────────
export interface ModelType { code: string; name: string }
export interface AppConfig { enable_training: boolean; models_dir: string; model_types: ModelType[] }

export type ModelRow = Record<string, string | number | boolean>
export interface BestModel { model_file: string; model_name: string; accuracy: number; stem: string }
export interface ModelsResponse { models: ModelRow[]; files: string[]; best: BestModel | null; count: number }

export interface TopEntry { label: string; description: string; proba: number | null; is_best: boolean }
export interface SingleResponse {
  prediction: string; description: string; top: TopEntry[]; model_type: string; uses_lemma: boolean
}
export interface ShapWord { word: string; score: number }
export interface ShapResponse { lemma: ShapWord[]; bedeutung: ShapWord[]; is_nn: boolean }
export interface BatchResponse { rows: Record<string, string>[]; count: number; uses_lemma: boolean }

export interface SachgruppenResponse {
  rows: Record<string, string | number>[]; model_name: string; model_file: string; accuracy: string
}

export interface TrainingCsvInfo {
  filename: string; num_samples: number; num_classes: number; time_per_type: Record<string, number>
}
export interface TrainingInfo { enable_training: boolean; csv: TrainingCsvInfo | null; running: boolean }
export interface TrainingStatus {
  state: 'idle' | 'running' | 'done' | 'error'
  mode?: 'single' | 'batch'
  pct?: number
  msg?: string
  done?: number
  total?: number
  error?: string
  model_file?: string
  accuracy?: number
  training_time?: number
  best_cv_score?: number
  best_params?: Record<string, unknown>
}

// ── API ──────────────────────────────────────────────────────────────────────
export const api = {
  config: () => request<AppConfig>('/config'),
  anleitung: () => request<{ markdown: string }>('/anleitung'),

  models: () => request<ModelsResponse>('/models'),
  report: (file: string) => request<{ model_file: string; report: string }>(`/models/${encodeURIComponent(file)}/report`),

  predictSingle: (body: { model_file: string; lemma: string; bedeutung: string; top_k?: number }) =>
    postJson<SingleResponse>('/predict/single', body),
  shap: (body: { model_file: string; lemma: string; bedeutung: string; predicted_label: string; filter_stopwords: boolean }) =>
    postJson<ShapResponse>('/predict/shap', body),
  predictBatch: (modelFile: string, file: File) => {
    const form = new FormData()
    form.append('model_file', modelFile)
    form.append('file', file)
    return request<BatchResponse>('/predict/batch', { method: 'POST', body: form })
  },

  sachgruppen: () => request<SachgruppenResponse>('/sachgruppen'),

  trainingInfo: () => request<TrainingInfo>('/training/info'),
  trainingUpload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<TrainingCsvInfo>('/training/upload', { method: 'POST', body: form })
  },
  trainingStart: (cfg: Record<string, unknown>) => postJson<{ status: string }>('/training/start', cfg),
  trainingBatch: (cfg: Record<string, unknown>) => postJson<{ status: string; total: number }>('/training/batch', cfg),
  trainingStatus: () => request<TrainingStatus>('/training/status'),
}
