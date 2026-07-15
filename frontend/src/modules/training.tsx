import {
  createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode,
} from 'react'
import { api, ApiError, type CrossValidation, type TrainingCsvInfo, type TrainingStatus } from '../api/client'
import { Icon } from '../design/icons'
import {
  Badge, Callout, Checkbox, Field, GhostButton, MonoBadge, PrimaryButton,
  ResultsFrame, SectionFold, Select, Toggle,
} from '../design/ui'
import { Segmented } from '../design/widgets'
import { useWorkbench } from '../state/workbench'

// ── Optionslisten (aus der alten App) ────────────────────────────────────────
const SVM_C = ['0.01', '0.1', '0.5', '1.0', '5.0', '10.0', '50.0', '100.0']
const XGB_N = ['100', '200', '300', '500', '800']
const XGB_DEPTH = ['3', '4', '5', '6', '7', '8', '10']
const XGB_LR = ['0.01', '0.03', '0.05', '0.1', '0.15', '0.2']
const XGB_SUB = ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
const NN_LAYERS = ['100', '200,100', '200,100,50', '300,150,75', '400,200,100,50']
const NN_ALPHA = ['0.00001', '0.0001', '0.001', '0.01', '0.1']
const NN_LR = ['0.0001', '0.0005', '0.001', '0.005', '0.01']
const BATCH_ANALYZERS = ['char_wb', 'word-(1,1)', 'word-(1,2)']
const BATCH_MINLENS = [1, 2, 3]
const TIME_FALLBACK: Record<string, number> = { svm: 120, logistic: 4286, rf: 30, nn: 111, xgboost: 6112 }
const TIME_FALLBACK_SAMPLES = 113127

// Entspricht der Dedupe-Logik in core/training.py:start_batch — word-(1,1) und
// word-(1,2) verschmelzen dort zu EINER word-Konfiguration (max. n-gram gewinnt).
const analyzerVariants = (analyzers: string[]) =>
  new Set(analyzers.map((a) => (a.startsWith('word') ? 'word' : 'char_wb'))).size

interface Cfg {
  model: string; test_size: number; use_stopword_removal: boolean; min_word_length: number
  analyzer_mode: string; word_ngram_max: number; use_spacy: boolean; use_dornseiff: boolean
  cross_validate: boolean; cv_folds: number; cv_mode: string
  tune_mode: string; tune_n_iter: number; tune_cv: number
  svm_c: string; xgb_n_estimators: string; xgb_max_depth: string; xgb_learning_rate: string; xgb_subsample: string
  nn_hidden_layers: string; nn_alpha: string; nn_learning_rate_init: string
  batch_model_types: string[]; batch_use_stopwords: boolean[]; batch_min_lengths: number[]; batch_analyzers: string[]
}

const DEFAULT_CFG: Cfg = {
  model: 'svm', test_size: 0.2, use_stopword_removal: false, min_word_length: 1,
  analyzer_mode: 'char_wb', word_ngram_max: 1, use_spacy: true, use_dornseiff: true,
  cross_validate: false, cv_folds: 5, cv_mode: 'stratified',
  tune_mode: 'standard', tune_n_iter: 20, tune_cv: 3,
  svm_c: '1.0', xgb_n_estimators: '300', xgb_max_depth: '6', xgb_learning_rate: '0.05', xgb_subsample: '0.8',
  nn_hidden_layers: '100', nn_alpha: '0.0001', nn_learning_rate_init: '0.0005',
  batch_model_types: ['svm'], batch_use_stopwords: [false], batch_min_lengths: [1], batch_analyzers: ['char_wb'],
}

interface TrainState {
  cfg: Cfg
  set: <K extends keyof Cfg>(k: K, v: Cfg[K]) => void
  csv: TrainingCsvInfo | null
  uploading: boolean
  uploadError: string
  upload: (f: File) => void
  status: TrainingStatus | null
  running: boolean
  startError: string
  startSingle: () => void
  startBatch: () => void
}
const Ctx = createContext<TrainState | null>(null)
const useTrain = () => {
  const v = useContext(Ctx)
  if (!v) throw new Error('TrainingProvider missing')
  return v
}

export function TrainingProvider({ children }: { children: ReactNode }) {
  const { reloadModels } = useWorkbench()
  const [cfg, setCfg] = useState<Cfg>(DEFAULT_CFG)
  const [csv, setCsv] = useState<TrainingCsvInfo | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [running, setRunning] = useState(false)
  const [startError, setStartError] = useState('')
  const pollRef = useRef<number | null>(null)

  const set = useCallback(<K extends keyof Cfg>(k: K, v: Cfg[K]) => setCfg((c) => ({ ...c, [k]: v })), [])

  const stopPolling = useCallback(() => {
    if (pollRef.current != null) { window.clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const startPolling = useCallback(() => {
    stopPolling()
    pollRef.current = window.setInterval(() => {
      api.trainingStatus().then((st) => {
        setStatus(st)
        if (st.state === 'done' || st.state === 'error' || st.state === 'idle') {
          stopPolling(); setRunning(false)
          if (st.state === 'done') void reloadModels()
        }
      }).catch(() => { /* weiter pollen */ })
    }, 700)
  }, [stopPolling, reloadModels])

  useEffect(() => {
    api.trainingInfo().then((info) => {
      if (info.csv) setCsv(info.csv)
      if (info.running) { setRunning(true); startPolling() }
    }).catch(() => { /* ignorieren */ })
    return () => stopPolling()
  }, [startPolling, stopPolling])

  const upload = useCallback((f: File) => {
    setUploading(true); setUploadError('')
    api.trainingUpload(f)
      .then(setCsv)
      .catch((e) => setUploadError(e instanceof ApiError ? `Fehler (${e.status}): ${e.message}` : String(e)))
      .finally(() => setUploading(false))
  }, [])

  const begin = useCallback((fn: () => Promise<unknown>) => {
    setStartError(''); setStatus(null); setRunning(true)
    fn().then(() => startPolling())
      .catch((e) => { setRunning(false); setStartError(e instanceof ApiError ? `Fehler (${e.status}): ${e.message}` : String(e)) })
  }, [startPolling])

  const startSingle = useCallback(() => begin(() => api.trainingStart(cfg as unknown as Record<string, unknown>)), [begin, cfg])
  const startBatch = useCallback(() => begin(() => api.trainingBatch(cfg as unknown as Record<string, unknown>)), [begin, cfg])

  return (
    <Ctx.Provider value={{ cfg, set, csv, uploading, uploadError, upload, status, running, startError, startSingle, startBatch }}>
      {children}
    </Ctx.Provider>
  )
}

// ── Config-Pane ──────────────────────────────────────────────────────────────
export function TrainingConfig() {
  const { enableTraining, modelTypes } = useWorkbench()
  const { cfg, set, csv, uploading, uploadError, upload } = useTrain()
  const fileRef = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState<Record<string, boolean>>({ daten: true, einzel: true, tuning: false, batch: false })
  const toggle = (k: string) => setOpen((o) => ({ ...o, [k]: !o[k] }))

  const toggleIn = <T,>(arr: T[], v: T, setter: (x: T[]) => void) => {
    if (arr.includes(v)) { if (arr.length > 1) setter(arr.filter((x) => x !== v)) }
    else setter([...arr, v])
  }

  const batchCount = cfg.batch_model_types.length * cfg.batch_use_stopwords.length
    * cfg.batch_min_lengths.length * analyzerVariants(cfg.batch_analyzers)

  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
      {!enableTraining && <Callout tone="warn" icon="warn">Training ist deaktiviert (<code>ENABLE_TRAINING=False</code>).</Callout>}

      {/* Daten */}
      <SectionFold title="Daten" icon="file" open={open.daten} onToggle={() => toggle('daten')}
        badge={csv ? `${csv.num_samples}` : undefined}>
        <div style={{ fontSize: 11.5, color: 'var(--lt-fg-3)' }}>CSV mit Spalten: <code>lemma</code>, <code>bedeutung</code>, <code>sachgruppe</code>.</div>
        <input ref={fileRef} type="file" accept=".csv,text/csv" style={{ display: 'none' }}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f) }} />
        <GhostButton icon="upload" onClick={() => fileRef.current?.click()} disabled={!enableTraining || uploading}>
          {uploading ? 'Lädt…' : 'CSV hochladen'}
        </GhostButton>
        {csv && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <Icon name="check" size={13} style={{ color: 'var(--lt-primary)' }} />
            <span style={{ fontSize: 11.5, fontFamily: 'var(--lt-font-mono)' }}>{csv.filename}</span>
            <Badge tone="neutral">{csv.num_samples} Samples · {csv.num_classes} Klassen</Badge>
          </div>
        )}
        {uploadError && <Callout tone="err" icon="warn">{uploadError}</Callout>}
      </SectionFold>

      {/* Einzeltraining */}
      <SectionFold title="Einzeltraining" icon="brain" open={open.einzel} onToggle={() => toggle('einzel')}>
        <Field label="Modell-Typ">
          <Select value={cfg.model} onChange={(v) => set('model', v)} options={modelTypes.map((m) => ({ value: m.code, label: m.name }))} />
        </Field>
        <Field label={`Test-Anteil: ${(cfg.test_size * 100).toFixed(0)} %`}>
          <input type="range" min={0.1} max={0.4} step={0.05} value={cfg.test_size}
            onChange={(e) => set('test_size', Number(e.target.value))} style={{ width: '100%', accentColor: 'var(--lt-primary)' }} />
        </Field>
        <Toggle checked={cfg.use_stopword_removal} onChange={(v) => set('use_stopword_removal', v)} label="Stoppwörter entfernen" />
        <Field label={`Min. Wortlänge: ${cfg.min_word_length}`}>
          <input type="range" min={1} max={5} step={1} value={cfg.min_word_length}
            onChange={(e) => set('min_word_length', Number(e.target.value))} style={{ width: '100%', accentColor: 'var(--lt-primary)' }} />
        </Field>
        <Field label="Analyzer">
          <Segmented options={['char_wb', 'word']} value={cfg.analyzer_mode} onChange={(v) => set('analyzer_mode', v)} />
        </Field>
        {cfg.analyzer_mode === 'word' && (
          <Field label="N-Gramm (max)">
            <Segmented options={['1', '2']} value={String(cfg.word_ngram_max)} onChange={(v) => set('word_ngram_max', Number(v))} />
          </Field>
        )}
        <div style={{ borderTop: '1px solid var(--lt-line-1)', paddingTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--lt-fg-2)' }}>Semantische Anreicherung</div>
          <Toggle checked={cfg.use_spacy} onChange={(v) => set('use_spacy', v)} label="spaCy-Wortvektoren (de_core_news_lg)" />
          <Toggle checked={cfg.use_dornseiff} onChange={(v) => set('use_dornseiff', v)} label="Dornseiff-Thesaurus" />
        </div>
        <div style={{ borderTop: '1px solid var(--lt-line-1)', paddingTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--lt-fg-2)' }}>Cross-Validierung</div>
          <Toggle checked={cfg.cross_validate} onChange={(v) => set('cross_validate', v)}
            label="Zusätzliche k-fold-Bewertung (split-unabhängig)" />
          {cfg.cross_validate && (
            <>
              <Field label="Folds (k)"><input type="number" min={2} max={20} value={cfg.cv_folds}
                onChange={(e) => set('cv_folds', Math.max(2, Number(e.target.value)))} style={numInput} /></Field>
              <Field label="Fold-Strategie">
                <Segmented options={['stratified', 'group']} value={cfg.cv_mode} onChange={(v) => set('cv_mode', v)} />
              </Field>
              <Callout tone="info" icon="info">
                {cfg.cv_mode === 'group'
                  ? 'GroupKFold nach bedeutung: identische Glossen bleiben im selben Fold.'
                  : 'StratifiedKFold: erhält die Klassenverteilung je Fold (fixer Seed).'}
              </Callout>
              <Callout tone="warn" icon="warn">Dauer ≈ (1 + k) × Einzeltraining. Nur Einzeltraining, nicht im Batch.</Callout>
            </>
          )}
        </div>
      </SectionFold>

      {/* Hyperparameter-Tuning */}
      <SectionFold title="Hyperparameter-Tuning" icon="settings" open={open.tuning} onToggle={() => toggle('tuning')}>
        <Segmented options={['standard', 'auto', 'manual']} value={cfg.tune_mode} onChange={(v) => set('tune_mode', v)} />
        {cfg.tune_mode === 'auto' && (
          <>
            <Field label="Kombinationen (n_iter)"><input type="number" min={1} value={cfg.tune_n_iter}
              onChange={(e) => set('tune_n_iter', Math.max(1, Number(e.target.value)))} style={numInput} /></Field>
            <Field label="CV-Folds"><input type="number" min={2} value={cfg.tune_cv}
              onChange={(e) => set('tune_cv', Math.max(2, Number(e.target.value)))} style={numInput} /></Field>
            <Callout tone="warn" icon="warn">Dauer ≈ n_iter × cv × Einzeltraining. Beste Parameter werden übernommen.</Callout>
          </>
        )}
        {cfg.tune_mode === 'manual' && (
          <>
            <ParamGroup title="Linear SVM">
              <Field label="C (Regularisierung)"><Select value={cfg.svm_c} onChange={(v) => set('svm_c', v)} options={opts(SVM_C)} /></Field>
            </ParamGroup>
            <ParamGroup title="XGBoost">
              <Field label="n_estimators"><Select value={cfg.xgb_n_estimators} onChange={(v) => set('xgb_n_estimators', v)} options={opts(XGB_N)} /></Field>
              <Field label="max_depth"><Select value={cfg.xgb_max_depth} onChange={(v) => set('xgb_max_depth', v)} options={opts(XGB_DEPTH)} /></Field>
              <Field label="learning_rate"><Select value={cfg.xgb_learning_rate} onChange={(v) => set('xgb_learning_rate', v)} options={opts(XGB_LR)} /></Field>
              <Field label="subsample"><Select value={cfg.xgb_subsample} onChange={(v) => set('xgb_subsample', v)} options={opts(XGB_SUB)} /></Field>
            </ParamGroup>
            <ParamGroup title="Neural Network (MLP)">
              <Field label="hidden_layers"><Select value={cfg.nn_hidden_layers} onChange={(v) => set('nn_hidden_layers', v)} options={opts(NN_LAYERS)} /></Field>
              <Field label="alpha (L2)"><Select value={cfg.nn_alpha} onChange={(v) => set('nn_alpha', v)} options={opts(NN_ALPHA)} /></Field>
              <Field label="learning_rate_init"><Select value={cfg.nn_learning_rate_init} onChange={(v) => set('nn_learning_rate_init', v)} options={opts(NN_LR)} /></Field>
            </ParamGroup>
          </>
        )}
      </SectionFold>

      {/* Batch */}
      <SectionFold title="Batch-Training" icon="layers" open={open.batch} onToggle={() => toggle('batch')} badge={`${batchCount}`}>
        <div style={{ fontSize: 11, fontWeight: 600 }}>Modell-Typen</div>
        {modelTypes.map((m) => (
          <Checkbox key={m.code} checked={cfg.batch_model_types.includes(m.code)}
            onChange={() => toggleIn(cfg.batch_model_types, m.code, (x) => set('batch_model_types', x))} label={m.name} />
        ))}
        <div style={{ fontSize: 11, fontWeight: 600, marginTop: 4 }}>Stoppwörter</div>
        <Checkbox checked={cfg.batch_use_stopwords.includes(false)} onChange={() => toggleIn(cfg.batch_use_stopwords, false, (x) => set('batch_use_stopwords', x))} label="nicht entfernen" />
        <Checkbox checked={cfg.batch_use_stopwords.includes(true)} onChange={() => toggleIn(cfg.batch_use_stopwords, true, (x) => set('batch_use_stopwords', x))} label="entfernen" />
        <div style={{ fontSize: 11, fontWeight: 600, marginTop: 4 }}>Min. Wortlänge</div>
        {BATCH_MINLENS.map((n) => (
          <Checkbox key={n} checked={cfg.batch_min_lengths.includes(n)}
            onChange={() => toggleIn(cfg.batch_min_lengths, n, (x) => set('batch_min_lengths', x.sort()))} label={`≥ ${n}`} />
        ))}
        <div style={{ fontSize: 11, fontWeight: 600, marginTop: 4 }}>Analyzer</div>
        {BATCH_ANALYZERS.map((a) => (
          <Checkbox key={a} checked={cfg.batch_analyzers.includes(a)}
            onChange={() => toggleIn(cfg.batch_analyzers, a, (x) => set('batch_analyzers', x))} label={a} />
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
          <MonoBadge>{batchCount} Modelle</MonoBadge>
          <span style={{ fontSize: 10.5, color: 'var(--lt-fg-4)' }}>{estimateBatch(cfg, csv)}</span>
        </div>
      </SectionFold>
    </div>
  )
}

const numInput = { width: 110, padding: '6px 9px', fontSize: 12, background: 'var(--lt-bg-1)', border: '1px solid var(--lt-line-1)', borderRadius: 'var(--lt-r-sm)', color: 'var(--lt-fg-1)', outline: 'none' } as const
const opts = (a: string[]) => a.map((v) => ({ value: v, label: v }))

function ParamGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ border: '1px solid var(--lt-line-1)', borderRadius: 'var(--lt-r-sm)', padding: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 11.5, fontWeight: 600 }}>{title}</div>
      {children}
    </div>
  )
}

function estimateBatch(cfg: Cfg, csv: TrainingCsvInfo | null): string {
  const n = csv?.num_samples ?? 0
  if (!n) return ''
  const perType = cfg.batch_use_stopwords.length * cfg.batch_min_lengths.length * analyzerVariants(cfg.batch_analyzers)
  let secs = 0
  for (const mt of cfg.batch_model_types) {
    const measured = csv?.time_per_type?.[mt]
    const est = measured && measured > 0 ? measured : (TIME_FALLBACK[mt] ?? 120) / TIME_FALLBACK_SAMPLES * n
    secs += est * perType
  }
  if (secs <= 0) return ''
  const t = Math.round(secs), h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60
  const src = csv?.time_per_type && Object.keys(csv.time_per_type).length ? 'gemessen' : 'geschätzt'
  if (h > 0) return `ca. ${h} h ${m} min (${src})`
  if (m > 0) return `ca. ${m} min ${s} s (${src})`
  return `ca. ${s} s (${src})`
}

// ── Config-Footer (Primäraktion) ─────────────────────────────────────────────
export function TrainingFooter() {
  const { enableTraining } = useWorkbench()
  const { cfg, csv, running, startSingle, startBatch } = useTrain()
  const disabled = !enableTraining || !csv || running
  const batchDisabled = disabled || cfg.tune_mode === 'auto'
  return (
    <div style={{ display: 'flex', gap: 8, width: '100%' }}>
      <PrimaryButton icon="play" onClick={startSingle} disabled={disabled} style={{ flex: 1 }}>{running ? 'Läuft…' : 'Trainieren'}</PrimaryButton>
      <GhostButton icon="layers" onClick={startBatch} disabled={batchDisabled} title={cfg.tune_mode === 'auto' ? 'Bei Auto-Tune deaktiviert' : 'Batch-Training'}>Batch</GhostButton>
    </div>
  )
}

// ── Ergebnis-Panel ───────────────────────────────────────────────────────────
export function TrainingMain() {
  const { status, running, startError, csv } = useTrain()

  return (
    <ResultsFrame title="Training" meta={csv ? <span>{csv.filename} · {csv.num_samples} Samples</span> : undefined}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 720 }}>
        {startError && <Callout tone="err" icon="warn">{startError}</Callout>}

        {running && status?.state === 'running' && status.mode === 'single' && (
          <ProgressBlock label={status.msg ?? 'Training…'} pct={status.pct ?? 0} />
        )}
        {running && status?.state === 'running' && status.mode === 'batch' && (
          <ProgressBlock label={`Trainiere ${(status.done ?? 0) + 1}/${status.total ?? 0}: ${status.msg ?? ''}`}
            pct={status.total ? Math.round(((status.done ?? 0) / status.total) * 100) : 0} />
        )}
        {running && !status && <div style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>Training wird gestartet…</div>}

        {status?.state === 'error' && <Callout tone="err" icon="warn"><b>Training-Fehler:</b> {status.error}</Callout>}

        {status?.state === 'done' && status.mode === 'single' && (
          <Callout tone="primary" icon="check">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--lt-primary)' }}>Accuracy: {(status.accuracy ?? 0).toFixed(4)}</div>
              <div>Trainingszeit: {(status.training_time ?? 0).toFixed(1)} s</div>
              <div>Gespeichert: <span style={{ fontFamily: 'var(--lt-font-mono)' }}>{status.model_file}</span></div>
              {status.best_params && Object.keys(status.best_params).length > 0 && (
                <div style={{ marginTop: 4 }}>
                  <div style={{ fontWeight: 600 }}>Beste Parameter (CV {(status.best_cv_score ?? 0).toFixed(4)}):</div>
                  <div style={{ fontFamily: 'var(--lt-font-mono)', fontSize: 11.5 }}>
                    {Object.entries(status.best_params).map(([k, v]) => `${k} = ${String(v)}`).join('  ·  ')}
                  </div>
                </div>
              )}
              {status.cross_validation && <CrossValBlock cv={status.cross_validation} splitAccuracy={status.accuracy ?? 0} />}
            </div>
          </Callout>
        )}
        {status?.state === 'done' && status.mode === 'batch' && (
          <Callout tone="primary" icon="check">Batch-Training abgeschlossen ({status.total} Modelle). Ergebnisse unter „Analyse".</Callout>
        )}

        {!running && !status && (
          <div style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>
            {csv ? 'Bereit. Konfiguration links wählen und „Trainieren".' : 'Zuerst eine Trainings-CSV hochladen (links).'}
          </div>
        )}
      </div>
    </ResultsFrame>
  )
}

function CrossValBlock({ cv, splitAccuracy }: { cv: CrossValidation; splitAccuracy: number }) {
  const modeLabel = cv.mode === 'group' ? 'Group / bedeutung' : 'Stratified'
  if (!cv.ok) {
    const why = cv.reason === 'fold_error'
      ? `fehlgeschlagen: ${cv.error ?? ''}`
      : cv.reason === 'too_few_groups' || cv.reason === 'no_groups'
        ? 'zu wenige Gruppen/Klassen'
        : 'zu wenige nutzbare Klassen/Samples'
    return (
      <div style={{ marginTop: 6, fontSize: 12, color: 'var(--lt-fg-3)' }}>
        Cross-Validierung ({modeLabel}, {cv.cv} Folds) {why}.
      </div>
    )
  }
  const mean = cv.mean ?? 0
  const std = cv.std ?? 0
  const delta = splitAccuracy - mean
  return (
    <div style={{ marginTop: 6, borderTop: '1px dashed var(--lt-line-1)', paddingTop: 6 }}>
      <div style={{ fontWeight: 600 }}>
        Cross-Validierung ({cv.cv}-fold {modeLabel}, {cv.scoring}): {mean.toFixed(4)} <span style={{ color: 'var(--lt-fg-3)' }}>± {std.toFixed(4)}</span>
      </div>
      <div style={{ fontFamily: 'var(--lt-font-mono)', fontSize: 11.5, color: 'var(--lt-fg-3)' }}>
        Folds: {(cv.scores ?? []).map((s) => s.toFixed(4)).join('  ·  ')}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--lt-fg-3)', marginTop: 2 }}>
        Einzel-Split vs. CV-Mittel: Δ {delta >= 0 ? '+' : ''}{delta.toFixed(4)}
        {cv.mode === 'group' && cv.n_groups != null && ` · ${cv.n_groups} Bedeutungs-Gruppen`}
        {cv.n_excluded_samples > 0 && ` · ${cv.n_excluded_samples} Samples / ${cv.n_excluded_classes} seltene Klassen ausgeschlossen`}
      </div>
    </div>
  )
}

function ProgressBlock({ label, pct }: { label: string; pct: number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, color: 'var(--lt-fg-2)' }}>
        <span>{label}</span><span style={{ fontFamily: 'var(--lt-font-mono)' }}>{pct} %</span>
      </div>
      <div style={{ height: 8, background: 'var(--lt-bg-2)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: 'var(--lt-primary)', transition: 'width .3s' }} />
      </div>
    </div>
  )
}
