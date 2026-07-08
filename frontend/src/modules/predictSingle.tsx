import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import { api, ApiError, type ShapResponse, type ShapWord, type SingleResponse } from '../api/client'
import { Icon } from '../design/icons'
import { Callout, Card, Field, PrimaryButton, TextInput, Toggle, Select } from '../design/ui'
import { kc } from '../design/widgets'
import { useRunAction, useWorkbench } from '../state/workbench'

interface SingleState {
  lemma: string; setLemma: (v: string) => void
  bedeutung: string; setBedeutung: (v: string) => void
  result: SingleResponse | null
  running: boolean
  error: string
  run: () => void
  shap: ShapResponse | null
  shapLoading: boolean
  shapError: string
  filterStopwords: boolean
  computeShap: () => void
  toggleStopwords: () => void
}
const Ctx = createContext<SingleState | null>(null)
const useSingle = () => {
  const v = useContext(Ctx)
  if (!v) throw new Error('SingleProvider missing')
  return v
}

export function SingleProvider({ children }: { children: ReactNode }) {
  const { selectedModel } = useWorkbench()
  const [lemma, setLemma] = useState('')
  const [bedeutung, setBedeutung] = useState('')
  const [result, setResult] = useState<SingleResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  const [shap, setShap] = useState<ShapResponse | null>(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState('')
  const [filterStopwords, setFilterStopwords] = useState(true)

  const computeShapFor = useCallback((pred: string, stops: boolean) => {
    if (!selectedModel) return
    setShapLoading(true); setShapError(''); setShap(null)
    api.shap({ model_file: selectedModel, lemma, bedeutung, predicted_label: pred, filter_stopwords: stops })
      .then(setShap)
      .catch((e) => setShapError(e instanceof ApiError ? `Fehler (${e.status}): ${e.message}` : String(e)))
      .finally(() => setShapLoading(false))
  }, [selectedModel, lemma, bedeutung])

  const run = useCallback(() => {
    if (!selectedModel) { setError('Kein Modell gewählt.'); return }
    if (!bedeutung.trim()) { setError('Bitte eine Bedeutung eingeben.'); return }
    setRunning(true); setError(''); setShap(null); setShapError('')
    api.predictSingle({ model_file: selectedModel, lemma, bedeutung })
      .then((res) => {
        setResult(res)
        if (res.model_type !== 'nn') computeShapFor(res.prediction, filterStopwords)
      })
      .catch((e) => { setError(e instanceof ApiError ? `Fehler (${e.status}): ${e.message}` : String(e)); setResult(null) })
      .finally(() => setRunning(false))
  }, [selectedModel, lemma, bedeutung, filterStopwords, computeShapFor])

  useRunAction(run, [run])

  const computeShap = useCallback(() => {
    if (result) computeShapFor(result.prediction, filterStopwords)
  }, [result, filterStopwords, computeShapFor])

  const toggleStopwords = useCallback(() => {
    const next = !filterStopwords
    setFilterStopwords(next)
    if (result && result.model_type !== 'nn') computeShapFor(result.prediction, next)
  }, [filterStopwords, result, computeShapFor])

  return (
    <Ctx.Provider value={{
      lemma, setLemma, bedeutung, setBedeutung, result, running, error, run,
      shap, shapLoading, shapError, filterStopwords, computeShap, toggleStopwords,
    }}>{children}</Ctx.Provider>
  )
}

export function SingleConfig() {
  const { models, selectedModel, setSelectedModel } = useWorkbench()
  const { lemma, setLemma, bedeutung, setBedeutung, error, run } = useSingle()
  const files = models?.files ?? []

  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Field label="Modell">
        <Select value={selectedModel} onChange={setSelectedModel} options={files.map((f) => ({ value: f, label: f }))} />
      </Field>
      <Field label="Lemma (optional)">
        <TextInput value={lemma} onChange={setLemma} placeholder="z. B. Waggala" onEnter={run} />
      </Field>
      <Field label="Bedeutung">
        <TextInput value={bedeutung} onChange={setBedeutung} placeholder="z. B. kleines Kind; wackelig auf den Beinen" onEnter={run} />
      </Field>
      {error && <Callout tone="err" icon="warn">{error}</Callout>}
    </div>
  )
}

export function SingleFooter() {
  const { run, running, bedeutung } = useSingle()
  return <PrimaryButton icon="play" onClick={run} disabled={running || !bedeutung.trim()} kbd={kc('↵')} full>{running ? 'Läuft…' : 'Vorhersagen'}</PrimaryButton>
}

// ── Ergebnis-Panel ───────────────────────────────────────────────────────────
export function SingleMain() {
  const { result, running, error } = useSingle()
  return (
    <main style={{ gridArea: 'main', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ height: 44, flexShrink: 0, padding: '0 16px', display: 'flex', alignItems: 'center', gap: 12, borderBottom: '1px solid var(--lt-line-1)', background: 'var(--lt-bg-0)' }}>
        <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Vorhersage</h3>
        {result && <span style={{ fontFamily: 'var(--lt-font-mono)', fontSize: 12, color: 'var(--lt-fg-3)' }}>Modell: {result.model_type}</span>}
      </div>
      <div className="agm-grid" style={{ flex: 1, overflow: 'auto', background: 'var(--lt-bg-1)', padding: 20 }}>
        {error ? <Callout tone="err" icon="warn">{error}</Callout>
          : running ? <div style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>Vorhersage läuft…</div>
          : !result ? <div style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>Modell wählen, Bedeutung eingeben und „Vorhersagen".</div>
          : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18, maxWidth: 900 }}>
              <TopPredictions top={result.top} />
              <ShapPanel />
            </div>
          )}
      </div>
    </main>
  )
}

function TopPredictions({ top }: { top: SingleResponse['top'] }) {
  return (
    <div style={{ display: 'flex', gap: 12, alignItems: 'stretch', flexWrap: 'wrap' }}>
      {top.map((t, i) => (
        <div key={i} style={{
          flex: '1 1 200px', minWidth: 200, padding: 14, borderRadius: 'var(--lt-r-md)',
          background: t.is_best ? 'var(--lt-primary-soft)' : 'var(--lt-bg-0)',
          border: '1px solid ' + (t.is_best ? 'var(--lt-primary-line)' : 'var(--lt-line-1)'),
          boxShadow: 'var(--lt-shadow-1)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
            {t.is_best && <Icon name="sparkle" size={14} style={{ color: 'var(--lt-primary)' }} />}
            <span style={{ fontFamily: 'var(--lt-font-mono)', fontSize: 18, fontWeight: 600, color: t.is_best ? 'var(--lt-primary)' : 'var(--lt-fg-1)' }}>{t.label}</span>
            {t.proba != null && <span style={{ marginLeft: 'auto', fontFamily: 'var(--lt-font-mono)', fontSize: 13, color: 'var(--lt-fg-3)' }}>{t.proba}%</span>}
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--lt-fg-2)', lineHeight: 1.4 }}>{t.description}</div>
        </div>
      ))}
    </div>
  )
}

function shapStyle(score: number) {
  if (score > 0.1) return { color: 'var(--lt-primary)', bg: 'var(--lt-primary-soft)', line: 'var(--lt-primary-line)' }
  if (score < -0.1) return { color: 'var(--lt-err)', bg: 'var(--lt-err-soft)', line: 'var(--lt-err-line)' }
  return { color: 'var(--lt-fg-3)', bg: 'var(--lt-bg-2)', line: 'var(--lt-line-1)' }
}

function WordBadge({ w }: { w: ShapWord }) {
  const s = shapStyle(w.score)
  return (
    <span title={w.score.toFixed(3)} style={{
      fontSize: 12, padding: '2px 8px', borderRadius: 999, color: s.color, background: s.bg, border: `1px solid ${s.line}`,
    }}>{w.word}</span>
  )
}

function ShapPanel() {
  const { shap, shapLoading, shapError, filterStopwords, computeShap, toggleStopwords, result } = useSingle()
  const isNN = result?.model_type === 'nn'
  const hasResults = Boolean(shap && (shap.lemma.length || shap.bedeutung.length))

  const topWords: ShapWord[] = shap
    ? [...shap.lemma, ...shap.bedeutung].sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 10)
    : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {shapLoading && <Callout tone="neutral" icon="sparkle">SHAP-Erklärung wird berechnet…</Callout>}
      {isNN && !shapLoading && !hasResults && (
        <PrimaryButton icon="sparkle" onClick={computeShap} style={{ alignSelf: 'flex-start', background: 'var(--lt-warn)', borderColor: 'var(--lt-warn)' }}>
          Erklärung anzeigen (Neural Network – dauert ~30–60 s)
        </PrimaryButton>
      )}
      {shapError && <Callout tone="err" icon="warn">SHAP-Fehler: {shapError}</Callout>}

      {hasResults && shap && (
        <Card style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <Icon name="sparkle" size={16} style={{ color: 'var(--lt-primary)' }} />
            <h4 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>Vorhersage-Erklärung (SHAP)</h4>
            <span style={{ flex: 1 }} />
            <Toggle checked={filterStopwords} onChange={toggleStopwords} label="Stoppwörter ausblenden" />
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--lt-fg-3)', marginBottom: 12 }}>
            Grün = stützt die Vorhersage · Rot = widerspricht · Grau = neutral
          </div>

          {shap.lemma.length > 0 && (
            <WordRow label="Lemma" words={shap.lemma} />
          )}
          {shap.bedeutung.length > 0 && (
            <WordRow label="Bedeutung" words={shap.bedeutung} />
          )}

          {topWords.length > 0 && (
            <>
              <div style={{ fontSize: 12, fontWeight: 600, margin: '14px 0 8px' }}>Top-Wörter nach Einfluss</div>
              <InfluenceBars words={topWords} />
            </>
          )}
        </Card>
      )}
    </div>
  )
}

function WordRow({ label, words }: { label: string; words: ShapWord[] }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 5 }}>{label}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {words.map((w, i) => <WordBadge key={i} w={w} />)}
      </div>
    </div>
  )
}

const BAR_AREA = 58
function InfluenceBars({ words }: { words: ShapWord[] }) {
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', gap: 6, width: '100%' }}>
      {words.map((w, i) => {
        const s = shapStyle(w.score)
        const h = Math.max(2, Math.min(1, Math.abs(w.score)) * BAR_AREA)
        const pos = w.score >= 0
        return (
          <div key={i} style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ height: BAR_AREA, width: '100%', display: 'flex', alignItems: 'flex-end' }}>
              {pos && <div title={w.score.toFixed(3)} style={{ height: h, width: '70%', margin: '0 auto', background: s.color, borderRadius: '3px 3px 0 0' }} />}
            </div>
            <div style={{ height: 1, width: '100%', background: 'var(--lt-line-2)' }} />
            <div style={{ height: BAR_AREA, width: '100%', display: 'flex', alignItems: 'flex-start' }}>
              {!pos && <div title={w.score.toFixed(3)} style={{ height: h, width: '70%', margin: '0 auto', background: s.color, borderRadius: '0 0 3px 3px' }} />}
            </div>
            <div style={{ fontSize: 10.5, color: 'var(--lt-fg-3)', textAlign: 'center', width: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 4 }}>{w.word}</div>
          </div>
        )
      })}
    </div>
  )
}
