import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react'
import { api, ApiError, type BatchResponse } from '../api/client'
import { Icon } from '../design/icons'
import { Callout, type Column, DataTable, Field, GhostButton, PrimaryButton, Select } from '../design/ui'
import { kc } from '../design/widgets'
import { useRunAction, useWorkbench } from '../state/workbench'

interface BatchState {
  file: File | null
  setFile: (f: File | null) => void
  result: BatchResponse | null
  running: boolean
  error: string
  run: () => void
}
const Ctx = createContext<BatchState | null>(null)
const useBatch = () => {
  const v = useContext(Ctx)
  if (!v) throw new Error('BatchProvider missing')
  return v
}

function columnsFor(usesLemma: boolean): Column[] {
  const cols: Column[] = []
  if (usesLemma) cols.push({ key: 'lemma', label: 'Lemma', width: 130 })
  cols.push({ key: 'bedeutung', label: 'Bedeutung' })
  cols.push({ key: 'sachgruppe', label: 'SG 1', width: 78, mono: true, primary: true })
  cols.push({ key: 'beschreibung', label: 'Beschreibung 1', width: 180 })
  cols.push({ key: 'wahrscheinlichkeit', label: 'W. 1', width: 66, align: 'right', mono: true })
  cols.push({ key: 'sachgruppe_2', label: 'SG 2', width: 70, mono: true })
  cols.push({ key: 'beschreibung_2', label: 'Beschreibung 2', width: 160 })
  cols.push({ key: 'wahrscheinlichkeit_2', label: 'W. 2', width: 66, align: 'right', mono: true })
  cols.push({ key: 'sachgruppe_3', label: 'SG 3', width: 70, mono: true })
  cols.push({ key: 'beschreibung_3', label: 'Beschreibung 3', width: 160 })
  cols.push({ key: 'wahrscheinlichkeit_3', label: 'W. 3', width: 66, align: 'right', mono: true })
  return cols
}

export function BatchProvider({ children }: { children: ReactNode }) {
  const { selectedModel } = useWorkbench()
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<BatchResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  const run = useCallback(() => {
    if (!selectedModel) { setError('Kein Modell gewählt.'); return }
    if (!file) { setError('Bitte zuerst eine CSV-Datei wählen.'); return }
    setRunning(true); setError('')
    api.predictBatch(selectedModel, file)
      .then(setResult)
      .catch((e) => { setError(e instanceof ApiError ? `Fehler (${e.status}): ${e.message}` : String(e)); setResult(null) })
      .finally(() => setRunning(false))
  }, [selectedModel, file])

  useRunAction(run, [run])

  return <Ctx.Provider value={{ file, setFile, result, running, error, run }}>{children}</Ctx.Provider>
}

export function BatchConfig() {
  const { models, selectedModel, setSelectedModel } = useWorkbench()
  const { file, setFile, error } = useBatch()
  const fileRef = useRef<HTMLInputElement>(null)
  const files = models?.files ?? []

  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Field label="Modell">
        <Select value={selectedModel} onChange={setSelectedModel} options={files.map((f) => ({ value: f, label: f }))} />
      </Field>
      <Field label="CSV-Datei" hint="Spalte 'bedeutung' erforderlich, 'lemma' optional.">
        <input ref={fileRef} type="file" accept=".csv,text/csv" style={{ display: 'none' }}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <GhostButton icon="upload" onClick={() => fileRef.current?.click()}>CSV wählen…</GhostButton>
      </Field>
      {file && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11.5, color: 'var(--lt-fg-3)' }}>
          <Icon name="file" size={12} style={{ color: 'var(--lt-primary)' }} />
          <span style={{ fontFamily: 'var(--lt-font-mono)' }}>{file.name}</span>
        </div>
      )}
      {error && <Callout tone="err" icon="warn">{error}</Callout>}
    </div>
  )
}

export function BatchFooter() {
  const { run, running, file } = useBatch()
  return <PrimaryButton icon="play" onClick={run} disabled={running || !file} kbd={kc('↵')} full>{running ? 'Läuft…' : 'Vorhersagen'}</PrimaryButton>
}

export function BatchMain() {
  const { result, running, error } = useBatch()
  const rows = result?.rows ?? []
  const columns = columnsFor(result?.uses_lemma ?? true)
  const hint = error ? error : running ? 'Vorhersage läuft…' : 'Modell + CSV wählen und „Vorhersagen".'
  return (
    <DataTable
      title="Batch-Vorhersage"
      meta={result ? <><span style={{ fontWeight: 600 }}>{result.count}</span><span> Vorhersagen</span></> : undefined}
      columns={columns}
      rows={rows}
      csvName="vorhersage_ergebnisse.csv"
      emptyHint={hint}
    />
  )
}
