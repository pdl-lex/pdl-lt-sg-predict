import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, type SachgruppenResponse } from '../api/client'
import { Badge, Callout, type Column, DataTable, GhostButton } from '../design/ui'
import { useRunAction } from '../state/workbench'

interface SgState {
  data: SachgruppenResponse | null
  loading: boolean
  error: string
  reload: () => void
}
const Ctx = createContext<SgState | null>(null)
const useSg = () => {
  const v = useContext(Ctx)
  if (!v) throw new Error('SachgruppenProvider missing')
  return v
}

const COLUMNS: Column[] = [
  { key: 'nummer', label: 'Nummer', width: 90, mono: true },
  { key: 'sachgruppe', label: 'Sachgruppe' },
  { key: 'support', label: 'Samples', width: 96, align: 'right', mono: true },
  { key: 'precision', label: 'Precision', width: 104, align: 'right', mono: true },
  { key: 'recall', label: 'Recall', width: 96, align: 'right', mono: true },
  { key: 'f1', label: 'F1-Score', width: 104, align: 'right', mono: true },
]

export function SachgruppenProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<SachgruppenResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const reload = useCallback(() => {
    setLoading(true); setError('')
    api.sachgruppen().then(setData).catch((e) => setError(String(e))).finally(() => setLoading(false))
  }, [])

  useEffect(() => { reload() }, [reload])
  useRunAction(reload, [reload])

  return <Ctx.Provider value={{ data, loading, error, reload }}>{children}</Ctx.Provider>
}

export function SachgruppenConfig() {
  const { data, loading, reload } = useSg()
  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Callout tone="primary" icon="info">
        <b>Taxonomie nach Hallig-Wartburg.</b> Diese traditionelle Liste entspricht nicht mehr
        modernem Sprachgebrauch und wird nur aus Kompatibilitätsgründen verwendet.
      </Callout>
      {data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {data.model_name ? (
            <>
              <div style={{ fontSize: 11, color: 'var(--lt-fg-3)' }}>Metriken aus Modell:</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                <Badge tone="primary">{data.model_name}</Badge>
                <Badge tone="neutral">Accuracy {data.accuracy}</Badge>
              </div>
              <div style={{ fontSize: 10.5, color: 'var(--lt-fg-4)', fontFamily: 'var(--lt-font-mono)', wordBreak: 'break-all' }}>{data.model_file}</div>
            </>
          ) : (
            <Callout tone="warn" icon="warn">Kein trainiertes Modell gefunden – Metriken nicht verfügbar.</Callout>
          )}
        </div>
      )}
      <GhostButton icon="refresh" onClick={reload} disabled={loading}>{loading ? 'Lädt…' : 'Neu laden'}</GhostButton>
    </div>
  )
}

export function SachgruppenMain() {
  const { data, error } = useSg()
  const rows = data?.rows ?? []
  return (
    <DataTable
      title="Sachgruppen"
      meta={<><span style={{ fontWeight: 600 }}>{rows.length}</span><span> Gruppen</span></>}
      columns={COLUMNS}
      rows={rows}
      csvName="sachgruppen.csv"
      emptyHint={error || 'Keine Sachgruppen geladen.'}
      rowKey={(r) => String(r.nummer)}
    />
  )
}
