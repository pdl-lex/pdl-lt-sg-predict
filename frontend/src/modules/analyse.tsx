import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import { api, type ModelRow } from '../api/client'
import { Icon } from '../design/icons'
import { Badge, type Column, DataTable, GhostButton, PrimaryButton } from '../design/ui'
import { useRunAction, useWorkbench } from '../state/workbench'

interface AnalyseState {
  reportOpen: boolean
  reportFile: string
  reportText: string
  reportLoading: boolean
  openReport: (f: string) => void
  closeReport: () => void
  downloading: boolean
  downloadReports: (rows: ModelRow[]) => void
}
const Ctx = createContext<AnalyseState | null>(null)
const useAnalyse = () => {
  const v = useContext(Ctx)
  if (!v) throw new Error('AnalyseProvider missing')
  return v
}

const COLUMNS: Column[] = [
  { key: 'model_file', label: 'Datei', mono: true },
  { key: 'model_name', label: 'Modell', width: 150 },
  { key: 'accuracy', label: 'Accuracy', width: 100, align: 'right', mono: true },
  { key: 'training_time', label: 'Zeit', width: 96, mono: true },
  { key: 'date', label: 'Datum', width: 86 },
  { key: 'num_samples', label: 'Samples', width: 92, align: 'right', mono: true },
  { key: 'num_classes', label: 'Klassen', width: 86, align: 'right', mono: true },
  { key: 'test_size', label: 'Test', width: 74 },
  { key: 'use_lemma', label: 'Lemma', width: 80 },
  { key: 'use_spacy', label: 'spaCy', width: 80 },
  { key: 'use_dornseiff', label: 'Dornseiff', width: 96 },
  { key: 'min_word_len', label: 'Min-Länge', width: 96 },
  { key: 'analyzer', label: 'Analyzer', width: 104 },
  { key: 'stopwords_removed', label: 'Stoppw.', width: 88 },
]

const stemOf = (file: string) => file.replace(/\.pkl$/, '')
const hasReport = (r: ModelRow) => Boolean(r.has_report)

function saveText(name: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  URL.revokeObjectURL(url)
}

export function AnalyseProvider({ children }: { children: ReactNode }) {
  const { reloadModels } = useWorkbench()
  const [reportOpen, setReportOpen] = useState(false)
  const [reportFile, setReportFile] = useState('')
  const [reportText, setReportText] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)

  const openReport = useCallback((f: string) => {
    setReportOpen(true); setReportFile(f); setReportLoading(true); setReportText('')
    api.report(f).then((r) => setReportText(r.report)).catch((e) => setReportText(String(e))).finally(() => setReportLoading(false))
  }, [])

  // Klassifikations-Report(e) der ausgewählten Modelle herunterladen: einzeln
  // als .txt, bei mehreren gebündelt in einer Datei.
  const downloadReports = useCallback((rows: ModelRow[]) => {
    const files = rows.filter(hasReport).map((r) => String(r.model_file))
    if (files.length === 0) return
    setDownloading(true)
    Promise.all(files.map((f) => api.report(f).then((r) => ({ f, text: r.report })).catch((e) => ({ f, text: String(e) }))))
      .then((results) => {
        if (results.length === 1) {
          saveText(`${stemOf(results[0].f)}_report.txt`, results[0].text)
        } else {
          const bundle = results
            .map((r) => `===== ${r.f} =====\n\n${r.text}`)
            .join('\n\n\n')
          saveText('klassifikations-reports.txt', bundle)
        }
      })
      .finally(() => setDownloading(false))
  }, [])

  useRunAction(() => { void reloadModels() }, [reloadModels])

  return (
    <Ctx.Provider value={{
      reportOpen, reportFile, reportText, reportLoading, openReport, closeReport: () => setReportOpen(false),
      downloading, downloadReports,
    }}>
      {children}
    </Ctx.Provider>
  )
}

export function AnalyseConfig() {
  const { modelsLoading, reloadModels } = useWorkbench()

  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--lt-fg-3)', lineHeight: 1.5 }}>
        Vergleich aller trainierten Modelle. Spalten sind sortier- und filterbar (Filterzeile unter
        den Kopfzeilen). Zeilen per Checkbox auswählen; die Aktionen dazu liegen unter der Tabelle.
      </div>
      <GhostButton icon="refresh" onClick={() => void reloadModels()} disabled={modelsLoading}>
        {modelsLoading ? 'Lädt…' : 'Modelle neu laden'}
      </GhostButton>
    </div>
  )
}

export function AnalyseMain() {
  const { models, setSelectedModel, setActiveId } = useWorkbench()
  const { reportOpen, reportFile, reportText, reportLoading, openReport, closeReport, downloading, downloadReports } = useAnalyse()
  const rows = models?.models ?? []

  const pickForPrediction = (file: string) => { setSelectedModel(file); setActiveId('predict-single') }

  return (
    <>
      <DataTable
        title="Modelle"
        meta={<><span style={{ fontWeight: 600 }}>{rows.length}</span><span> Modelle</span></>}
        columns={COLUMNS}
        rows={rows}
        csvName="modelle.csv"
        emptyHint="Keine trainierten Modelle gefunden."
        rowKey={(r) => String(r.model_file)}
        selectable
        footerActions={(selected) => {
          const sel = selected as ModelRow[]
          const one = sel.length === 1 ? sel[0] : null
          const withReport = sel.filter(hasReport)
          return (
            <>
              <GhostButton
                icon="chart"
                onClick={() => one && openReport(String(one.model_file))}
                disabled={!one || !hasReport(one)}
                title={one ? undefined : 'Genau ein Modell auswählen'}
              >
                Report anzeigen
              </GhostButton>
              <GhostButton
                icon="download"
                onClick={() => downloadReports(withReport)}
                disabled={downloading || withReport.length === 0}
              >
                {downloading ? 'Lädt…' : `Report herunterladen${withReport.length > 1 ? ` (${withReport.length})` : ''}`}
              </GhostButton>
              <PrimaryButton
                icon="sparkle"
                onClick={() => one && pickForPrediction(String(one.model_file))}
                disabled={!one}
                style={{ height: 34 }}
              >
                Für Vorhersage nutzen
              </PrimaryButton>
            </>
          )
        }}
      />
      {reportOpen && <ReportModal text={reportText} loading={reportLoading} file={reportFile} onClose={closeReport} />}
    </>
  )
}

function ReportModal({ text, loading, file, onClose }: { text: string; loading: boolean; file: string; onClose: () => void }) {
  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(8,12,10,0.42)', backdropFilter: 'blur(1.5px)',
      display: 'flex', justifyContent: 'center', alignItems: 'flex-start', paddingTop: '6%',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 760, maxWidth: '92%', maxHeight: '82%', background: 'var(--lt-bg-0)', border: '1px solid var(--lt-line-2)',
        borderRadius: 'var(--lt-r-md)', boxShadow: 'var(--lt-shadow-pop)', overflow: 'hidden', display: 'flex', flexDirection: 'column',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', borderBottom: '1px solid var(--lt-line-1)' }}>
          <Icon name="chart" size={15} style={{ color: 'var(--lt-primary)' }} />
          <span style={{ fontSize: 14, fontWeight: 600, flex: 1 }}>Klassifikations-Report</span>
          <Badge tone="neutral">{file}</Badge>
          <Icon name="x" size={14} style={{ cursor: 'pointer', color: 'var(--lt-fg-3)' }} onClick={onClose} />
        </div>
        <div className="md-body" style={{ overflow: 'auto', padding: 16 }}>
          {loading ? <div style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>Lädt…</div>
            : <pre style={{ fontFamily: 'var(--lt-font-mono)', fontSize: 11.5, whiteSpace: 'pre', margin: 0 }}>{text}</pre>}
        </div>
      </div>
    </div>
  )
}
