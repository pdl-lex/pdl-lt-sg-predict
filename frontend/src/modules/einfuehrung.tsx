import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { Markdown } from '../design/markdown'
import { Badge, Callout, Card, ResultsFrame } from '../design/ui'
import { useWorkbench } from '../state/workbench'

const FEATURES: { badge: string; text: string }[] = [
  { badge: 'Vorhersage', text: 'Sachgruppe für einzelne Einträge (lemma + bedeutung) inkl. SHAP-Erklärung.' },
  { badge: 'Batch', text: 'Vorhersage für ganze CSV-Dateien, als CSV exportierbar.' },
  { badge: 'Analyse', text: 'Übersicht und Vergleich aller trainierten Modelle.' },
  { badge: 'Sachgruppen', text: 'Taxonomie nach Hallig-Wartburg mit Klassifikationsmetriken.' },
  { badge: 'Training', text: 'Training neuer Modelle (einzeln oder als Batch) auf eigenen Daten.' },
]

export function EinfuehrungConfig() {
  const { models, config } = useWorkbench()
  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Card>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Funktionen</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
          {FEATURES.map((f) => (
            <div key={f.badge} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <Badge tone="primary">{f.badge}</Badge>
              <span style={{ fontSize: 11.5, color: 'var(--lt-fg-3)', lineHeight: 1.45 }}>{f.text}</span>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Umgebung</div>
        <div style={{ fontSize: 11.5, color: 'var(--lt-fg-3)', lineHeight: 1.7, fontFamily: 'var(--lt-font-mono)' }}>
          <div>Modelle: {models?.count ?? '–'}</div>
          <div>Training: {config ? (config.enable_training ? 'aktiv' : 'deaktiviert') : '–'}</div>
          <div style={{ wordBreak: 'break-all' }}>Verz.: {config?.models_dir ?? '–'}</div>
        </div>
      </Card>
    </div>
  )
}

export function EinfuehrungMain() {
  const [text, setText] = useState<string | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    api.anleitung().then((r) => setText(r.markdown)).catch((e) => setError(String(e)))
  }, [])
  return (
    <ResultsFrame title="Einführung" meta={<span>LexoTerm Sachgruppen-Vorhersage</span>}>
      {error ? <Callout tone="err" icon="warn">{error}</Callout>
        : text == null ? <div style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>Lädt…</div>
        : <Markdown text={text} />}
    </ResultsFrame>
  )
}
