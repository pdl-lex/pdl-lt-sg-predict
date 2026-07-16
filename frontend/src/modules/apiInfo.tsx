// API-Referenz: dokumentiert die REST-Endpunkte (Einzel-/Batch-Vorhersage, SHAP,
// Modell-Liste) mit kopierbaren Beispielabfragen. Backend: api/routers/predict.py.
import { useState, type ReactNode } from 'react'
import { Icon } from '../design/icons'
import { Badge, Callout, Card, GhostButton, ResultsFrame } from '../design/ui'
import { useWorkbench } from '../state/workbench'

// Im Dev-Modus läuft die API auf Port 8000 (Vite proxyt nur /api, nicht /docs);
// in Produktion liefert FastAPI Frontend und API same-origin aus.
const BASE = import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin

function CodeBlock({ label, code }: { label?: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    void navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1400)
  }
  return (
    <div style={{ minWidth: 0 }}>
      {label && (
        <div style={{
          fontSize: 10.5, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase',
          color: 'var(--lt-fg-4)', fontFamily: 'var(--lt-font-mono)', margin: '0 0 4px',
        }}>{label}</div>
      )}
      <div style={{ position: 'relative' }}>
        <pre style={{
          margin: 0, padding: '10px 12px', background: 'var(--lt-bg-2)',
          border: '1px solid var(--lt-line-1)', borderRadius: 'var(--lt-r-sm)',
          fontSize: 11.5, lineHeight: 1.55, fontFamily: 'var(--lt-font-mono)',
          color: 'var(--lt-fg-1)', overflowX: 'auto',
        }}>{code}</pre>
        <button onClick={copy} title="In Zwischenablage kopieren" style={{
          position: 'absolute', top: 6, right: 6, width: 24, height: 24,
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--lt-bg-0)', border: '1px solid var(--lt-line-1)',
          borderRadius: 'var(--lt-r-sm)', color: copied ? 'var(--lt-primary)' : 'var(--lt-fg-3)',
          cursor: 'pointer',
        }}>
          <Icon name={copied ? 'check' : 'file'} size={11} />
        </button>
      </div>
    </div>
  )
}

function Endpoint({ method, path, children }: { method: 'GET' | 'POST'; path: string; children: ReactNode }) {
  return (
    <Card>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Badge tone={method === 'GET' ? 'info' : 'primary'}>{method}</Badge>
        <code style={{ fontFamily: 'var(--lt-font-mono)', fontSize: 12.5, fontWeight: 600, color: 'var(--lt-fg-1)' }}>{path}</code>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12.5, color: 'var(--lt-fg-2)', lineHeight: 1.55 }}>
        {children}
      </div>
    </Card>
  )
}

export function ApiInfoConfig() {
  const { models } = useWorkbench()
  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Card>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Schnittstelle</div>
        <div style={{ fontSize: 11.5, color: 'var(--lt-fg-3)', lineHeight: 1.7, fontFamily: 'var(--lt-font-mono)' }}>
          <div style={{ wordBreak: 'break-all' }}>Basis: {BASE}/api</div>
          <div>Format: JSON (Batch: multipart)</div>
          <div>Auth: keine</div>
          <div>Modelle: {models?.count ?? '–'}</div>
        </div>
      </Card>
      <Card>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Interaktive Dokumentation</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <GhostButton icon="book" onClick={() => window.open(`${BASE}/docs`, '_blank')}>Swagger UI (/docs)</GhostButton>
          <GhostButton icon="file" onClick={() => window.open(`${BASE}/openapi.json`, '_blank')}>OpenAPI-Schema (JSON)</GhostButton>
        </div>
      </Card>
      <Callout tone="info" icon="info">
        Gültige Werte für <code>model_file</code> liefert <code>GET /api/models</code> (Feld <code>files</code>);
        das aktuell beste Modell steht in <code>best.model_file</code>.
      </Callout>
    </div>
  )
}

export function ApiInfoMain() {
  const { selectedModel, best } = useWorkbench()
  const model = selectedModel || best?.model_file || 'modell.pkl'

  const singleRequest = `curl -X POST ${BASE}/api/predict/single \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_file": "${model}",
    "lemma": "Almrausch",
    "bedeutung": "Alpenrose",
    "top_k": 3
  }'`

  const singleResponse = `{
  "prediction": "3500",
  "description": "Wiesen- und Waldpflanze/sonstige Pflanze",
  "top": [
    { "label": "3500", "description": "Wiesen- und Waldpflanze/sonstige Pflanze", "proba": 62.4, "is_best": true },
    { "label": "3000", "description": "Pflanze und ihre Frucht", "proba": 21.9, "is_best": false },
    { "label": "3010", "description": "allg. Pflanzenteile (Knospe, Zweig u.a.)", "proba": 4.7, "is_best": false }
  ],
  "model_type": "nn",
  "uses_lemma": true
}`

  const batchRequest = `curl -X POST ${BASE}/api/predict/batch \\
  -F "model_file=${model}" \\
  -F "file=@eintraege.csv"`

  const batchCsv = `lemma;bedeutung
Almrausch;Alpenrose
;kleiner Wasserlauf`

  const batchResponse = `{
  "count": 2,
  "uses_lemma": true,
  "rows": [
    {
      "lemma": "Almrausch",
      "bedeutung": "Alpenrose",
      "sachgruppe": "3500",
      "beschreibung": "Wiesen- und Waldpflanze/sonstige Pflanze",
      "wahrscheinlichkeit": "62.4%",
      "sachgruppe_2": "3000",
      "beschreibung_2": "Pflanze und ihre Frucht",
      "wahrscheinlichkeit_2": "21.9%",
      "sachgruppe_3": "…", "beschreibung_3": "…", "wahrscheinlichkeit_3": "…"
    }
  ]
}`

  const shapRequest = `curl -X POST ${BASE}/api/predict/shap \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_file": "${model}",
    "lemma": "Almrausch",
    "bedeutung": "Alpenrose",
    "predicted_label": "3500",
    "filter_stopwords": true
  }'`

  return (
    <ResultsFrame title="API-Referenz" meta={<span>{BASE}/api</span>}>
      <div style={{ maxWidth: 860, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <Callout tone="primary" icon="api">
          Alle Endpunkte liegen unter <code>{BASE}/api</code>, Antworten sind JSON, eine Authentifizierung
          ist nicht erforderlich. Fehler kommen als HTTP-Status (404 = Modell unbekannt, 422 = ungültige
          Eingabe, 500 = interner Fehler) mit Body <code>{'{"detail": "…"}'}</code>.
        </Callout>

        <Endpoint method="POST" path="/api/predict/single">
          <p style={{ margin: 0 }}>
            Einzelvorhersage für einen Eintrag (<code>lemma</code> optional, <code>bedeutung</code> Pflicht).
            {' '}<code>top_k</code> (1–10, Standard 3) steuert die Länge des Rankings; <code>proba</code> ist die
            Wahrscheinlichkeit in Prozent (bei SVM-Modellen ohne Kalibrierung <code>null</code>).
          </p>
          <CodeBlock label="Anfrage" code={singleRequest} />
          <CodeBlock label="Antwort" code={singleResponse} />
        </Endpoint>

        <Endpoint method="POST" path="/api/predict/batch">
          <p style={{ margin: 0 }}>
            Batch-Vorhersage über eine CSV-Datei (multipart/form-data). Die CSV braucht mindestens die
            Spalte <code>bedeutung</code>; <code>lemma</code> ist optional, der Trenner
            (<code>;</code> <code>,</code> Tab) wird automatisch erkannt. Jede Zeile der Antwort enthält
            die Top-3-Sachgruppen mit Beschreibung und Wahrscheinlichkeit.
          </p>
          <CodeBlock label="Anfrage" code={batchRequest} />
          <CodeBlock label="eintraege.csv" code={batchCsv} />
          <CodeBlock label="Antwort" code={batchResponse} />
        </Endpoint>

        <Endpoint method="POST" path="/api/predict/shap">
          <p style={{ margin: 0 }}>
            SHAP-Worterklärung zu einer Einzelvorhersage: welche Wörter aus Lemma und Bedeutung haben wie
            stark für die Sachgruppe <code>predicted_label</code> gesprochen. Antwort:
            {' '}<code>{'{"lemma": [{"word", "score"}], "bedeutung": […], "is_nn": bool}'}</code>.
          </p>
          <CodeBlock label="Anfrage" code={shapRequest} />
        </Endpoint>

        <Endpoint method="GET" path="/api/models">
          <p style={{ margin: 0 }}>
            Liste aller trainierten Modelle. <code>files</code> enthält die gültigen Werte für
            {' '}<code>model_file</code>, <code>best</code> das Modell mit der höchsten Genauigkeit.
          </p>
          <CodeBlock label="Anfrage" code={`curl ${BASE}/api/models`} />
        </Endpoint>

        <Endpoint method="GET" path="/api/health">
          <p style={{ margin: 0 }}>
            Liveness-Check, antwortet mit <code>{'{"status": "ok"}'}</code>.
          </p>
          <CodeBlock label="Anfrage" code={`curl ${BASE}/api/health`} />
        </Endpoint>
      </div>
    </ResultsFrame>
  )
}
