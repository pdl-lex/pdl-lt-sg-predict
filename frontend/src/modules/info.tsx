// Info-Seite: Projekt, Daten/Lizenz, genutzte Software, Konzeption, Impressum.
// Analog zur "Informationen"-Seite in pdl-register-fischer-web (Rail-Icon statt Zahnrad).
import type { ReactNode } from 'react'
import { Card, ResultsFrame } from '../design/ui'

function Section({ eyebrow, children }: { eyebrow: string; children: ReactNode }) {
  return (
    <Card>
      <div className="lt-eyebrow" style={{ marginBottom: 6 }}>{eyebrow}</div>
      <div style={{ fontSize: 12.5, color: 'var(--lt-fg-2)', lineHeight: 1.6, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {children}
      </div>
    </Card>
  )
}

export function InfoConfig() {
  return (
    <div className="cfg-scroll" style={{ overflowY: 'auto', flex: 1, background: 'var(--lt-bg-2)', padding: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Card>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Kurzfassung</div>
        <div style={{ fontSize: 11.5, color: 'var(--lt-fg-3)', lineHeight: 1.7, fontFamily: 'var(--lt-font-mono)' }}>
          <div>Lizenz: CC-BY 4.0</div>
          <div>Träger: BAdW</div>
        </div>
      </Card>
    </div>
  )
}

export function InfoMain() {
  return (
    <ResultsFrame title="Informationen" meta={<span>Über dieses Werkzeug</span>}>
      <div style={{ maxWidth: 680, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <Section eyebrow="Über dieses Werkzeug">
          <p style={{ margin: 0 }}>
            LexoTerm Tools sagt die <strong>Sachgruppe</strong> eines Wörterbucheintrags aus Lemma
            und Bedeutungsdefinition per Machine Learning voraus. Entwickelt für die BDO-Wörterbücher
            „Fränkisches Wörterbuch" und „Dialektologisches Informationssystem Bayerisch-Schwaben"
            (Bayerische Akademie der Wissenschaften).
          </p>
        </Section>
        <Section eyebrow="Daten &amp; Modelle">
          <p style={{ margin: 0 }}>
            Die Modelle werden auf vorhandenen Wörterbuchdaten trainiert; die Sachgruppen-Taxonomie
            folgt Hallig/Wartburg/Post. Vorhersagen sind statistischer Natur — bei mehrdeutigen oder
            unterrepräsentierten Bedeutungen können sie danebenliegen; deshalb liefert jede Vorhersage
            Top-k-Alternativen samt Wahrscheinlichkeit und eine SHAP-Worterklärung.
          </p>
        </Section>
        <Section eyebrow="Literatur">
          <p style={{ margin: 0 }}>
            Rudolf Post: Möglichkeiten der elektronischen Strukturierung, Vernetzung und
            Verfügbarmachung von lexikographischen Daten bei der Arbeit am Pfälzischen Wörterbuch.
            In: Rudolf Grosse (Hrsg.): Bedeutungserfassung und Bedeutungsbeschreibung in
            historischen und dialektologischen Wörterbüchern. Beiträge zu einer Arbeitstagung der
            deutschsprachigen Wörterbücher. Projekte an Akademien und Universitäten vom 7. bis
            9. März 1996 anläßlich des 150jährigen Jubiläums der Sächsischen Akademie der
            Wissenschaften zu Leipzig. Stuttgart, Leipzig: Hirzel 1998, S. 211–220 (Abhandlungen
            der Sächsischen Akademie der Wissenschaften zu Leipzig, Philologisch-historische
            Klasse, Bd. 75, H. 1)
          </p>
          <p style={{ margin: 0 }}>
            Rudolf Hallig / Walther von Wartburg: Begriffssystem als Grundlage für die
            Lexikographie. Versuch eines Ordnungsschemas / Système raisonné des concepts pour
            servir de base à la lexicographie. 1. Aufl. Berlin: Akademie-Verlag 1952 (Abhandlungen
            der Deutschen Akademie der Wissenschaften zu Berlin, Kl. f. Sprachen, Literatur und
            Kunst 1952/4). — 2., neu bearbeitete und erweiterte Aufl. Berlin: Akademie-Verlag 1963
            (Veröffentlichungen des Instituts für Romanische Sprachwissenschaft 19), 315 S.; seit
            2021 als De-Gruyter-Reprint (ISBN 978-3-11-258029-5).
          </p>
        </Section>
        <Section eyebrow="Genutzte Software">
          <p style={{ margin: 0 }}>Wir danken den Autoren der verwendeten Software:</p>
          <p style={{ margin: 0 }}>scikit-learn: <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer">https://scikit-learn.org/</a></p>
          <p style={{ margin: 0 }}>XGBoost: <a href="https://xgboost.ai/" target="_blank" rel="noreferrer">https://xgboost.ai/</a></p>
          <p style={{ margin: 0 }}>SHAP: <a href="https://shap.readthedocs.io/" target="_blank" rel="noreferrer">https://shap.readthedocs.io/</a></p>
        </Section>
        <Section eyebrow="Lizenz">
          <p style={{ margin: 0 }}>Der Quelltext und die Modelle dieses Tools sind unter CC-BY 4.0 veröffentlicht (Bayerische Akademie der Wissenschaften).</p>
          <p style={{ margin: 0 }}>Der Code steht unter <a href="https://github.com/pdl-lex/pdl-lt-sg-predict" target="_blank" rel="noreferrer">https://github.com/pdl-lex/pdl-lt-sg-predict</a> zur Verfügung.</p>
        </Section>
        <Section eyebrow="Konzeption &amp; Programmierung">
          <p style={{ margin: 0 }}>
            Wolfgang Huang<br />
            <a href="https://pdl.badw.de" target="_blank" rel="noreferrer">Neue Potenziale für die Digitale Lexikographie des Deutschen</a><br />
            Bayerische Akademie der Wissenschaften
          </p>
        </Section>
        <Section eyebrow="Impressum">
          <p style={{ margin: 0 }}><a href="https://badw.de/impressum" target="_blank" rel="noreferrer">Siehe Impressum der BAdW</a></p>
        </Section>
      </div>
    </ResultsFrame>
  )
}
