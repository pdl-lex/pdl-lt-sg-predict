// Modul-Registry: verbindet Navigation, Konfigurator und Ergebnis-Panel.
// Jedes Modul liefert Provider (modul-lokaler Zustand), Config (Konfigurator-Pane),
// optional Footer (Primäraktion) und Main (Ergebnis-Panel).
import type { FC, ReactNode } from 'react'
import type { IconName } from '../design/icons'
import { EinfuehrungConfig, EinfuehrungMain } from './einfuehrung'
import { SingleProvider, SingleConfig, SingleFooter, SingleMain } from './predictSingle'
import { BatchProvider, BatchConfig, BatchFooter, BatchMain } from './predictBatch'
import { AnalyseProvider, AnalyseConfig, AnalyseMain } from './analyse'
import { SachgruppenProvider, SachgruppenConfig, SachgruppenMain } from './sachgruppen'
import { TrainingProvider, TrainingConfig, TrainingFooter, TrainingMain } from './training'

export interface ModuleDef {
  id: string
  label: string
  group: string
  eyebrow: string
  title: string
  tag?: string
  description: string
  icon: IconName
  Provider: FC<{ children: ReactNode }>
  Config: FC
  Footer?: FC
  Main: FC
}

const Passthrough: FC<{ children: ReactNode }> = ({ children }) => <>{children}</>

export const MODULES: ModuleDef[] = [
  {
    id: 'einfuehrung', label: 'Einführung', group: 'Start', icon: 'book',
    eyebrow: 'Start', title: 'Einführung',
    description: 'ML-gestützte Klassifikation von Sachgruppen aus Wörterbuchdaten (lemma + bedeutung).',
    Provider: Passthrough, Config: EinfuehrungConfig, Main: EinfuehrungMain,
  },
  {
    id: 'predict-single', label: 'Einzelvorhersage', group: 'Vorhersage', icon: 'sparkle',
    eyebrow: 'Vorhersage', title: 'Einzelvorhersage', tag: 'lemma + bedeutung',
    description: 'Sagt die Sachgruppe für einen einzelnen Eintrag voraus – mit Top-3 und SHAP-Erklärung.',
    Provider: SingleProvider, Config: SingleConfig, Footer: SingleFooter, Main: SingleMain,
  },
  {
    id: 'predict-batch', label: 'Batch-Vorhersage', group: 'Vorhersage', icon: 'table',
    eyebrow: 'Vorhersage', title: 'Batch-Vorhersage', tag: 'CSV',
    description: 'Vorhersage für eine ganze CSV-Datei; Ergebnisse als CSV exportierbar.',
    Provider: BatchProvider, Config: BatchConfig, Footer: BatchFooter, Main: BatchMain,
  },
  {
    id: 'analyse', label: 'Analyse', group: 'Modelle', icon: 'chart',
    eyebrow: 'Modelle', title: 'Analyse',
    description: 'Übersicht und Vergleich aller trainierten Modelle inkl. Klassifikations-Report.',
    Provider: AnalyseProvider, Config: AnalyseConfig, Main: AnalyseMain,
  },
  {
    id: 'sachgruppen', label: 'Sachgruppen', group: 'Modelle', icon: 'list',
    eyebrow: 'Modelle', title: 'Sachgruppen', tag: 'Hallig-Wartburg',
    description: 'Alle Sachgruppen mit Precision, Recall und F1 des besten Modells.',
    Provider: SachgruppenProvider, Config: SachgruppenConfig, Main: SachgruppenMain,
  },
  {
    id: 'training', label: 'Training', group: 'Modelle', icon: 'brain',
    eyebrow: 'Modelle', title: 'Training',
    description: 'Training neuer Modelle (einzeln oder als Batch) auf eigenen Daten.',
    Provider: TrainingProvider, Config: TrainingConfig, Footer: TrainingFooter, Main: TrainingMain,
  },
]

export const MODULE_GROUPS = ['Start', 'Vorhersage', 'Modelle'] as const

export const GROUP_ICON: Record<string, IconName> = {
  Start: 'book', Vorhersage: 'bolt', Modelle: 'layers',
}

export function moduleById(id: string): ModuleDef {
  return MODULES.find((m) => m.id === id) ?? MODULES[0]
}
