// Zentraler Workbench-Zustand: Layout, Theme, aktives Modul, Modell-Liste und
// das aktuell gewählte Vorhersage-Modell. Modulspezifischer Zustand lebt in den
// jeweiligen Modul-Providern (siehe modules/*).
import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode,
} from 'react'
import { api, type AppConfig, type BestModel, type ModelsResponse, type ModelType } from '../api/client'

export type LayoutMode = 'left' | 'right' | 'bottom'

interface WorkbenchState {
  theme: 'light' | 'dark'
  toggleTheme: () => void
  layout: LayoutMode
  setLayout: (l: LayoutMode) => void
  railPinned: boolean
  setRailPinned: (v: boolean) => void
  activeId: string
  setActiveId: (id: string) => void

  config: AppConfig | null
  enableTraining: boolean
  modelTypes: ModelType[]

  models: ModelsResponse | null
  modelsLoading: boolean
  reloadModels: () => Promise<void>
  best: BestModel | null

  selectedModel: string
  setSelectedModel: (f: string) => void

  run: () => void
  registerRun: (fn: (() => void) | null) => void
}

const Ctx = createContext<WorkbenchState | null>(null)

export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [layout, setLayout] = useState<LayoutMode>('left')
  const [railPinned, setRailPinned] = useState(false)
  const [activeId, setActiveId] = useState('predict-single')

  const [config, setConfig] = useState<AppConfig | null>(null)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')

  const runRef = useRef<(() => void) | null>(null)

  useEffect(() => { document.documentElement.setAttribute('data-theme', theme) }, [theme])

  const reloadModels = useCallback(async () => {
    setModelsLoading(true)
    try {
      const res = await api.models()
      setModels(res)
      // Standardauswahl: bestes Modell, sonst neuestes.
      setSelectedModel((prev) => {
        if (prev && res.files.includes(prev)) return prev
        return res.best?.model_file ?? res.files[0] ?? ''
      })
    } catch { /* Fehler werden in den Modulen angezeigt */ }
    finally { setModelsLoading(false) }
  }, [])

  useEffect(() => {
    api.config().then(setConfig).catch(() => setConfig(null))
    void reloadModels()
  }, [reloadModels])

  const registerRun = useCallback((fn: (() => void) | null) => { runRef.current = fn }, [])
  const run = useCallback(() => { runRef.current?.() }, [])

  const value = useMemo<WorkbenchState>(() => ({
    theme, toggleTheme: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')),
    layout, setLayout, railPinned, setRailPinned,
    activeId, setActiveId,
    config, enableTraining: config?.enable_training ?? true, modelTypes: config?.model_types ?? [],
    models, modelsLoading, reloadModels, best: models?.best ?? null,
    selectedModel, setSelectedModel,
    run, registerRun,
  }), [theme, layout, railPinned, activeId, config, models, modelsLoading, reloadModels, selectedModel, run, registerRun])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useWorkbench(): WorkbenchState {
  const v = useContext(Ctx)
  if (!v) throw new Error('useWorkbench must be used within WorkbenchProvider')
  return v
}

/** Registriert die Primäraktion des aktiven Moduls für ⌘↵ und die Befehlspalette. */
export function useRunAction(fn: () => void, deps: unknown[]) {
  const { registerRun } = useWorkbench()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { registerRun(fn); return () => registerRun(null) }, deps)
}
