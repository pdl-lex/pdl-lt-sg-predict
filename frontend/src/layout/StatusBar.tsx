import type { ReactNode } from 'react'
import { Kbd, kc } from '../design/widgets'
import { useWorkbench } from '../state/workbench'

function Item({ dot, k, v }: { dot?: string; k?: string; v: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0 10px', height: 32, borderRight: '1px solid var(--lt-line-1)' }}>
      {dot && <span style={{ width: 6, height: 6, borderRadius: 3, background: dot }} />}
      {k && <span style={{ color: 'var(--lt-fg-4)' }}>{k}</span>}
      <span>{v}</span>
    </div>
  )
}

export function StatusBar() {
  const { models, best, enableTraining, config } = useWorkbench()
  const count = models?.count ?? 0
  return (
    <footer style={{
      gridArea: 'stat', display: 'flex', alignItems: 'center', background: 'var(--lt-bg-0)',
      borderTop: '1px solid var(--lt-line-1)', fontSize: 11, fontFamily: 'var(--lt-font-mono)', color: 'var(--lt-fg-3)',
    }}>
      <Item dot={count > 0 ? 'var(--lt-primary)' : 'var(--lt-warn)'} k="Modelle:" v={String(count)} />
      <Item k="Bestes:" v={best ? `${best.model_name} (${best.accuracy.toFixed(4)})` : '–'} />
      <Item dot={enableTraining ? 'var(--lt-primary)' : 'var(--lt-warn)'} k="Training:" v={enableTraining ? 'aktiv' : 'aus'} />
      <Item k="Daten:" v={config?.models_dir ?? '–'} />
      <span style={{ flex: 1 }} />
      <Item v="UTF-8" />
      <Item v="DE" />
      <span style={{ padding: '0 12px', color: 'var(--lt-fg-4)' }}>
        <Kbd>{kc('K')}</Kbd> Befehle
      </span>
    </footer>
  )
}
