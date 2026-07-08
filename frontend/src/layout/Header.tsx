import type { CSSProperties } from 'react'
import { Icon, Logo } from '../design/icons'
import { Kbd, kc } from '../design/widgets'
import { moduleById } from '../modules/registry'
import { useWorkbench } from '../state/workbench'

const iconBtnSquare: CSSProperties = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
  width: 28, height: 28, background: 'var(--lt-bg-1)', border: '1px solid var(--lt-line-1)',
  borderRadius: 'var(--lt-r-sm)', color: 'var(--lt-fg-2)', cursor: 'pointer',
}

function iconChip(active: boolean): CSSProperties {
  return {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 22, height: 20,
    background: active ? 'var(--lt-bg-0)' : 'transparent',
    color: active ? 'var(--lt-fg-1)' : 'var(--lt-fg-3)', border: 'none', borderRadius: 3,
    cursor: 'pointer', boxShadow: active ? 'var(--lt-shadow-1)' : 'none',
  }
}

export function Header({ onOpenPalette }: { onOpenPalette: () => void }) {
  const { theme, toggleTheme, layout, setLayout, activeId } = useWorkbench()
  const module = moduleById(activeId)
  const dark = theme === 'dark'
  return (
    <header style={{
      gridArea: 'head', display: 'flex', alignItems: 'center', background: 'var(--lt-bg-0)',
      borderBottom: '1px solid var(--lt-line-1)', padding: '0 12px', gap: 14,
    }}>
      <Logo size={18} />
      <span style={{ fontWeight: 600, fontSize: 13 }}>LexoTerm Tools</span>
      <span style={{ color: 'var(--lt-fg-4)' }}>/</span>
      <span style={{ color: 'var(--lt-fg-3)', fontSize: 13 }}>Sachgruppen-Vorhersage</span>
      <span style={{ color: 'var(--lt-fg-4)' }}>/</span>
      <span style={{ color: 'var(--lt-fg-2)', fontSize: 13 }}>{module.title}</span>

      <span style={{ flex: 1 }} />

      <button onClick={onOpenPalette} style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '5px 10px', background: 'var(--lt-bg-1)',
        border: '1px solid var(--lt-line-1)', borderRadius: 'var(--lt-r-md)', color: 'var(--lt-fg-3)',
        fontSize: 12, width: 240, cursor: 'pointer',
      }}>
        <Icon name="search" size={12} />
        <span>Befehl, Modul…</span>
        <span style={{ flex: 1 }} />
        <Kbd>{kc('K')}</Kbd>
      </button>

      <div style={{
        display: 'inline-flex', alignItems: 'center', border: '1px solid var(--lt-line-1)',
        borderRadius: 'var(--lt-r-sm)', padding: 2, background: 'var(--lt-bg-1)',
      }}>
        <button onClick={() => setLayout('left')} style={iconChip(layout === 'left')} title="Konfiguration links"><Icon name="panelL" size={12} /></button>
        <button onClick={() => setLayout('right')} style={iconChip(layout === 'right')} title="Konfiguration rechts"><Icon name="panelR" size={12} /></button>
        <button onClick={() => setLayout('bottom')} style={iconChip(layout === 'bottom')} title="Konfiguration oben"><Icon name="panelT" size={12} /></button>
      </div>

      <button onClick={toggleTheme} style={iconBtnSquare} title={dark ? 'Heller Modus' : 'Dunkler Modus'}>
        <Icon name={dark ? 'moon' : 'sun'} size={13} />
      </button>
    </header>
  )
}
