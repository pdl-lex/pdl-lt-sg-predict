import { useEffect, useMemo, useRef, useState } from 'react'
import { Icon, type IconName } from '../design/icons'
import { Kbd } from '../design/widgets'
import { MODULES } from '../modules/registry'
import { useWorkbench } from '../state/workbench'

interface Action { group: string; icon: IconName; label: string; run: () => void }

export function CommandPalette({ onClose }: { onClose: () => void }) {
  const wb = useWorkbench()
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { inputRef.current?.focus() }, [])

  const actions = useMemo<Action[]>(() => [
    ...MODULES.map((m): Action => ({ group: 'Module', icon: m.icon, label: `${m.label} öffnen`, run: () => wb.setActiveId(m.id) })),
    { group: 'Aktionen', icon: 'play', label: 'Aktion des Moduls ausführen', run: () => wb.run() },
    { group: 'Aktionen', icon: 'refresh', label: 'Modelle neu laden', run: () => { void wb.reloadModels() } },
    { group: 'Aktionen', icon: wb.theme === 'dark' ? 'sun' : 'moon', label: wb.theme === 'dark' ? 'Heller Modus' : 'Dunkler Modus', run: () => wb.toggleTheme() },
    { group: 'Aktionen', icon: 'panelL', label: 'Layout · Konfiguration links', run: () => wb.setLayout('left') },
    { group: 'Aktionen', icon: 'panelR', label: 'Layout · Konfiguration rechts', run: () => wb.setLayout('right') },
    { group: 'Aktionen', icon: 'panelT', label: 'Layout · Konfiguration oben', run: () => wb.setLayout('bottom') },
  ], [wb])

  const q = query.trim().toLowerCase()
  const filtered = q ? actions.filter((a) => (a.label + ' ' + a.group).toLowerCase().includes(q)) : actions
  useEffect(() => { setActive(0) }, [query])

  const run = (a?: Action) => { if (!a) return; a.run(); onClose() }
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((i) => Math.min(i + 1, filtered.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((i) => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter') { e.preventDefault(); run(filtered[active]) }
  }

  let flat = -1
  const groups: { name: string; items: { a: Action; idx: number }[] }[] = []
  filtered.forEach((a) => {
    let g = groups.find((x) => x.name === a.group)
    if (!g) { g = { name: a.group, items: [] }; groups.push(g) }
    g.items.push({ a, idx: ++flat })
  })

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(8, 12, 10, 0.42)', backdropFilter: 'blur(1.5px)',
      display: 'flex', justifyContent: 'center', alignItems: 'flex-start', paddingTop: '10%',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 580, maxWidth: '86%', background: 'var(--lt-bg-0)', border: '1px solid var(--lt-line-2)',
        borderRadius: 'var(--lt-r-md)', boxShadow: 'var(--lt-shadow-pop)', overflow: 'hidden', display: 'flex', flexDirection: 'column',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 14px', borderBottom: '1px solid var(--lt-line-1)' }}>
          <Icon name="search" size={15} style={{ color: 'var(--lt-fg-3)' }} />
          <input ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKeyDown}
            placeholder="Befehl, Modul…" style={{ flex: 1, border: 'none', outline: 'none', background: 'transparent', fontSize: 14, color: 'var(--lt-fg-1)' }} />
          <Kbd>Esc</Kbd>
        </div>
        <div style={{ maxHeight: 360, overflowY: 'auto', padding: '6px 0' }}>
          {filtered.length === 0 && <div style={{ padding: '22px 16px', textAlign: 'center', color: 'var(--lt-fg-3)', fontSize: 13 }}>Keine Treffer für „{query}"</div>}
          {groups.map((g) => (
            <div key={g.name}>
              <div style={{ padding: '8px 16px 4px', fontSize: 10.5, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--lt-fg-4)', fontFamily: 'var(--lt-font-mono)' }}>{g.name}</div>
              {g.items.map(({ a, idx }) => (
                <div key={idx} onMouseEnter={() => setActive(idx)} onClick={() => run(a)} style={{
                  display: 'flex', alignItems: 'center', gap: 11, margin: '0 6px', padding: '8px 10px',
                  borderRadius: 'var(--lt-r-sm)', background: idx === active ? 'var(--lt-bg-2)' : 'transparent',
                  color: 'var(--lt-fg-1)', fontSize: 13, cursor: 'pointer',
                }}>
                  <Icon name={a.icon} size={14} style={{ color: 'var(--lt-fg-3)', flexShrink: 0 }} />
                  <span style={{ flex: 1 }}>{a.label}</span>
                  {idx === active && <Kbd>↵</Kbd>}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
