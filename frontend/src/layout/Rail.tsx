import { useState, type CSSProperties } from 'react'
import { Icon, type IconName } from '../design/icons'
import { GROUP_ICON, MODULES, MODULE_GROUPS } from '../modules/registry'
import { useWorkbench } from '../state/workbench'

interface MenuItem { id: string; label: string }
interface MenuGroup { group: string; icon: IconName; items: MenuItem[] }

const MENU: MenuGroup[] = MODULE_GROUPS.map((group) => ({
  group, icon: GROUP_ICON[group],
  items: MODULES.filter((m) => m.group === group).map((m) => ({ id: m.id, label: m.label })),
}))

function railRow(active: boolean): CSSProperties {
  return {
    width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '7px 8px',
    background: 'transparent', border: 'none', borderRadius: 'var(--lt-r-sm)',
    color: active ? 'var(--lt-fg-1)' : 'var(--lt-fg-2)', cursor: 'pointer', textAlign: 'left',
  }
}

function RailGlyph({ icon, label, active, count }: { icon: IconName; label: string; active: boolean; count: number }) {
  return (
    <div title={label} style={{
      width: 40, height: 40, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      position: 'relative', background: active ? 'var(--lt-primary-soft)' : 'transparent',
      color: active ? 'var(--lt-primary)' : 'var(--lt-fg-3)',
      border: '1px solid ' + (active ? 'var(--lt-primary-line)' : 'transparent'), borderRadius: 'var(--lt-r-md)',
    }}>
      <Icon name={icon} size={16} />
      {count > 1 && <span style={{
        position: 'absolute', top: 2, right: 2, fontSize: 9, fontFamily: 'var(--lt-font-mono)',
        color: active ? 'var(--lt-primary)' : 'var(--lt-fg-4)',
      }}>{count}</span>}
    </div>
  )
}

function RailGroup({ g, open, onToggle }: { g: MenuGroup; open: boolean; onToggle: () => void }) {
  const { activeId, setActiveId } = useWorkbench()
  const hasActive = g.items.some((it) => it.id === activeId)
  return (
    <div style={{ marginBottom: 2 }}>
      <button onClick={onToggle} style={{ ...railRow(hasActive), fontWeight: 600, fontSize: 13 }}>
        <Icon name={g.icon} size={15} style={{ color: hasActive ? 'var(--lt-primary)' : 'var(--lt-fg-3)' }} />
        <span style={{ flex: 1 }}>{g.group}</span>
        <Icon name={open ? 'chevDown' : 'chevron'} size={10} style={{ color: 'var(--lt-fg-4)' }} />
      </button>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 1, padding: '2px 0 4px' }}>
          {g.items.map((it) => {
            const active = it.id === activeId
            return (
              <button key={it.id} onClick={() => setActiveId(it.id)} style={{
                ...railRow(active), padding: '5px 8px 5px 34px', fontSize: 12.5,
                background: active ? 'var(--lt-primary-soft)' : 'transparent',
                color: active ? 'var(--lt-primary)' : 'var(--lt-fg-2)', fontWeight: active ? 600 : 400,
              }}>
                {it.label}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

export function Rail() {
  const { railPinned, setRailPinned, activeId } = useWorkbench()
  const [hover, setHover] = useState(false)
  const activeGroup = MENU.find((g) => g.items.some((it) => it.id === activeId))?.group ?? MENU[0].group
  const [openGroup, setOpenGroup] = useState<string | null>(activeGroup)
  const expanded = railPinned || hover

  return (
    <nav onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} style={{
      gridArea: 'rail', position: 'relative', background: 'var(--lt-bg-0)',
      borderRight: '1px solid var(--lt-line-1)', zIndex: 30,
    }}>
      {!railPinned && (
        <div onClick={() => setRailPinned(true)} title="Navigation öffnen" style={{
          width: 56, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center',
          padding: '12px 0 10px', gap: 4, boxSizing: 'border-box', cursor: 'pointer',
        }}>
          {MENU.map((g) => (
            <RailGlyph key={g.group} icon={g.icon} label={g.group} active={g.group === activeGroup} count={g.items.length} />
          ))}
          <span style={{ flex: 1 }} />
          <RailGlyph icon="settings" label="Einstellungen" active={false} count={0} />
        </div>
      )}

      <div style={{
        position: railPinned ? 'relative' : 'absolute', top: 0, left: 0, bottom: 0,
        width: railPinned ? '100%' : 248, height: '100%', background: 'var(--lt-bg-0)',
        borderRight: railPinned ? 'none' : '1px solid var(--lt-line-1)',
        boxShadow: !railPinned && hover ? 'var(--lt-shadow-pop)' : 'none',
        display: 'flex', flexDirection: 'column', overflow: 'hidden', boxSizing: 'border-box',
        opacity: expanded ? 1 : 0, transform: expanded ? 'translateX(0)' : 'translateX(-6px)',
        pointerEvents: expanded ? 'auto' : 'none', transition: 'opacity .14s ease, transform .14s ease',
      }}>
        <div style={{
          height: 44, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 8,
          padding: '0 8px 0 14px', borderBottom: '1px solid var(--lt-line-1)',
        }}>
          <span className="lt-eyebrow" style={{ flex: 1 }}>Navigation</span>
          <button title={railPinned ? 'Einklappen' : 'Anheften'} onClick={() => setRailPinned(!railPinned)} style={{
            width: 26, height: 26, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            background: railPinned ? 'var(--lt-primary-soft)' : 'transparent',
            color: railPinned ? 'var(--lt-primary)' : 'var(--lt-fg-3)',
            border: '1px solid ' + (railPinned ? 'var(--lt-primary-line)' : 'var(--lt-line-1)'),
            borderRadius: 'var(--lt-r-sm)', cursor: 'pointer',
          }}>
            <Icon name="pin" size={13} />
          </button>
        </div>

        <div style={{ overflowY: 'auto', flex: 1, padding: '8px' }}>
          {MENU.map((g) => (
            <RailGroup key={g.group} g={g} open={openGroup === g.group}
              onToggle={() => setOpenGroup(openGroup === g.group ? null : g.group)} />
          ))}
        </div>

        <div style={{ borderTop: '1px solid var(--lt-line-1)', padding: '8px' }}>
          <button style={railRow(false)}>
            <Icon name="settings" size={15} style={{ color: 'var(--lt-fg-3)' }} />
            <span style={{ flex: 1, fontSize: 13 }}>Einstellungen</span>
          </button>
        </div>
      </div>
    </nav>
  )
}
