import type { ModuleDef } from '../modules/registry'
import type { LayoutMode } from '../state/workbench'

export function ConfigPane({ module, layout }: { module: ModuleDef; layout: LayoutMode }) {
  const { Config, Footer } = module
  return (
    <section style={{
      gridArea: 'cfg', background: 'var(--lt-bg-0)',
      borderRight: layout === 'left' ? '1px solid var(--lt-line-1)' : 'none',
      borderLeft: layout === 'right' ? '1px solid var(--lt-line-1)' : 'none',
      borderBottom: layout === 'bottom' ? '1px solid var(--lt-line-1)' : 'none',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{ padding: '14px 18px 12px', background: 'var(--lt-bg-2)', borderBottom: '1px solid var(--lt-line-1)' }}>
        <div className="lt-eyebrow" style={{ marginBottom: 4 }}>{module.eyebrow}</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{module.title}</h2>
          {module.tag && <span style={{ fontSize: 11, color: 'var(--lt-fg-3)', fontFamily: 'var(--lt-font-mono)' }}>{module.tag}</span>}
        </div>
        <p style={{ margin: '6px 0 0', color: 'var(--lt-fg-3)', fontSize: 12, lineHeight: 1.45 }}>{module.description}</p>
      </div>

      <Config />

      {Footer && (
        <div style={{
          minHeight: 56, boxSizing: 'border-box', padding: '10px 14px', borderTop: '1px solid var(--lt-line-1)',
          background: 'var(--lt-bg-2)', display: 'flex', gap: 10, alignItems: 'center',
        }}>
          <Footer />
        </div>
      )}
    </section>
  )
}
