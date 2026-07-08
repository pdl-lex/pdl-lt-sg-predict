// Mini-Charts und kleine Steuerelemente aus dem Design-Handoff.
import { useState, type CSSProperties, type ReactNode } from 'react'

export interface SparkDatum { name: string; count: number; pct?: number }

export function Sparkbars({
  data, max, color = 'var(--lt-primary)', width = 120, height = 28, gap = 2,
  interactive = false, activeIndex = null, onSelect,
}: {
  data: SparkDatum[]; max?: number; color?: string; width?: number; height?: number
  gap?: number; interactive?: boolean; activeIndex?: number | null
  onSelect?: (i: number) => void
}) {
  const [hover, setHover] = useState<number | null>(null)
  const m = max ?? Math.max(...data.map((d) => d.count), 1)
  const bw = (width - gap * (data.length - 1)) / Math.max(data.length, 1)
  const totalC = data.reduce((s, d) => s + d.count, 0)
  return (
    <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
      <svg width={width} height={height} style={{ display: 'block', overflow: 'visible' }}>
        {data.map((d, i) => {
          const h = Math.max(2, (d.count / m) * height)
          const isActive = activeIndex === i
          const isHover = hover === i
          const faded = (activeIndex != null && !isActive) || (hover != null && !isHover)
          return (
            <rect key={i} x={i * (bw + gap)} y={height - h} width={bw} height={h}
              fill={color} rx={1}
              opacity={faded ? 0.26 : isActive || isHover ? 1 : 0.85}
              style={{ transition: 'opacity .1s' }} />
          )
        })}
        {interactive && data.map((_, i) => (
          <rect key={'hit' + i} x={i * (bw + gap)} y={0} width={bw} height={height}
            fill="transparent" style={{ cursor: 'pointer' }}
            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
            onClick={onSelect ? () => onSelect(i) : undefined} />
        ))}
      </svg>
      {interactive && hover != null && (
        <SparkTip d={data[hover]} totalC={totalC} x={hover * (bw + gap) + bw / 2} />
      )}
    </div>
  )
}

function SparkTip({ d, totalC, x }: { d: SparkDatum; totalC: number; x: number }) {
  const pct = d.pct != null ? d.pct : Math.round((d.count / Math.max(totalC, 1)) * 100)
  return (
    <div style={{
      position: 'absolute', left: x, bottom: 'calc(100% + 6px)', transform: 'translateX(-50%)',
      background: 'var(--lt-fg-1)', color: 'var(--lt-bg-0)', padding: '4px 8px',
      borderRadius: 'var(--lt-r-sm)', fontSize: 11, whiteSpace: 'nowrap', pointerEvents: 'none',
      boxShadow: 'var(--lt-shadow-pop)', zIndex: 50, lineHeight: 1.3,
    }}>
      <span style={{ fontWeight: 600 }}>{d.name}</span>
      <span style={{ opacity: 0.7 }}> · {d.count} · {pct}%</span>
    </div>
  )
}

export function HBar({
  value, max, color = 'var(--lt-primary)', track = 'var(--lt-bg-2)', height = 4, radius = 2,
}: { value: number; max: number; color?: string; track?: string; height?: number; radius?: number }) {
  const w = Math.max(2, (value / Math.max(max, 1)) * 100)
  return (
    <div style={{ height, background: track, borderRadius: radius, overflow: 'hidden', width: '100%' }}>
      <div style={{ width: `${w}%`, height: '100%', background: color, borderRadius: radius }} />
    </div>
  )
}

export function Segmented({
  options, value, onChange,
}: { options: string[]; value: string; onChange?: (v: string) => void }) {
  return (
    <div style={{
      display: 'inline-flex', padding: 2, gap: 2, background: 'var(--lt-bg-2)',
      border: '1px solid var(--lt-line-1)', borderRadius: 'var(--lt-r-sm)',
    }}>
      {options.map((opt) => {
        const active = opt === value
        return (
          <button key={opt} onClick={onChange ? () => onChange(opt) : undefined} style={{
            appearance: 'none', border: 'none', cursor: onChange ? 'pointer' : 'default',
            font: 'inherit', fontSize: 12, fontWeight: active ? 600 : 500, padding: '3px 12px',
            borderRadius: 'calc(var(--lt-r-sm) - 2px)',
            background: active ? 'var(--lt-bg-0)' : 'transparent',
            color: active ? 'var(--lt-fg-1)' : 'var(--lt-fg-3)',
            boxShadow: active ? 'var(--lt-shadow-1)' : 'none',
          }}>{opt}</button>
        )
      })}
    </div>
  )
}

const kbdStyle: CSSProperties = {
  fontFamily: 'var(--lt-font-mono)', fontSize: 11, padding: '1px 6px',
  border: '1px solid var(--lt-line-2)', borderBottomWidth: 2, borderRadius: 'var(--lt-r-xs)',
  color: 'var(--lt-fg-2)', background: 'var(--lt-bg-0)',
}
export function Kbd({ children }: { children: ReactNode }) {
  return <span style={kbdStyle}>{children}</span>
}

export const IS_MAC = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent || '')
export function kc(key: string) { return IS_MAC ? `⌘${key}` : `Strg+${key}` }
