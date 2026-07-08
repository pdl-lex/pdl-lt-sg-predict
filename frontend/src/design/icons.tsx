// Dünnstrichiges Inline-SVG-Icon-Set + LexoTerm-Logo (aus dem Design-Handoff).
import type { CSSProperties, JSX } from 'react'

export type IconName =
  | 'search' | 'chevron' | 'chevDown' | 'plus' | 'x' | 'check' | 'download'
  | 'upload' | 'play' | 'filter' | 'sun' | 'moon' | 'settings' | 'grid'
  | 'table' | 'file' | 'folder' | 'book' | 'flask' | 'command' | 'bolt'
  | 'refresh' | 'dot' | 'diamond' | 'sparkle' | 'pin'
  | 'panelL' | 'panelR' | 'panelB' | 'panelT' | 'layers'
  | 'chart' | 'list' | 'brain' | 'info' | 'warn'

const PATHS: Record<IconName, JSX.Element> = {
  search: <><circle cx="7" cy="7" r="4.5" /><path d="M10.5 10.5 14 14" /></>,
  chevron: <path d="M5.5 3.5 10 8l-4.5 4.5" />,
  chevDown: <path d="M3.5 5.5 8 10l4.5-4.5" />,
  plus: <><path d="M8 3v10" /><path d="M3 8h10" /></>,
  x: <><path d="M4 4l8 8" /><path d="M12 4l-8 8" /></>,
  check: <path d="M3 8.5 6 11.5 13 4.5" />,
  download: <><path d="M8 3v8" /><path d="M5 8l3 3 3-3" /><path d="M3 13h10" /></>,
  upload: <><path d="M8 13V5" /><path d="M5 8l3-3 3 3" /><path d="M3 3h10" /></>,
  play: <path d="M5 3.5 12.5 8 5 12.5z" />,
  filter: <path d="M2 3h12l-4.5 6V13l-3-1.5V9z" />,
  sun: <><circle cx="8" cy="8" r="3" /><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.5 1.5M11.5 11.5 13 13M3 13l1.5-1.5M11.5 4.5 13 3" /></>,
  moon: <path d="M13 9.5A5 5 0 1 1 6.5 3a4 4 0 0 0 6.5 6.5z" />,
  settings: <g transform="scale(0.6667)" strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></g>,
  grid: <><rect x="2" y="2" width="5" height="5" /><rect x="9" y="2" width="5" height="5" /><rect x="2" y="9" width="5" height="5" /><rect x="9" y="9" width="5" height="5" /></>,
  table: <><rect x="2" y="3" width="12" height="10" rx="1" /><path d="M2 7h12M6 7v6" /></>,
  file: <><path d="M4 1.5h5l3 3V14a.5.5 0 0 1-.5.5h-7A.5.5 0 0 1 4 14V2a.5.5 0 0 1 .5-.5z" /><path d="M9 1.5V5h3" /></>,
  folder: <path d="M1.5 3.5h4l1.5 1.5h7v8h-12.5z" />,
  book: <path d="M3 2h4.5a1.5 1.5 0 0 1 1.5 1.5V14a1.5 1.5 0 0 0-1.5-1.5H3zm10 0H8.5A1.5 1.5 0 0 0 7 3.5V14a1.5 1.5 0 0 1 1.5-1.5H13z" />,
  flask: <><path d="M6 1.5v4L2.5 12a1 1 0 0 0 .9 1.5h9.2a1 1 0 0 0 .9-1.5L10 5.5v-4" /><path d="M5 1.5h6" /></>,
  command: <path d="M5 2.5a1.5 1.5 0 1 1 0 3h6a1.5 1.5 0 1 1 0 3M5 5.5h6m-6 0v5a1.5 1.5 0 1 1-1.5-1.5h9a1.5 1.5 0 1 1 0 3" />,
  bolt: <path d="M9 1.5 3.5 9h4L7 14.5 12.5 7h-4z" />,
  refresh: <><path d="M13 7A5 5 0 0 0 3.5 5.5" /><path d="M3 3v3h3" /><path d="M3 9a5 5 0 0 0 9.5 1.5" /><path d="M13 13v-3h-3" /></>,
  dot: <circle cx="8" cy="8" r="2.5" fill="currentColor" />,
  diamond: <path d="M8 1.5 14.5 8 8 14.5 1.5 8z" />,
  sparkle: <path d="M8 1.5 9.3 6.7 14.5 8 9.3 9.3 8 14.5 6.7 9.3 1.5 8 6.7 6.7z" />,
  pin: <path d="M8 1.5v8M5 9.5h6M6.5 13.5 8 9.5l1.5 4z" />,
  panelL: <><rect x="2" y="3" width="12" height="10" rx="1" /><path d="M6 3v10" /></>,
  panelR: <><rect x="2" y="3" width="12" height="10" rx="1" /><path d="M10 3v10" /></>,
  panelB: <><rect x="2" y="3" width="12" height="10" rx="1" /><path d="M2 10h12" /></>,
  panelT: <><rect x="2" y="3" width="12" height="10" rx="1" /><path d="M2 6h12" /></>,
  layers: <><path d="M8 1.5 1.5 5 8 8.5 14.5 5z" /><path d="M1.5 8 8 11.5 14.5 8" /><path d="M1.5 11 8 14.5 14.5 11" /></>,
  chart: <><path d="M2 2v11.5a.5.5 0 0 0 .5.5H14" /><path d="M5 11V8M8 11V5M11 11V7" /></>,
  list: <><path d="M5.5 4H14M5.5 8H14M5.5 12H14" /><path d="M2.5 4h.01M2.5 8h.01M2.5 12h.01" /></>,
  brain: <><path d="M6 2.5a2 2 0 0 0-2 2 2 2 0 0 0-1 3.5 2 2 0 0 0 1 3.5 2 2 0 0 0 4 0V4a1.5 1.5 0 0 0-2-1.5z" /><path d="M10 2.5a2 2 0 0 1 2 2 2 2 0 0 1 1 3.5 2 2 0 0 1-1 3.5 2 2 0 0 1-4 0" /></>,
  info: <><circle cx="8" cy="8" r="6.5" /><path d="M8 7.2v4M8 5.2h.01" /></>,
  warn: <><path d="M8 2 14.5 13.5H1.5z" /><path d="M8 6.5v3.5M8 12h.01" /></>,
}

export function Icon({
  name, size = 14, stroke = 'currentColor', style, className, onClick,
}: {
  name: IconName; size?: number; stroke?: string; style?: CSSProperties; className?: string
  onClick?: () => void
}) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 16 16"
      fill="none" stroke={stroke} strokeWidth={1.4}
      strokeLinecap="round" strokeLinejoin="round"
      className={className} onClick={onClick}
      style={{ display: 'inline-block', flexShrink: 0, ...style }}
    >
      {PATHS[name] ?? null}
    </svg>
  )
}

export function Logo({ size = 18, style }: { size?: number; style?: CSSProperties }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ display: 'inline-block', ...style }}>
      <defs>
        <linearGradient id="lt-g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="var(--lt-g-400)" />
          <stop offset="1" stopColor="var(--lt-g-700)" />
        </linearGradient>
      </defs>
      <path d="M12 1.5 22.5 12 12 22.5 1.5 12z" fill="url(#lt-g)" />
      <path d="M12 6.5 17.5 12 12 17.5 6.5 12z" fill="var(--lt-bg-0)" opacity="0.18" />
    </svg>
  )
}
