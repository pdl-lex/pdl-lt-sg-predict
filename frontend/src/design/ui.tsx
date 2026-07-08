// Wiederverwendbare UI-Bausteine im LexoTerm-Werkzeug-Look.
import { useCallback, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { AgGridReact } from 'ag-grid-react'
import type { ColDef, RowSelectionOptions, SelectionChangedEvent } from 'ag-grid-community'
import { Icon, type IconName } from './icons'
import { Kbd, kc } from './widgets'
import { defaultColDef, ltGridTheme } from './grid'

// ── Style-Konstanten ────────────────────────────────────────────────────────
export const inputStyle: CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '6px 9px', fontSize: 12,
  background: 'var(--lt-bg-1)', border: '1px solid var(--lt-line-1)',
  borderRadius: 'var(--lt-r-sm)', color: 'var(--lt-fg-1)', outline: 'none',
}
export const labelStyle: CSSProperties = {
  fontSize: 11, fontWeight: 600, color: 'var(--lt-fg-2)', marginBottom: 5, display: 'block',
}
export const cardStyle: CSSProperties = {
  background: 'var(--lt-bg-0)', border: '1px solid var(--lt-line-1)',
  borderRadius: 'var(--lt-r-md)', boxShadow: 'var(--lt-shadow-1)', padding: 12,
}

// ── Buttons ─────────────────────────────────────────────────────────────────
export function PrimaryButton({
  children, onClick, disabled, icon, kbd, style, full,
}: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; icon?: IconName
  kbd?: string; style?: CSSProperties; full?: boolean
}) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
      background: 'var(--lt-primary)', color: 'var(--lt-on-primary)', border: '1px solid var(--lt-primary)',
      height: 36, padding: '0 16px', borderRadius: 'var(--lt-r-md)', fontSize: 13, fontWeight: 600,
      cursor: disabled ? 'default' : 'pointer', opacity: disabled ? 0.55 : 1, width: full ? '100%' : undefined,
      ...style,
    }}>
      {icon && <Icon name={icon} size={12} />}
      {children}
      {kbd && <span style={{ opacity: 0.65, fontSize: 11, marginLeft: 2, fontFamily: 'var(--lt-font-mono)' }}>{kbd}</span>}
    </button>
  )
}

export function GhostButton({
  children, onClick, disabled, icon, active, style, title,
}: {
  children?: ReactNode; onClick?: () => void; disabled?: boolean; icon?: IconName
  active?: boolean; style?: CSSProperties; title?: string
}) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} style={{
      display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12, padding: '5px 10px',
      background: active ? 'var(--lt-primary-soft)' : 'var(--lt-bg-1)',
      border: '1px solid ' + (active ? 'var(--lt-primary-line)' : 'var(--lt-line-1)'),
      borderRadius: 'var(--lt-r-sm)', color: active ? 'var(--lt-primary)' : 'var(--lt-fg-2)',
      cursor: disabled ? 'default' : 'pointer', opacity: disabled ? 0.55 : 1, ...style,
    }}>
      {icon && <Icon name={icon} size={13} />}
      {children}
    </button>
  )
}

// ── Formularfelder ──────────────────────────────────────────────────────────
export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <label style={labelStyle}>{label}</label>
      {children}
      {hint && <div style={{ fontSize: 11, color: 'var(--lt-fg-3)', marginTop: 4, lineHeight: 1.4 }}>{hint}</div>}
    </div>
  )
}

export function TextInput({
  value, onChange, placeholder, mono, onEnter,
}: {
  value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean; onEnter?: () => void
}) {
  return (
    <input value={value} placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onEnter ? (e) => { if (e.key === 'Enter') onEnter() } : undefined}
      style={{ ...inputStyle, fontFamily: mono ? 'var(--lt-font-mono)' : 'inherit' }} />
  )
}

export function Select({
  value, onChange, options,
}: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

export function Toggle({
  checked, onChange, label,
}: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 9, cursor: 'pointer' }}>
      <span onClick={() => onChange(!checked)} style={{
        width: 32, height: 18, borderRadius: 9, position: 'relative', flexShrink: 0,
        background: checked ? 'var(--lt-primary)' : 'var(--lt-bg-3)',
        border: '1px solid ' + (checked ? 'var(--lt-primary)' : 'var(--lt-line-2)'), transition: 'background .12s',
      }}>
        <span style={{
          position: 'absolute', top: 1, left: checked ? 15 : 1, width: 14, height: 14, borderRadius: 7,
          background: '#fff', boxShadow: 'var(--lt-shadow-1)', transition: 'left .12s',
        }} />
      </span>
      {label && <span style={{ fontSize: 12, color: 'var(--lt-fg-2)' }}>{label}</span>}
    </label>
  )
}

export function Checkbox({
  checked, onChange, label,
}: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 12 }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: 'var(--lt-primary)' }} />
      <span>{label}</span>
    </label>
  )
}

// ── Karten / Sektionen ──────────────────────────────────────────────────────
export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ ...cardStyle, ...style }}>{children}</div>
}

export function SectionFold({
  title, badge, icon, open, onToggle, children,
}: {
  title: string; badge?: ReactNode; icon?: IconName; open: boolean; onToggle: () => void; children: ReactNode
}) {
  return (
    <div style={cardStyle}>
      <button onClick={onToggle} style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 8, background: 'transparent',
        border: 'none', cursor: 'pointer', padding: 0, color: 'var(--lt-fg-1)',
      }}>
        <Icon name={open ? 'chevDown' : 'chevron'} size={11} style={{ color: 'var(--lt-fg-3)' }} />
        {icon && <Icon name={icon} size={13} style={{ color: 'var(--lt-fg-3)' }} />}
        <span style={{ fontSize: 12.5, fontWeight: 600, flex: 1, textAlign: 'left' }}>{title}</span>
        {badge != null && <MonoBadge>{badge}</MonoBadge>}
      </button>
      {open && <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>{children}</div>}
    </div>
  )
}

export function MonoBadge({ children }: { children: ReactNode }) {
  return (
    <span style={{
      fontSize: 10, fontFamily: 'var(--lt-font-mono)', color: 'var(--lt-fg-3)',
      background: 'var(--lt-bg-2)', padding: '1px 6px', borderRadius: 3,
    }}>{children}</span>
  )
}

type Tone = 'primary' | 'info' | 'warn' | 'err' | 'neutral'
const TONE_COLORS: Record<Tone, { fg: string; bg: string; line: string }> = {
  primary: { fg: 'var(--lt-primary)', bg: 'var(--lt-primary-soft)', line: 'var(--lt-primary-line)' },
  info: { fg: 'var(--lt-info)', bg: 'var(--lt-info-soft)', line: 'var(--lt-info-line)' },
  warn: { fg: 'var(--lt-warn)', bg: 'var(--lt-warn-soft)', line: 'var(--lt-warn-line)' },
  err: { fg: 'var(--lt-err)', bg: 'var(--lt-err-soft)', line: 'var(--lt-err-line)' },
  neutral: { fg: 'var(--lt-fg-2)', bg: 'var(--lt-bg-2)', line: 'var(--lt-line-1)' },
}

export function Badge({ children, tone = 'primary' }: { children: ReactNode; tone?: Tone }) {
  const c = TONE_COLORS[tone]
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600, padding: '1px 8px',
      borderRadius: 999, color: c.fg, background: c.bg, border: `1px solid ${c.line}`,
    }}>{children}</span>
  )
}

export function Callout({ tone = 'info', icon = 'info', children }: { tone?: Tone; icon?: IconName; children: ReactNode }) {
  const c = TONE_COLORS[tone]
  return (
    <div style={{
      display: 'flex', gap: 10, padding: '10px 12px', borderRadius: 'var(--lt-r-md)',
      background: c.bg, border: `1px solid ${c.line}`, color: 'var(--lt-fg-2)', fontSize: 12.5, lineHeight: 1.5,
    }}>
      <Icon name={icon} size={15} style={{ color: c.fg, flexShrink: 0, marginTop: 1 }} />
      <div style={{ minWidth: 0 }}>{children}</div>
    </div>
  )
}

// ── Ergebnis-Tabelle: Toolbar + AG-Grid + Footer ─────────────────────────────
// AG Grid liefert Sortierung, Spaltenfilter (inkl. Floating-Filter), Resizing,
// Pagination, Checkbox-Auswahl und CSV-Export. Toolbar (Titel/Meta/Schnellfilter)
// und Footer (CSV + auswahlabhängige Aktionen) bleiben im LexoTerm-Look; die
// Aktionen liegen bewusst UNTER der Tabelle (Links→Rechts-Workflow).
export interface Column {
  key: string
  label: string
  width?: number
  align?: 'right'
  mono?: boolean
  danger?: boolean
  italic?: boolean
  chip?: boolean
  primary?: boolean
}

const PAGE_SIZES = [25, 50, 100, 200]

const HTML_ESC: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;' }
const escHtml = (s: string) => s.replace(/[&<>]/g, (c) => HTML_ESC[c])

function cellClass(col: Column): string | undefined {
  const cls: string[] = []
  if (col.mono || col.chip) cls.push('lt-cell-mono')
  if (col.danger) cls.push('lt-cell-danger')
  if (col.primary) cls.push('lt-cell-primary')
  if (col.italic) cls.push('lt-cell-italic')
  return cls.length ? cls.join(' ') : undefined
}

function toColDefs(columns: Column[]): ColDef[] {
  return columns.map((c) => ({
    field: c.key,
    headerName: c.label,
    ...(c.width ? { width: c.width } : { flex: 1, minWidth: 120 }),
    ...(c.align === 'right' ? { type: 'rightAligned' } : {}),
    cellClass: cellClass(c),
  }))
}

export function DataTable({
  title, meta, columns, rows, csvName, emptyHint = 'Keine Einträge.', rowKey, selectable, footerActions,
}: {
  title: string
  meta?: ReactNode
  columns: Column[]
  rows: Record<string, unknown>[]
  csvName: string
  emptyHint?: string
  rowKey?: (r: Record<string, unknown>) => string
  /** Checkbox-Auswahlspalte einblenden (Mehrfachauswahl). */
  selectable?: boolean
  /** Aktionen unter der Tabelle, abhängig von der aktuellen Zeilenauswahl. */
  footerActions?: (selected: Record<string, unknown>[]) => ReactNode
}) {
  const gridRef = useRef<AgGridReact>(null)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<Record<string, unknown>[]>([])

  const colDefs = useMemo(() => toColDefs(columns), [columns])
  const getRowId = useMemo(
    () => (rowKey ? (p: { data: Record<string, unknown> }) => rowKey(p.data) : undefined),
    [rowKey],
  )
  const rowSelection = useMemo<RowSelectionOptions | undefined>(
    () => (selectable
      ? { mode: 'multiRow', checkboxes: true, headerCheckbox: true, enableClickSelection: false }
      : undefined),
    [selectable],
  )
  const noRowsTemplate = useMemo(
    () => `<span style="color:var(--lt-fg-3);font-size:12.5px">${escHtml(emptyHint)}</span>`,
    [emptyHint],
  )

  const onSelectionChanged = useCallback((e: SelectionChangedEvent) => {
    setSelected(e.api.getSelectedRows() as Record<string, unknown>[])
  }, [])

  const download = useCallback(() => {
    gridRef.current?.api.exportDataAsCsv({ fileName: csvName, columnSeparator: ';' })
  }, [csvName])

  return (
    <main style={{ gridArea: 'main', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Toolbar */}
      <div style={{
        height: 44, flexShrink: 0, padding: '0 16px', display: 'flex', alignItems: 'center', gap: 12,
        borderBottom: '1px solid var(--lt-line-1)', background: 'var(--lt-bg-0)',
      }}>
        <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>{title}</h3>
        {meta && <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, fontFamily: 'var(--lt-font-mono)', fontSize: 12, color: 'var(--lt-fg-3)' }}>{meta}</div>}
        <span style={{ flex: 1 }} />
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', background: 'var(--lt-bg-1)',
          border: '1px solid var(--lt-line-1)', borderRadius: 'var(--lt-r-sm)', fontSize: 12, color: 'var(--lt-fg-3)', width: 220,
        }}>
          <Icon name="filter" size={11} style={{ flexShrink: 0 }} />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Schnellfilter…"
            style={{ flex: 1, minWidth: 0, border: 'none', outline: 'none', background: 'transparent', fontSize: 12, color: 'var(--lt-fg-1)' }} />
          <Kbd>{kc('F')}</Kbd>
        </div>
      </div>

      {/* AG Grid */}
      <div className="agm-grid" style={{ flex: 1, minHeight: 0, overflow: 'hidden', background: 'var(--lt-bg-0)' }}>
        <AgGridReact
          ref={gridRef}
          theme={ltGridTheme}
          rowData={rows}
          columnDefs={colDefs}
          defaultColDef={defaultColDef}
          getRowId={getRowId}
          rowSelection={rowSelection}
          onSelectionChanged={selectable ? onSelectionChanged : undefined}
          quickFilterText={query}
          pagination
          paginationPageSize={50}
          paginationPageSizeSelector={PAGE_SIZES}
          suppressCellFocus
          animateRows={false}
          overlayNoRowsTemplate={noRowsTemplate}
          containerStyle={{ height: '100%' }}
        />
      </div>

      {/* Footer: Aktionen unter der Tabelle */}
      <div style={{
        minHeight: 48, flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10, padding: '7px 16px',
        borderTop: '1px solid var(--lt-line-1)', background: 'var(--lt-bg-0)', fontSize: 12, color: 'var(--lt-fg-2)',
      }}>
        <PrimaryButton onClick={download} disabled={rows.length === 0} icon="download" style={{ height: 34 }}>CSV herunterladen</PrimaryButton>
        {footerActions?.(selected)}
        <span style={{ flex: 1 }} />
        {selectable && (
          <span style={{ fontFamily: 'var(--lt-font-mono)', color: 'var(--lt-fg-3)' }}>
            {selected.length} ausgewählt
          </span>
        )}
      </div>
    </main>
  )
}

// ── Ergebnis-Rahmen für Nicht-Tabellen-Module (Einzelvorhersage/Training) ────
export function ResultsFrame({ title, meta, children }: { title: string; meta?: ReactNode; children: ReactNode }) {
  return (
    <main style={{ gridArea: 'main', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{
        height: 44, flexShrink: 0, padding: '0 16px', display: 'flex', alignItems: 'center', gap: 12,
        borderBottom: '1px solid var(--lt-line-1)', background: 'var(--lt-bg-0)',
      }}>
        <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>{title}</h3>
        {meta && <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, fontFamily: 'var(--lt-font-mono)', fontSize: 12, color: 'var(--lt-fg-3)' }}>{meta}</div>}
      </div>
      <div className="agm-grid" style={{ flex: 1, overflow: 'auto', background: 'var(--lt-bg-1)', padding: 20 }}>
        {children}
      </div>
    </main>
  )
}
