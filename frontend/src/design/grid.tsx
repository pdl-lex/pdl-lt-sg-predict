// AG-Grid-Anbindung: Modulregistrierung (einmalig) + LexoTerm-Theme.
// Wir nutzen die Theming-API (kein Legacy-CSS-Import) und speisen die
// Farben aus unseren CSS-Tokens, damit Light/Dark automatisch über das
// [data-theme]-Attribut mitschalten.
import { AllCommunityModule, ModuleRegistry, themeQuartz, type ColDef } from 'ag-grid-community'

ModuleRegistry.registerModules([AllCommunityModule])

export const ltGridTheme = themeQuartz.withParams({
  accentColor: 'var(--lt-primary)',
  backgroundColor: 'var(--lt-bg-0)',
  foregroundColor: 'var(--lt-fg-1)',
  borderColor: 'var(--lt-line-1)',
  chromeBackgroundColor: 'var(--lt-bg-0)',
  headerBackgroundColor: 'var(--lt-bg-0)',
  headerTextColor: 'var(--lt-fg-1)',
  oddRowBackgroundColor: 'var(--lt-bg-0)',
  rowHoverColor: 'var(--lt-bg-1)',
  selectedRowBackgroundColor: 'var(--lt-primary-soft)',
  inputBackgroundColor: 'var(--lt-bg-1)',
  headerColumnResizeHandleColor: 'var(--lt-line-2)',
  fontFamily: '"Noto Sans", system-ui, sans-serif',
  fontSize: 12.5,
  headerFontSize: 12.5,
  headerFontWeight: 500,
  rowHeight: 36,
  headerHeight: 40,
  cellHorizontalPadding: 12,
  spacing: 6,
  wrapperBorder: false,
  rowBorder: true,
  browserColorScheme: 'inherit',
})

// Gemeinsame Spalten-Defaults: sortierbar, resizbar, Spaltenfilter +
// Floating-Filter (die zusätzlichen Filteroptionen).
export const defaultColDef: ColDef = {
  sortable: true,
  resizable: true,
  filter: true,
  floatingFilter: true,
  minWidth: 72,
}
