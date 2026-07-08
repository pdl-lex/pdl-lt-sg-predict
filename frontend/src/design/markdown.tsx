// Minimaler, abhängigkeitsfreier Markdown-Renderer für die Anleitung.
// Unterstützt: #/##/### Überschriften, Absätze, Listen (-, *, 1.), **fett**,
// `Code`, [Links](url), Pipe-Tabellen und --- Trennlinien.
import type { JSX, ReactNode } from 'react'

function inline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = []
  // Reihenfolge: Code, dann Fett, dann Links — via kombinierter Regex.
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)]+\))/g
  let last = 0
  let m: RegExpExecArray | null
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const tok = m[0]
    const key = `${keyBase}-${i++}`
    if (tok.startsWith('`')) {
      out.push(<code key={key}>{tok.slice(1, -1)}</code>)
    } else if (tok.startsWith('**')) {
      out.push(<strong key={key}>{tok.slice(2, -2)}</strong>)
    } else {
      const lm = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok)
      if (lm) out.push(<a key={key} href={lm[2]} target="_blank" rel="noreferrer">{lm[1]}</a>)
    }
    last = re.lastIndex
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

function splitRow(line: string): string[] {
  return line.replace(/^\||\|$/g, '').split('|').map((c) => c.trim())
}

export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: JSX.Element[] = []
  let i = 0
  let k = 0

  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()

    if (!trimmed) { i++; continue }

    // Überschriften
    const h = /^(#{1,3})\s+(.*)$/.exec(trimmed)
    if (h) {
      const level = h[1].length
      const content = inline(h[2], `h${k}`)
      if (level === 1) blocks.push(<h1 key={k++}>{content}</h1>)
      else if (level === 2) blocks.push(<h2 key={k++}>{content}</h2>)
      else blocks.push(<h3 key={k++}>{content}</h3>)
      i++; continue
    }

    // Trennlinie
    if (/^---+$/.test(trimmed)) { blocks.push(<hr key={k++} />); i++; continue }

    // Tabelle (mindestens Kopf + Trennzeile)
    if (trimmed.startsWith('|') && i + 1 < lines.length && /^\|?[\s:|-]+\|?$/.test(lines[i + 1].trim())) {
      const header = splitRow(trimmed)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(splitRow(lines[i].trim())); i++
      }
      blocks.push(
        <table key={k++}>
          <thead><tr>{header.map((c, ci) => <th key={ci}>{inline(c, `th${k}-${ci}`)}</th>)}</tr></thead>
          <tbody>{rows.map((r, ri) => (
            <tr key={ri}>{r.map((c, ci) => <td key={ci}>{inline(c, `td${k}-${ri}-${ci}`)}</td>)}</tr>
          ))}</tbody>
        </table>,
      )
      continue
    }

    // Listen
    const isUl = /^[-*]\s+/.test(trimmed)
    const isOl = /^\d+\.\s+/.test(trimmed)
    if (isUl || isOl) {
      const items: ReactNode[] = []
      while (i < lines.length && (/^[-*]\s+/.test(lines[i].trim()) || /^\d+\.\s+/.test(lines[i].trim()))) {
        const t = lines[i].trim().replace(/^([-*]|\d+\.)\s+/, '')
        items.push(<li key={items.length}>{inline(t, `li${k}-${items.length}`)}</li>)
        i++
      }
      blocks.push(isOl ? <ol key={k++}>{items}</ol> : <ul key={k++}>{items}</ul>)
      continue
    }

    // Absatz (aufeinanderfolgende Textzeilen zusammenfassen)
    const para: string[] = []
    while (i < lines.length && lines[i].trim() && !/^(#{1,3}\s|[-*]\s|\d+\.\s|\||---+$)/.test(lines[i].trim())) {
      para.push(lines[i].trim()); i++
    }
    blocks.push(<p key={k++}>{inline(para.join(' '), `p${k}`)}</p>)
  }

  return <div className="md-body">{blocks}</div>
}
