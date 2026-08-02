import { useRef, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CodeBlock } from './CodeBlock';
import { FloatingCopyButton } from '../ui/FloatingCopyButton';
import 'highlight.js/styles/atom-one-dark.css';

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose-msg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          // `pre` wraps the highlighted code — we pull its inner text out
          // and hand it to our minimal CodeBlock for the copy button etc.
          pre({ children }) {
            const firstChild = Array.isArray(children) ? children[0] : children;
            if (firstChild && typeof firstChild === 'object' && 'props' in firstChild) {
              const props = (firstChild as { props: { className?: string; children?: unknown } })
                .props;
              const lang = /language-(\w+)/.exec(props.className || '')?.[1];
              const text =
                typeof props.children === 'string'
                  ? props.children
                  : Array.isArray(props.children)
                    ? props.children.filter((c) => typeof c === 'string').join('')
                    : '';
              return (
                <CodeBlock language={lang}>{text.replace(/\n$/, '')}</CodeBlock>
              );
            }
            return <pre>{children}</pre>;
          },

          // Wrap tables so they get a floating copy button that exports
          // the table as tab-separated values (paste into Sheets/Excel).
          table({ children }) {
            return <TableWithCopy>{children}</TableWithCopy>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function TableWithCopy({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);

  const getTsv = () => {
    const table = ref.current?.querySelector('table');
    if (!table) return '';
    return Array.from(table.rows)
      .map((r) =>
        Array.from(r.cells)
          .map((c) => c.innerText.replace(/\t/g, ' ').trim())
          .join('\t'),
      )
      .join('\n');
  };

  // Scroll container holds the table at its natural width so wide tables
  // get a horizontal scrollbar instead of stretching the layout; the
  // wrapper itself fills 100% of the message column.
  return (
    <div
      ref={ref}
      className="group relative my-4 overflow-auto rounded-card border"
      style={{ borderColor: 'var(--gray-200)' }}
    >
      <FloatingCopyButton text={getTsv} tone="light" title="Copy table as TSV" />
      {children}
    </div>
  );
}
