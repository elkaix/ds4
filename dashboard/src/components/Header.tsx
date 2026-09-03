import type { Connection } from '../hooks/useStats';
import { fmtDuration } from '../lib/format';

interface Props {
  connection: Connection;
  busy: boolean;
  modelIds: readonly string[];
  modelName: string | null;
  uptimeS: number | undefined;
  base: string;
}

export function Header({ connection, busy, modelIds, modelName, uptimeS, base }: Props) {
  const status =
    connection.state === 'error' ? { cls: 'down', text: 'Disconnected' } :
    connection.state === 'connecting' ? { cls: 'wait', text: 'Connecting' } :
    busy ? { cls: 'busy', text: 'Generating' } : { cls: 'up', text: 'Idle' };

  return (
    <header className="top">
      <div className="top__brand">
        <span className="top__logo" aria-hidden="true">ds4</span>
        <h1>ds4-server</h1>
        <span className={`pill pill--${status.cls}`} role="status" aria-live="polite">
          <i aria-hidden="true" />
          {status.text}
        </span>
      </div>
      <div className="top__meta">
        {modelName && <span className="top__model" title={modelIds.join(', ')}>{modelName}</span>}
        {uptimeS != null && (
          <span>
            <span className="tone-muted">up </span>
            {fmtDuration(uptimeS)}
          </span>
        )}
        <code className="top__url">{base}</code>
      </div>
    </header>
  );
}
