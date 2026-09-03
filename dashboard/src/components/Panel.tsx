import type { ReactNode } from 'react';

interface Props {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Panel({ title, aside, children, className }: Props) {
  return (
    <section className={`tile ${className ?? ''}`} aria-label={title}>
      <header className="tile__head">
        <h2 className="tile__label">{title}</h2>
        {aside && <span className="tile__aside">{aside}</span>}
      </header>
      {children}
    </section>
  );
}
