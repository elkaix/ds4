import { measuredNumber, type Telemetry } from './metrics.ts';

export function labelTelemetry(t: Telemetry<number>, format: (v: number) => string): string {
  switch (t.state) {
    case 'missing':
      return 'Not instrumented';
    case 'inactive':
      return 'Inactive';
    case 'zero':
      return format(t.value);
    case 'measured':
      return format(t.value);
  }
}

export function numberTelemetry(value: number | null | undefined): Telemetry<number> {
  return measuredNumber(value);
}

/** Missing field → Not instrumented. Present 0 → "0". Never coerce missing to 0. */
export function displayNumber(
  value: number | null | undefined,
  format: (v: number) => string,
): string {
  return labelTelemetry(numberTelemetry(value), format);
}

export function isInstrumented<T>(value: T | null | undefined): value is T {
  return value != null;
}
