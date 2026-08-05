/**
 * Hand-written wire types for the alerting API.
 *
 * There is no proto -> TS codegen for these endpoints: alerting is served by
 * hand-rolled JSON handlers (proto regeneration needs Docker and is the
 * productionization step), so these interfaces mirror the dataclasses in
 * `mlflow/alerts/entities.py` field for field.
 */

import { defineMessages } from 'react-intl';
import type { MessageDescriptor } from 'react-intl';

export type AlertSeverity = 'LOW' | 'MEDIUM' | 'HIGH';
export type AlertComparator = 'GT' | 'GTE' | 'LT' | 'LTE';
export type AlertAggregation = 'COUNT' | 'SUM' | 'AVG' | 'PERCENTILE';
export type AlertDimensionKey = 'TRACES' | 'SPAN_TYPE' | 'SPAN_NAME' | 'SPAN_MODEL' | 'ASSESSMENTS' | 'ERROR';
/**
 * The lifecycle of one firing episode.
 *
 * `INACTIVE` is "fired, has since recovered, nobody has acknowledged it yet". It
 * is still in the active list and still dismissible -- what it preserves is the
 * difference between *a human reviewed this* (`DISMISSED`) and *it stopped on its
 * own*. Only `PENDING` and `FIRED` block the rule from opening the next episode,
 * which is how a second incident becomes visible while the first is still listed.
 */
export type AlertInstanceState = 'PENDING' | 'FIRED' | 'INACTIVE' | 'DISMISSED';

/** Every state that is still waiting on a person -- the active-alerts view. */
export const UNDISMISSED_ALERT_STATES: AlertInstanceState[] = ['PENDING', 'FIRED', 'INACTIVE'];

/** Every state, for the history view. */
export const ALL_ALERT_STATES: AlertInstanceState[] = [...UNDISMISSED_ALERT_STATES, 'DISMISSED'];

export interface AlertRule {
  alert_rule_id: string;
  experiment_id: number;
  name: string;
  metric_key: string;
  dimension_key: AlertDimensionKey;
  aggregation: AlertAggregation;
  comparator: AlertComparator;
  threshold: number;
  window_seconds: number;
  /** Derived server-side from the window; never sent by the client. */
  evaluation_interval_seconds: number;
  dimension_value?: string | null;
  percentile_value?: number | null;
  sustain_seconds: number;
  /**
   * Samples the window must hold before the rule may fire at all.
   *
   * The user's, not the server's. The form seeds it from `deriveMinSampleCount`
   * and the server falls back to the same function when a request omits it, but
   * whatever is sent -- including 0 -- is stored verbatim.
   */
  min_sample_count: number;
  severity: AlertSeverity;
  enabled: boolean;
  channels?: AlertChannel[];
  last_evaluated_ms?: number | null;
  next_evaluation_at_ms?: number | null;
  /** How many samples the last evaluation saw. Drives the "not enough data" status. */
  last_sample_count?: number | null;
  deleted_at_ms?: number | null;
  created_by?: string | null;
  creation_timestamp?: number | null;
  last_updated_timestamp?: number | null;
}

export interface AlertInstance {
  alert_instance_id: string;
  alert_rule_id: string;
  experiment_id: number;
  state: AlertInstanceState;
  started_at_ms: number;
  window_start_ms: number;
  window_end_ms: number;
  observed_value?: number | null;
  /** Worst value seen -- what the notification quotes. */
  peak_value?: number | null;
  /** The threshold as of firing; survives later edits to the rule. */
  threshold?: number | null;
  sample_count: number;
  fired_at_ms?: number | null;
  dismissed_at_ms?: number | null;
  /** A user, or a reserved `system:` actor for rule edits and disables. */
  dismissed_by?: string | null;
  /**
   * Start of the current unbroken run of healthy evaluations, cleared by the next
   * breaching one. For an `INACTIVE` instance this is when it actually recovered.
   */
  healthy_since_ms?: number | null;
  exemplar_trace_ids: string[];
}

export interface CreateAlertRuleRequest {
  experiment_id: string;
  name: string;
  metric_key: string;
  dimension_key: AlertDimensionKey;
  aggregation: AlertAggregation;
  comparator: AlertComparator;
  threshold: number;
  window_seconds: number;
  dimension_value?: string;
  percentile_value?: number;
  sustain_seconds?: number;
  /** Omit to accept the server's suggestion; send 0 to mean "no minimum". */
  min_sample_count?: number;
  severity?: AlertSeverity;
  /**
   * Where the alert goes, beyond the in-app list. Preview.
   *
   * Free-form on the wire: the server stores it as JSON and only resolves a
   * `type` against the channel registry when the alert fires. A type nothing has
   * registered is accepted here and silently fails to deliver, which is why the
   * editor labels this preview rather than offering it as a finished feature.
   */
  channels?: AlertChannel[];
}

/** One notification target. `type` must match a registered channel. */
export interface AlertChannel {
  type: string;
  target?: string;
}

/**
 * The subset of a rule a PATCH may carry, which is deliberately *not*
 * `Partial<CreateAlertRuleRequest>`.
 *
 * `experiment_id` is the trap: a rule cannot move between experiments, and the
 * store rejects any field outside `_ALERT_RULE_UPDATABLE_FIELDS` with a 400, so
 * echoing the create body back would fail the whole patch.
 *
 * `percentile_value` is nullable: the server accepts an explicit null so a rule
 * leaving PERCENTILE can drop the percentile it no longer has.
 *
 * Sending `min_sample_count` in the same patch is what stops the store
 * re-deriving it -- it only fills one in when the caller has not said.
 */
export interface UpdateAlertRuleRequest {
  name?: string;
  metric_key?: string;
  dimension_key?: AlertDimensionKey;
  aggregation?: AlertAggregation;
  comparator?: AlertComparator;
  threshold?: number;
  window_seconds?: number;
  /** Empty string clears the slice; the evaluator reads `dimension_value or ""`. */
  dimension_value?: string;
  percentile_value?: number | null;
  sustain_seconds?: number;
  min_sample_count?: number;
  severity?: AlertSeverity;
  enabled?: boolean;
  /** Send `[]` to clear; the server stores null for an empty list. */
  channels?: AlertChannel[];
}

export type AlertThresholdUnit =
  | 'minutes'
  | 'requests'
  | 'failures'
  | 'usd'
  | 'score'
  | 'tokens'
  | 'verdicts'
  | 'percent';

/** Mirrors `MAX_DIMENSION_VALUE_LENGTH`: the width of the `metric_series` column. */
export const MAX_ALERT_DIMENSION_VALUE_LENGTH = 250;

/**
 * Mirrors `MIN_WINDOW_SECONDS` / `MAX_WINDOW_SECONDS`.
 *
 * The ceiling is raw retention, not a preference: the evaluator expires buckets by
 * re-reading them at one-minute granularity, so a window reaching past retention
 * would read an empty range and silently stop shedding.
 */
export const WINDOW_MIN_SECONDS = 300;
export const WINDOW_MAX_SECONDS = 259_200;

/**
 * The window is entered and displayed as a number plus a unit, never as raw minutes.
 *
 * Three days is 4,320 minutes, which is neither typeable nor readable. The stored
 * value is seconds either way -- the unit exists so the number on screen stays one
 * a person would say out loud.
 */
export const WINDOW_UNIT_SECONDS = { minutes: 60, hours: 3_600, days: 86_400 } as const;

export type WindowUnit = keyof typeof WINDOW_UNIT_SECONDS;

/** The largest unit that divides the value exactly, so 259,200 reads as "3 days". */
export const splitWindow = (windowSeconds: number): { amount: number; unit: WindowUnit } => {
  const units: WindowUnit[] = ['days', 'hours', 'minutes'];
  const unit = units.find((u) => windowSeconds % WINDOW_UNIT_SECONDS[u] === 0) ?? 'minutes';
  return { amount: Math.round(windowSeconds / WINDOW_UNIT_SECONDS[unit]), unit };
};

const metricLabels = defineMessages({
  latency: { defaultMessage: 'Latency', description: 'Alert rule editor: metric option for trace or span latency' },
  total_tokens: { defaultMessage: 'Tokens', description: 'Alert rule editor: metric option for token counts' },
  input_tokens: {
    defaultMessage: 'Input tokens',
    description: 'Alert rule editor: metric option for prompt token counts',
  },
  output_tokens: {
    defaultMessage: 'Output tokens',
    description: 'Alert rule editor: metric option for completion token counts',
  },
  cache_read_input_tokens: {
    defaultMessage: 'Cache-read tokens',
    description: 'Alert rule editor: metric option for prompt tokens served from cache',
  },
  cache_creation_input_tokens: {
    defaultMessage: 'Cache-write tokens',
    description: 'Alert rule editor: metric option for prompt tokens written to cache',
  },
  total_cost: { defaultMessage: 'Cost', description: 'Alert rule editor: metric option for spend' },
  input_cost: { defaultMessage: 'Input cost', description: 'Alert rule editor: metric option for prompt spend' },
  output_cost: {
    defaultMessage: 'Output cost',
    description: 'Alert rule editor: metric option for completion spend',
  },
  error_count: { defaultMessage: 'Errors', description: 'Alert rule editor: metric option for failures' },
  error_rate: {
    defaultMessage: 'Error rate',
    description: 'Alert rule editor: metric option for the share of calls that failed',
  },
  assessment_value: { defaultMessage: 'Judge score', description: 'Alert rule editor: metric option for judge output' },
});

/**
 * How a COUNT of each metric reads as a whole phrase.
 *
 * A whole phrase rather than a noun spliced into "Number of {noun}": the two are
 * not the same sentence in every language, and "Number of judge verdicts" is not
 * a translation of the metric's own label.
 */
const countLabels = defineMessages({
  requests: { defaultMessage: 'Number of requests', description: 'Alert rule summary: a COUNT rule over traces' },
  errors: { defaultMessage: 'Number of errors', description: 'Alert rule summary: a COUNT rule over failed spans' },
  verdicts: {
    defaultMessage: 'Number of judge verdicts',
    description: 'Alert rule summary: a COUNT rule over judge assessments',
  },
});

/**
 * The legal `(metric, dimension, aggregation)` combinations, mirroring
 * `METRIC_CATALOGUE` in `mlflow/alerts/entities.py`.
 *
 * One source of truth for the form's dropdowns and the server's validation, so the
 * form cannot offer a combination the evaluator would reject.
 *
 * A catalogue rather than a list of preset rules: the form previously offered nine
 * fixed combinations, so there was no way to ask for p99 instead of p95, to scope a
 * latency rule to one tool, or to name a judge nobody had anticipated.
 */
export interface AlertMetricSpec {
  label: MessageDescriptor;
  /** What the threshold means, and how it scales into the stored unit. */
  thresholdUnit: AlertThresholdUnit;
  thresholdScale: number;
  /**
   * What a COUNT of this metric counts, when it is not requests. COUNT counts the
   * rows behind the metric, which for most metrics is one per request.
   */
  countUnit?: AlertThresholdUnit;
  countLabel?: MessageDescriptor;
  dimensions: AlertDimensionKey[];
  aggregations: AlertAggregation[];
  /**
   * Dimensions this metric is aggregated *across* rather than *per value of*,
   * mirroring `MetricSpec.unsliceable_dimension_keys`.
   *
   * Whole-trace token counts come from a grouping with no `dim_index`, so every
   * series is written with an empty `dimension_value`. Naming one would produce a
   * rule the server accepts and no series can ever match: it would silently never
   * fire, which is worse than being rejected.
   */
  unsliceableDimensions?: AlertDimensionKey[];
}

export const ALERT_METRIC_CATALOGUE = {
  latency: {
    label: metricLabels.latency,
    thresholdUnit: 'minutes',
    thresholdScale: 60_000,
    dimensions: ['TRACES', 'SPAN_TYPE', 'SPAN_NAME'],
    aggregations: ['PERCENTILE', 'AVG', 'COUNT'],
  },
  total_tokens: {
    label: metricLabels.total_tokens,
    thresholdUnit: 'tokens',
    thresholdScale: 1,
    dimensions: ['TRACES'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
    unsliceableDimensions: ['TRACES'],
  },
  // The components of the total above. Listed next to it rather than at the end:
  // someone reaching for "Tokens" is usually one dropdown away from wanting the
  // input/output split, which is what distinguishes prompt bloat from verbosity.
  input_tokens: {
    label: metricLabels.input_tokens,
    thresholdUnit: 'tokens',
    thresholdScale: 1,
    dimensions: ['TRACES'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
    unsliceableDimensions: ['TRACES'],
  },
  output_tokens: {
    label: metricLabels.output_tokens,
    thresholdUnit: 'tokens',
    thresholdScale: 1,
    dimensions: ['TRACES'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
    unsliceableDimensions: ['TRACES'],
  },
  cache_read_input_tokens: {
    label: metricLabels.cache_read_input_tokens,
    thresholdUnit: 'tokens',
    thresholdScale: 1,
    dimensions: ['TRACES'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
    unsliceableDimensions: ['TRACES'],
  },
  cache_creation_input_tokens: {
    label: metricLabels.cache_creation_input_tokens,
    thresholdUnit: 'tokens',
    thresholdScale: 1,
    dimensions: ['TRACES'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
    unsliceableDimensions: ['TRACES'],
  },
  total_cost: {
    label: metricLabels.total_cost,
    thresholdUnit: 'usd',
    thresholdScale: 1,
    dimensions: ['SPAN_MODEL', 'SPAN_TYPE'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
  },
  input_cost: {
    label: metricLabels.input_cost,
    thresholdUnit: 'usd',
    thresholdScale: 1,
    dimensions: ['SPAN_MODEL', 'SPAN_TYPE'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
  },
  output_cost: {
    label: metricLabels.output_cost,
    thresholdUnit: 'usd',
    thresholdScale: 1,
    dimensions: ['SPAN_MODEL', 'SPAN_TYPE'],
    aggregations: ['SUM', 'AVG', 'PERCENTILE'],
  },
  error_count: {
    label: metricLabels.error_count,
    thresholdUnit: 'failures',
    thresholdScale: 1,
    countUnit: 'failures',
    countLabel: countLabels.errors,
    dimensions: ['ERROR', 'SPAN_NAME'],
    aggregations: ['COUNT'],
  },
  // AVG only, and the scale turns a typed 5 into the stored 0.05. The series holds
  // attempts in `count` and failures in `sum`, so AVG *is* the rate -- SUM and COUNT
  // over it would read back as something else entirely.
  //
  // TRACES is unsliceable: the trace rate is one number for the experiment, not one
  // per status, because the error rate of ERROR traces is 100% by construction.
  error_rate: {
    label: metricLabels.error_rate,
    thresholdUnit: 'percent',
    thresholdScale: 0.01,
    dimensions: ['TRACES', 'SPAN_TYPE', 'SPAN_NAME'],
    aggregations: ['AVG'],
    unsliceableDimensions: ['TRACES'],
  },
  assessment_value: {
    label: metricLabels.assessment_value,
    thresholdUnit: 'score',
    thresholdScale: 1,
    countUnit: 'verdicts',
    countLabel: countLabels.verdicts,
    dimensions: ['ASSESSMENTS'],
    aggregations: ['AVG', 'PERCENTILE', 'COUNT'],
  },
} satisfies Record<string, AlertMetricSpec>;

const dimensionLabels = defineMessages({
  TRACES: { defaultMessage: 'Whole trace', description: 'Alert rule editor: breakdown option for the whole trace' },
  SPAN_TYPE: { defaultMessage: 'Span type', description: 'Alert rule editor: breakdown option for span type' },
  SPAN_NAME: { defaultMessage: 'Tool / span name', description: 'Alert rule editor: breakdown option for span name' },
  SPAN_MODEL: { defaultMessage: 'Model', description: 'Alert rule editor: breakdown option for the model' },
  ASSESSMENTS: { defaultMessage: 'Judge', description: 'Alert rule editor: breakdown option for the judge' },
  ERROR: { defaultMessage: 'Exception type', description: 'Alert rule editor: breakdown option for exception type' },
});

const dimensionValueLabels = defineMessages({
  TRACES: { defaultMessage: 'Trace status', description: 'Alert rule editor: value field label for trace status' },
  SPAN_TYPE: { defaultMessage: 'Span type', description: 'Alert rule editor: value field label for span type' },
  SPAN_NAME: {
    defaultMessage: 'Tool or span name',
    description: 'Alert rule editor: value field label for span name',
  },
  SPAN_MODEL: { defaultMessage: 'Model', description: 'Alert rule editor: value field label for the model' },
  ASSESSMENTS: { defaultMessage: 'Judge', description: 'Alert rule editor: value field label for the judge' },
  ERROR: {
    defaultMessage: 'Exception type',
    description: 'Alert rule editor: value field label for exception type',
  },
});

const dimensionAllLabels = defineMessages({
  TRACES: { defaultMessage: 'Any status', description: 'Alert rule editor: unscoped option for trace status' },
  SPAN_TYPE: { defaultMessage: 'Any span type', description: 'Alert rule editor: unscoped option for span type' },
  SPAN_NAME: { defaultMessage: 'Any tool', description: 'Alert rule editor: unscoped option for span name' },
  SPAN_MODEL: { defaultMessage: 'Any model', description: 'Alert rule editor: unscoped option for the model' },
  ASSESSMENTS: { defaultMessage: 'Any judge', description: 'Alert rule editor: unscoped option for the judge' },
  ERROR: {
    defaultMessage: 'Any exception type',
    description: 'Alert rule editor: unscoped option for exception type',
  },
});

/**
 * How each dimension is described, and how its value gets chosen.
 *
 * `openVocabulary` is the load-bearing flag. Judge names, tool names, model names
 * and exception types are unbounded -- `make_judge()` lets a user name a judge
 * anything -- so those must accept a typed value as well as offering the ones
 * actually observed. Trace status is effectively closed, so suggestions suffice.
 */
export interface AlertDimensionMeta {
  label: MessageDescriptor;
  valueLabel: MessageDescriptor;
  allLabel: MessageDescriptor;
  openVocabulary: boolean;
  /** Offered even before anything has been observed in traffic. */
  suggestions?: string[];
}

export const ALERT_DIMENSION_META: Record<AlertDimensionKey, AlertDimensionMeta> = {
  TRACES: {
    label: dimensionLabels.TRACES,
    valueLabel: dimensionValueLabels.TRACES,
    allLabel: dimensionAllLabels.TRACES,
    openVocabulary: false,
    suggestions: ['OK', 'ERROR'],
  },
  SPAN_TYPE: {
    label: dimensionLabels.SPAN_TYPE,
    valueLabel: dimensionValueLabels.SPAN_TYPE,
    allLabel: dimensionAllLabels.SPAN_TYPE,
    openVocabulary: true,
    suggestions: ['AGENT', 'CHAIN', 'LLM', 'TOOL', 'RETRIEVER', 'PARSER', 'EMBEDDING'],
  },
  SPAN_NAME: {
    label: dimensionLabels.SPAN_NAME,
    valueLabel: dimensionValueLabels.SPAN_NAME,
    allLabel: dimensionAllLabels.SPAN_NAME,
    openVocabulary: true,
  },
  SPAN_MODEL: {
    label: dimensionLabels.SPAN_MODEL,
    valueLabel: dimensionValueLabels.SPAN_MODEL,
    allLabel: dimensionAllLabels.SPAN_MODEL,
    openVocabulary: true,
  },
  ASSESSMENTS: {
    label: dimensionLabels.ASSESSMENTS,
    valueLabel: dimensionValueLabels.ASSESSMENTS,
    allLabel: dimensionAllLabels.ASSESSMENTS,
    openVocabulary: true,
  },
  ERROR: {
    label: dimensionLabels.ERROR,
    valueLabel: dimensionValueLabels.ERROR,
    allLabel: dimensionAllLabels.ERROR,
    openVocabulary: true,
  },
};

export const ALERT_AGGREGATION_LABELS = defineMessages({
  PERCENTILE: { defaultMessage: 'Percentile of', description: 'Alert rule editor: PERCENTILE aggregation option' },
  AVG: { defaultMessage: 'Average', description: 'Alert rule editor: AVG aggregation option' },
  SUM: { defaultMessage: 'Total', description: 'Alert rule editor: SUM aggregation option' },
  COUNT: { defaultMessage: 'Count of', description: 'Alert rule editor: COUNT aggregation option' },
}) satisfies Record<AlertAggregation, MessageDescriptor>;

/** The metric keys the catalogue actually defines. */
export type AlertMetricKey = keyof typeof ALERT_METRIC_CATALOGUE;

export const ALERT_METRIC_KEYS = Object.keys(ALERT_METRIC_CATALOGUE) as AlertMetricKey[];

/**
 * Look up a metric by a key of unknown provenance.
 *
 * Rules arrive over the wire, so a stored rule may name a metric this build does
 * not know -- an older UI against a newer server, or the reverse. Callers get
 * `undefined` and fall back, rather than indexing into a missing entry.
 */
export const metricSpecFor = (metricKey: string): AlertMetricSpec | undefined =>
  (ALERT_METRIC_CATALOGUE as Record<string, AlertMetricSpec>)[metricKey];

export const isAlertMetricKey = (metricKey: string): metricKey is AlertMetricKey =>
  metricSpecFor(metricKey) !== undefined;

/**
 * Whether a rule on this `(metric, dimension)` pair may name a `dimension_value`,
 * mirroring `is_sliceable` in `entities.py`.
 *
 * Sliceability is a property of the pair, not of the dimension: latency is
 * bucketed by trace status, while whole-trace token counts are summed across
 * every status at once.
 */
export const isAlertDimensionSliceable = (metricKey: string, dimensionKey: AlertDimensionKey): boolean => {
  const spec = metricSpecFor(metricKey);
  return spec !== undefined && !(spec.unsliceableDimensions ?? []).includes(dimensionKey);
};

/** The unit a COUNT threshold is expressed in -- rows, never the metric's own unit. */
export const alertCountUnitFor = (metricKey: string): AlertThresholdUnit =>
  metricSpecFor(metricKey)?.countUnit ?? 'requests';

export const alertCountLabelFor = (metricKey: string): MessageDescriptor =>
  metricSpecFor(metricKey)?.countLabel ?? countLabels.requests;

/** One point on the alert detail chart. See `mlflow/alerts/series.py`. */
export interface AlertSeriesPoint {
  timestamp_ms: number;
  /**
   * `null` when that window had a gap or no data.
   *
   * The chart must break the line rather than draw through it: joining across a
   * gap invents a slope nobody measured, and for an absence rule a fabricated
   * dip to zero is the exact shape of a real incident.
   */
  value: number | null;
  sample_count: number;
  is_gap: boolean;
}

export interface AlertSeries {
  points: AlertSeriesPoint[];
  threshold: number;
  window_seconds: number;
  /** Widened from the rule's interval when the range would exceed the point cap. */
  step_seconds: number;
}
