import { defineMessages } from 'react-intl';
import type { IntlShape, MessageDescriptor } from 'react-intl';

import type { AlertInstance, AlertInstanceState, AlertRule, AlertThresholdUnit, WindowUnit } from './types';
import { alertCountLabelFor, alertCountUnitFor, metricSpecFor, splitWindow } from './types';

/**
 * A rule's status is a *join*, not a stored column: it comes from the rule's
 * undismissed instances plus the sample count of its last evaluation.
 *
 * `ALERTING` means **firing and unacknowledged**. `RECOVERED` is the other half
 * of that distinction: the episode is over, but nobody has said so, and the row
 * is still on screen until someone does.
 *
 * `NO_DATA` and `BELOW_MINIMUM` are deliberately separate. Both mean the rule
 * cannot currently fire, but only one of them is the user's own setting holding
 * it back, and only that one is fixable by editing the rule -- reporting them as
 * a single status left "structurally cannot fire" indistinguishable from "no
 * traffic yet", with nothing on screen naming the number responsible.
 */
export type AlertRuleStatus =
  | 'ALERTING'
  | 'CONFIRMING'
  | 'RECOVERED'
  | 'NORMAL'
  | 'NO_DATA'
  | 'BELOW_MINIMUM'
  | 'DISABLED'
  | 'CLOSED';

/**
 * How one instance reads as a status, for the active-alerts list.
 *
 * The list shows episodes, not rules, so it needs the per-instance answer rather
 * than the join below.
 */
export const getAlertInstanceStatus = (state: AlertInstanceState): AlertRuleStatus => {
  switch (state) {
    case 'FIRED':
      return 'ALERTING';
    case 'INACTIVE':
      return 'RECOVERED';
    case 'DISMISSED':
      // Acknowledged and over. Only reachable from the history list -- the rule
      // join filters dismissed instances out before it gets here -- but it must
      // be named, or a closed episode reads as one still waiting to confirm.
      return 'CLOSED';
    default:
      return 'CONFIRMING';
  }
};

export const getAlertRuleStatus = (rule: AlertRule, instances: AlertInstance[]): AlertRuleStatus => {
  if (!rule.enabled) {
    return 'DISABLED';
  }
  const open = instances.filter(
    (instance) => instance.alert_rule_id === rule.alert_rule_id && instance.state !== 'DISMISSED',
  );
  // Most urgent wins. A rule with a recovered episode still waiting to be
  // acknowledged *and* a new one firing is alerting, not recovered.
  if (open.some((instance) => instance.state === 'FIRED')) {
    return 'ALERTING';
  }
  if (open.some((instance) => instance.state === 'PENDING')) {
    return 'CONFIRMING';
  }
  if (open.some((instance) => instance.state === 'INACTIVE')) {
    return 'RECOVERED';
  }
  const samples = rule.last_sample_count;
  if (samples === null || samples === undefined) {
    // Never evaluated, so there is no sample count to judge -- not the same as
    // having been evaluated and found empty.
    return 'NO_DATA';
  }
  if (samples < (rule.min_sample_count ?? 0)) {
    return 'BELOW_MINIMUM';
  }
  return 'NORMAL';
};

const sampleShortfall = defineMessages({
  belowMinimum: {
    defaultMessage: '{samples} of {minimum} samples',
    description: 'Alert rules table: the rule saw fewer samples than its own minimum, so it cannot fire',
  },
});

/**
 * Why a rule is held back, in the numbers responsible.
 *
 * `null` when nothing is holding it back. The minimum used to be derived and
 * never shown, so a rule sat permanently unfireable with no way to find out
 * which number to change.
 */
export const describeSampleShortfall = (rule: AlertRule, intl: IntlShape): string | null => {
  const minimum = rule.min_sample_count ?? 0;
  const samples = rule.last_sample_count;
  if (!minimum || samples === null || samples === undefined || samples >= minimum) {
    return null;
  }
  return intl.formatMessage(sampleShortfall.belowMinimum, {
    samples: String(samples),
    minimum: String(minimum),
  });
};

/**
 * The threshold with its unit attached.
 *
 * A whole phrase per unit rather than a suffix appended to a number: the unit
 * does not always trail the value (`$5`), and in other languages it trails
 * differently.
 */
const thresholdFormats = defineMessages({
  minutes: { defaultMessage: '{value} min', description: 'Alert rule summary: a latency threshold in minutes' },
  requests: { defaultMessage: '{value} requests', description: 'Alert rule summary: a threshold counting requests' },
  failures: { defaultMessage: '{value} failures', description: 'Alert rule summary: a threshold counting failures' },
  verdicts: {
    defaultMessage: '{value} verdicts',
    description: 'Alert rule summary: a threshold counting judge verdicts',
  },
  // eslint-disable-next-line no-template-curly-in-string -- an ICU placeholder behind a currency symbol, not a template literal
  usd: { defaultMessage: '${value}', description: 'Alert rule summary: a spend threshold in US dollars' },
  score: { defaultMessage: '{value}', description: 'Alert rule summary: a unitless judge score threshold' },
  percent: { defaultMessage: '{value}%', description: 'Alert rule summary: an error-rate threshold' },
  tokens: { defaultMessage: '{value} tokens', description: 'Alert rule summary: a token count threshold' },
}) satisfies Record<AlertThresholdUnit, MessageDescriptor>;

const metricDescriptions = defineMessages({
  percentile: {
    defaultMessage: '{percentile}th percentile {metric}',
    description: 'Alert rule summary: a PERCENTILE rule, e.g. "95th percentile latency"',
  },
  average: { defaultMessage: 'Average {metric}', description: 'Alert rule summary: an AVG rule' },
  total: { defaultMessage: 'Total {metric}', description: 'Alert rule summary: a SUM rule' },
  other: {
    defaultMessage: '{metric} ({aggregation})',
    description: 'Alert rule summary: fallback for an aggregation this build does not know',
  },
});

const scopeDescriptions = defineMessages({
  allTraces: {
    defaultMessage: 'all traces in this experiment',
    description: 'Alert rule scope: the rule is not narrowed to any slice',
  },
  judge: { defaultMessage: 'judge "{value}"', description: 'Alert rule scope: narrowed to one judge' },
  tool: { defaultMessage: 'tool "{value}"', description: 'Alert rule scope: narrowed to one tool or span name' },
  model: { defaultMessage: 'model "{value}"', description: 'Alert rule scope: narrowed to one model' },
  exception: {
    defaultMessage: 'exception type "{value}"',
    description: 'Alert rule scope: narrowed to one exception type',
  },
  spanType: { defaultMessage: 'spans of type "{value}"', description: 'Alert rule scope: narrowed to one span type' },
  traceStatus: {
    defaultMessage: 'traces with status "{value}"',
    description: 'Alert rule scope: narrowed to one trace status',
  },
});

/**
 * Also used as the unit dropdown's labels, so the editor and the summary line can
 * never disagree about what to call a unit.
 */
export const windowUnitLabels = defineMessages({
  minutes: {
    defaultMessage: '{amount, plural, one {# minute} other {# minutes}}',
    description: 'A window length in minutes',
  },
  hours: { defaultMessage: '{amount, plural, one {# hour} other {# hours}}', description: 'A window length in hours' },
  days: { defaultMessage: '{amount, plural, one {# day} other {# days}}', description: 'A window length in days' },
}) satisfies Record<WindowUnit, MessageDescriptor>;

/** "3 days", not "4320 min". */
export const describeAlertWindow = (windowSeconds: number, intl: IntlShape): string => {
  const { amount, unit } = splitWindow(windowSeconds);
  return intl.formatMessage(windowUnitLabels[unit], { amount });
};

const ruleDescription = defineMessages({
  line: {
    defaultMessage: '{metric}{scope} {comparator} {threshold} over {window}',
    description: 'Alert rules table: one line describing what a rule watches',
  },
});

const COMPARATOR_SYMBOL: Record<AlertRule['comparator'], string> = {
  GT: '>',
  GTE: '>=',
  LT: '<',
  LTE: '<=',
};

/**
 * The threshold in the unit the user typed it in.
 *
 * A COUNT rule counts *rows*, so it is never scaled -- a "fewer than 5 requests"
 * rule must not have its 5 divided by 60,000 just because the metric it counts is
 * a latency. The noun comes from the metric, because a COUNT of judge output is
 * verdicts and a COUNT of errors is failures.
 */
export const formatAlertThreshold = (rule: AlertRule, intl: IntlShape): string => {
  const spec = metricSpecFor(rule.metric_key);
  if (!spec) {
    return String(rule.threshold);
  }
  if (rule.aggregation === 'COUNT') {
    return intl.formatMessage(thresholdFormats[alertCountUnitFor(rule.metric_key)], {
      value: String(rule.threshold),
    });
  }
  // Passed as a string so ICU renders the number the user typed rather than
  // regrouping it -- a 2,700,000 token budget is stored, and read back, plain.
  return intl.formatMessage(thresholdFormats[spec.thresholdUnit], {
    value: String(rule.threshold / spec.thresholdScale),
  });
};

/**
 * An *observed* value in the same unit as the rule's threshold.
 *
 * Rules and readings have to be quoted the same way or the pair is unreadable:
 * a peak was being rendered raw as `4080906.815974849` beside a threshold that
 * read "> 30 min", which is the same quantity written two different ways and
 * leaves the reader to divide by 60,000.
 *
 * Shares `formatAlertThreshold`'s scaling rules, including the one that matters:
 * COUNT counts rows, so its readings are never scaled either.
 */
export const formatAlertValue = (rule: AlertRule, value: number | null | undefined, intl: IntlShape): string => {
  if (value === null || value === undefined) {
    return '—';
  }
  const spec = metricSpecFor(rule.metric_key);
  if (!spec) {
    return String(value);
  }
  const unit = rule.aggregation === 'COUNT' ? alertCountUnitFor(rule.metric_key) : spec.thresholdUnit;
  const scale = rule.aggregation === 'COUNT' ? 1 : spec.thresholdScale;
  // Rounded before display: a p95 read off a sketch carries far more digits than
  // it has accuracy -- the sketch is 2% relative -- so printing them all claims a
  // precision the number does not have.
  const scaled = value / scale;
  const rounded = Math.abs(scaled) >= 100 ? Math.round(scaled) : Number(scaled.toFixed(2));
  return intl.formatMessage(thresholdFormats[unit], { value: String(rounded) });
};

/** "95th percentile latency", "average judge score", "number of requests". */
export const describeAlertMetric = (rule: AlertRule, intl: IntlShape): string => {
  const spec = metricSpecFor(rule.metric_key);
  const metric = spec ? intl.formatMessage(spec.label).toLowerCase() : rule.metric_key;
  switch (rule.aggregation) {
    case 'PERCENTILE':
      return intl.formatMessage(metricDescriptions.percentile, {
        percentile: String(rule.percentile_value ?? 95),
        metric,
      });
    case 'AVG':
      return intl.formatMessage(metricDescriptions.average, { metric });
    case 'SUM':
      return intl.formatMessage(metricDescriptions.total, { metric });
    case 'COUNT':
      return intl.formatMessage(alertCountLabelFor(rule.metric_key));
    default:
      return intl.formatMessage(metricDescriptions.other, {
        metric: spec ? intl.formatMessage(spec.label) : rule.metric_key,
        aggregation: rule.aggregation,
      });
  }
};

/**
 * One line describing what the rule watches, including its slice.
 *
 * The scope is always rendered: a filter hidden inside a rule is worse than a
 * prescriptive dropdown, because nothing on screen says what the number covers.
 */
export const describeAlertRule = (rule: AlertRule, intl: IntlShape): string =>
  intl.formatMessage(ruleDescription.line, {
    metric: describeAlertMetric(rule, intl),
    scope: rule.dimension_value ? ` [${rule.dimension_value}]` : '',
    comparator: COMPARATOR_SYMBOL[rule.comparator],
    threshold: formatAlertThreshold(rule, intl),
    window: describeAlertWindow(rule.window_seconds, intl),
  });

/**
 * Which slice of the experiment a rule measures, in words.
 *
 * `TRACES` with no value is every trace in the experiment; every other
 * dimension names the member it is scoped to.
 */
export const describeAlertScope = (
  dimensionKey: string,
  dimensionValue: string | null | undefined,
  intl: IntlShape,
) => {
  if (!dimensionValue) {
    return intl.formatMessage(scopeDescriptions.allTraces);
  }
  const values = { value: dimensionValue };
  switch (dimensionKey) {
    case 'ASSESSMENTS':
      return intl.formatMessage(scopeDescriptions.judge, values);
    case 'SPAN_NAME':
      return intl.formatMessage(scopeDescriptions.tool, values);
    case 'SPAN_MODEL':
      return intl.formatMessage(scopeDescriptions.model, values);
    case 'ERROR':
      return intl.formatMessage(scopeDescriptions.exception, values);
    case 'SPAN_TYPE':
      return intl.formatMessage(scopeDescriptions.spanType, values);
    default:
      return intl.formatMessage(scopeDescriptions.traceStatus, values);
  }
};

/**
 * How many samples a percentile needs before it means anything.
 *
 * Mirrors `derive_min_sample_count` in `mlflow/genai/alerts/entities.py`. This is
 * what the form *seeds* its minimum-samples input with -- a suggestion the user
 * can raise, lower, or clear to zero. The server applies the same function only
 * when a request omits the field entirely.
 *
 * Zero for everything but a percentile: the floor exists to stop a percentile
 * being computed from too few points, and applying it to a COUNT rule would
 * make "requests dropped to zero" read as insufficient data at exactly the
 * moment it should fire.
 */
export const deriveMinSampleCount = (aggregation: string, percentileValue?: number): number => {
  if (aggregation !== 'PERCENTILE' || percentileValue === undefined) {
    return 0;
  }
  const tail = Math.min(percentileValue, 100 - percentileValue);
  if (tail <= 0) {
    return 0;
  }
  return Math.floor(10 / (tail / 100));
};
