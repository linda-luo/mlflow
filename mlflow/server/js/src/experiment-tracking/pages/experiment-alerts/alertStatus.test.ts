import { describe, it, expect } from '@jest/globals';
import { createIntl, createIntlCache } from 'react-intl';
import {
  deriveMinSampleCount,
  describeAlertRule,
  describeAlertScope,
  getAlertInstanceStatus,
  getAlertRuleStatus,
} from './alertStatus';
import type { AlertRuleStatus } from './alertStatus';
import type { AlertInstance, AlertRule } from './types';

const intl = createIntl({ locale: 'en' }, createIntlCache());

const rule = (overrides: Partial<AlertRule> = {}): AlertRule => ({
  alert_rule_id: 'rule-1',
  experiment_id: 1,
  name: 'Slow checkout responses',
  metric_key: 'latency',
  dimension_key: 'TRACES',
  aggregation: 'PERCENTILE',
  percentile_value: 95,
  comparator: 'GT',
  threshold: 45 * 60_000,
  window_seconds: 3600,
  evaluation_interval_seconds: 300,
  sustain_seconds: 0,
  min_sample_count: 200,
  severity: 'HIGH',
  enabled: true,
  last_sample_count: 500,
  ...overrides,
});

const instance = (overrides: Partial<AlertInstance> = {}): AlertInstance => ({
  alert_instance_id: 'inst-1',
  alert_rule_id: 'rule-1',
  experiment_id: 1,
  state: 'FIRED',
  started_at_ms: 1,
  window_start_ms: 1,
  window_end_ms: 2,
  sample_count: 500,
  exemplar_trace_ids: [],
  ...overrides,
});

describe('getAlertRuleStatus', () => {
  const statusCases: [AlertRuleStatus, AlertInstance[], Partial<AlertRule>][] = [
    ['ALERTING', [instance({ state: 'FIRED' })], {}],
    ['CONFIRMING', [instance({ state: 'PENDING' })], {}],
    // Recovered on its own, but nobody has acknowledged it, so it is not NORMAL.
    ['RECOVERED', [instance({ state: 'INACTIVE' })], {}],
    ['NORMAL', [], {}],
    // Below the rule's own minimum is not the same as never evaluated: one is
    // fixable by editing the rule, the other only by waiting for traffic.
    ['BELOW_MINIMUM', [], { last_sample_count: 3 }],
    ['NO_DATA', [], { last_sample_count: null }],
    ['NORMAL', [], { last_sample_count: 3, min_sample_count: 0 }],
    ['DISABLED', [instance({ state: 'FIRED' })], { enabled: false }],
  ];

  it.each(statusCases)('is %s', (expected, instances, overrides) => {
    expect(getAlertRuleStatus(rule(overrides), instances)).toBe(expected);
  });

  it('ignores instances belonging to other rules', () => {
    expect(getAlertRuleStatus(rule(), [instance({ alert_rule_id: 'other' })])).toBe('NORMAL');
  });

  it('treats a dismissed instance as closed', () => {
    expect(getAlertRuleStatus(rule(), [instance({ state: 'DISMISSED' })])).toBe('NORMAL');
  });

  it('prefers ALERTING when both a fired and a pending instance are open', () => {
    expect(getAlertRuleStatus(rule(), [instance({ state: 'PENDING' }), instance({ alert_instance_id: 'i2' })])).toBe(
      'ALERTING',
    );
  });

  it('prefers the live episode when a recovered one is still unacknowledged', () => {
    // The scenario INACTIVE exists for: one rule, two episodes. The rule's own
    // status has to describe the one still happening.
    expect(
      getAlertRuleStatus(rule(), [
        instance({ alert_instance_id: 'i1', state: 'INACTIVE' }),
        instance({ alert_instance_id: 'i2', state: 'FIRED' }),
      ]),
    ).toBe('ALERTING');
    expect(
      getAlertRuleStatus(rule(), [
        instance({ alert_instance_id: 'i1', state: 'INACTIVE' }),
        instance({ alert_instance_id: 'i2', state: 'PENDING' }),
      ]),
    ).toBe('CONFIRMING');
  });
});

describe('getAlertInstanceStatus', () => {
  const instanceCases: [AlertInstance['state'], AlertRuleStatus][] = [
    ['FIRED', 'ALERTING'],
    ['PENDING', 'CONFIRMING'],
    ['INACTIVE', 'RECOVERED'],
  ];

  it.each(instanceCases)('renders %s as %s', (state, expected) => {
    expect(getAlertInstanceStatus(state)).toBe(expected);
  });
});

describe('describeAlertRule', () => {
  it('renders a percentile latency rule in minutes', () => {
    expect(describeAlertRule(rule(), intl)).toBe('95th percentile latency > 45 min over 1 hour');
  });

  // A 3-day window is 4,320 minutes, which is the number the summary used to print.
  it('states the longest window in days rather than in thousands of minutes', () => {
    expect(describeAlertRule(rule({ window_seconds: 259_200 }), intl)).toBe(
      '95th percentile latency > 45 min over 3 days',
    );
  });

  it('does not pluralise a window of exactly one unit', () => {
    expect(describeAlertRule(rule({ window_seconds: 86_400 }), intl)).toBe(
      '95th percentile latency > 45 min over 1 day',
    );
  });

  it('falls back to minutes when no larger unit divides the window exactly', () => {
    expect(describeAlertRule(rule({ window_seconds: 5_400 }), intl)).toBe(
      '95th percentile latency > 45 min over 90 minutes',
    );
  });

  it('renders the slice of a scoped rule', () => {
    expect(
      describeAlertRule(
        rule({
          metric_key: 'assessment_value',
          dimension_key: 'ASSESSMENTS',
          aggregation: 'AVG',
          percentile_value: null,
          dimension_value: 'safety',
          comparator: 'LT',
          threshold: 0.9,
          window_seconds: 1800,
        }),
        intl,
      ),
    ).toBe('Average judge score [safety] < 0.9 over 30 minutes');
  });

  it('renders whatever percentile the rule actually uses', () => {
    // The form no longer offers only p50 and p95, so the description cannot
    // assume one either.
    expect(describeAlertRule(rule({ percentile_value: 99 }), intl)).toBe(
      '99th percentile latency > 45 min over 1 hour',
    );
    expect(describeAlertRule(rule({ percentile_value: 99.9 }), intl)).toBe(
      '99.9th percentile latency > 45 min over 1 hour',
    );
  });

  it('renders a latency rule scoped to one tool', () => {
    // p99 of one tool is exactly the combination the preset list could not express.
    expect(
      describeAlertRule(
        rule({ dimension_key: 'SPAN_NAME', dimension_value: 'search_docs', percentile_value: 99 }),
        intl,
      ),
    ).toBe('99th percentile latency [search_docs] > 45 min over 1 hour');
  });

  it('does not scale a COUNT threshold into the metric unit', () => {
    // COUNT counts rows. Scaling "fewer than 5 requests" by the latency scale
    // would render it as a fraction of a minute.
    expect(
      describeAlertRule(rule({ aggregation: 'COUNT', percentile_value: null, comparator: 'LT', threshold: 5 }), intl),
    ).toBe('Number of requests < 5 requests over 1 hour');
  });

  it('counts errors in failures rather than requests', () => {
    expect(
      describeAlertRule(
        rule({
          metric_key: 'error_count',
          dimension_key: 'ERROR',
          dimension_value: 'TimeoutError',
          aggregation: 'COUNT',
          percentile_value: null,
          comparator: 'GTE',
          threshold: 5,
        }),
        intl,
      ),
    ).toBe('Number of errors [TimeoutError] >= 5 failures over 1 hour');
  });

  it('renders an error rate as a percentage', () => {
    // Stored as a fraction, shown as a percent: the user typed 5, not 0.05.
    expect(
      describeAlertRule(
        rule({
          metric_key: 'error_rate',
          dimension_key: 'SPAN_NAME',
          dimension_value: 'search_docs',
          aggregation: 'AVG',
          percentile_value: null,
          comparator: 'GT',
          threshold: 0.05,
        }),
        intl,
      ),
    ).toBe('Average error rate [search_docs] > 5% over 1 hour');
  });

  it('counts judge output in verdicts rather than requests', () => {
    // A COUNT of `assessment_value` counts judge verdicts. Calling them requests
    // was wrong twice over: one request can produce several verdicts, and a judge
    // that never ran produces none.
    expect(
      describeAlertRule(
        rule({
          metric_key: 'assessment_value',
          dimension_key: 'ASSESSMENTS',
          dimension_value: 'safety',
          aggregation: 'COUNT',
          percentile_value: null,
          comparator: 'LT',
          threshold: 20,
        }),
        intl,
      ),
    ).toBe('Number of judge verdicts [safety] < 20 verdicts over 1 hour');
  });

  it('leaves a large threshold ungrouped', () => {
    // The number is rendered as the user typed it, not through a locale's
    // grouping separators, so a token budget reads back the way it was entered.
    expect(
      describeAlertRule(
        rule({
          metric_key: 'total_tokens',
          aggregation: 'SUM',
          percentile_value: null,
          threshold: 2_700_000,
        }),
        intl,
      ),
    ).toBe('Total tokens > 2700000 tokens over 1 hour');
  });

  it('renders a cost threshold with its currency prefix', () => {
    expect(
      describeAlertRule(
        rule({
          metric_key: 'total_cost',
          dimension_key: 'SPAN_MODEL',
          dimension_value: 'gpt-5',
          aggregation: 'SUM',
          percentile_value: null,
          threshold: 12.5,
        }),
        intl,
      ),
    ).toBe('Total cost [gpt-5] > $12.5 over 1 hour');
  });

  it('falls back to the raw metric key for a rule this build does not know', () => {
    // Rules arrive over the wire, so an older UI can meet a newer server.
    expect(
      describeAlertRule(rule({ metric_key: 'future_metric', aggregation: 'AVG', percentile_value: null }), intl),
    ).toBe('Average future_metric > 2700000 over 1 hour');
  });
});

describe('describeAlertScope', () => {
  const scopeCases: [string, string | undefined, string][] = [
    ['TRACES', undefined, 'all traces in this experiment'],
    ['ASSESSMENTS', 'safety', 'judge "safety"'],
    ['SPAN_NAME', 'search_docs', 'tool "search_docs"'],
    ['SPAN_MODEL', 'gpt-5', 'model "gpt-5"'],
    ['ERROR', 'TimeoutError', 'exception type "TimeoutError"'],
    ['SPAN_TYPE', 'LLM', 'spans of type "LLM"'],
    ['TRACES', 'ERROR', 'traces with status "ERROR"'],
  ];

  it.each(scopeCases)('describes %s/%s', (dimensionKey, dimensionValue, expected) => {
    expect(describeAlertScope(dimensionKey, dimensionValue, intl)).toBe(expected);
  });
});

describe('deriveMinSampleCount', () => {
  const sampleCases: [string, number | undefined, number][] = [
    ['PERCENTILE', 50, 20],
    ['PERCENTILE', 95, 200],
    ['PERCENTILE', 99, 1000],
    ['COUNT', undefined, 0],
    ['AVG', undefined, 0],
  ];

  it.each(sampleCases)('%s at p%s needs %s samples', (aggregation, percentile, expected) => {
    expect(deriveMinSampleCount(aggregation, percentile)).toBe(expected);
  });
});
