import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';

import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { AlertDetailDrawer } from './AlertDetailDrawer';
import type { AlertInstance, AlertRule, AlertSeries } from './types';

const mockSeries = jest.fn<() => { series: AlertSeries | undefined; isLoading: boolean; error: null }>();

jest.mock('./hooks/useAlertRuleSeries', () => ({
  useAlertRuleSeries: () => mockSeries(),
}));

const rule: AlertRule = {
  alert_rule_id: 'rule-1',
  experiment_id: 123,
  name: 'Checkout p95 latency',
  metric_key: 'latency',
  dimension_key: 'TRACES',
  dimension_value: '',
  aggregation: 'PERCENTILE',
  percentile_value: 95,
  comparator: 'GT',
  threshold: 30 * 60_000,
  window_seconds: 600,
  evaluation_interval_seconds: 60,
  sustain_seconds: 0,
  min_sample_count: 0,
  severity: 'HIGH',
  enabled: true,
};

const instance: AlertInstance = {
  alert_instance_id: 'inst-1',
  alert_rule_id: 'rule-1',
  experiment_id: 123,
  state: 'FIRED',
  started_at_ms: 1_700_000_000_000,
  fired_at_ms: 1_700_000_120_000,
  window_start_ms: 1_699_999_400_000,
  window_end_ms: 1_700_000_000_000,
  observed_value: 68 * 60_000,
  peak_value: 68 * 60_000,
  threshold: 30 * 60_000,
  sample_count: 619,
  exemplar_trace_ids: [],
};

describe('AlertDetailDrawer', () => {
  beforeEach(() => {
    mockSeries.mockReset();
    mockSeries.mockReturnValue({ series: undefined, isLoading: false, error: null });
  });

  const render = (overrides: Partial<AlertInstance> = {}) =>
    renderWithDesignSystem(
      <AlertDetailDrawer instance={{ ...instance, ...overrides }} rule={rule} onClose={jest.fn()} />,
    );

  it('quotes the peak in the same unit as the threshold', () => {
    render();
    // 4,080,000 ms is the stored figure; showing it raw next to "> 30 min"
    // leaves the reader dividing by 60,000.
    expect(document.body.textContent).toContain('68 min');
    expect(document.body.textContent).toContain('30 min');
    expect(document.body.textContent).not.toContain('4080000');
  });

  it('renders the condition the rule actually encodes', () => {
    render();
    expect(document.body.textContent).toContain('95th percentile latency > 30 min over 10 minutes');
  });

  it('shows the sample count behind the reading', () => {
    render();
    expect(document.body.textContent).toContain('619');
  });

  it('lists a recovery only once the episode has recovered', () => {
    render();
    expect(screen.queryByText('Recovered')).not.toBeInTheDocument();

    render({ state: 'INACTIVE', healthy_since_ms: 1_700_000_300_000 });
    expect(screen.getAllByText('Recovered').length).toBeGreaterThan(0);
  });

  it('says so when an alert captured no example traces', () => {
    render();
    expect(document.body.textContent).toContain('None were captured');
  });

  it('lists the exemplar traces it did capture', () => {
    render({ exemplar_trace_ids: ['tr-8f2a', 'tr-1c04'] });
    expect(document.body.textContent).toContain('tr-8f2a');
    expect(document.body.textContent).toContain('tr-1c04');
  });

  it('reports an aged-out window rather than drawing an empty chart', () => {
    mockSeries.mockReturnValue({
      series: { points: [], threshold: 30 * 60_000, window_seconds: 600, step_seconds: 60 },
      isLoading: false,
      error: null,
    });
    render();
    expect(document.body.textContent).toContain('No measurements are retained');
  });
});
