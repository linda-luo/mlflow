import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import userEventGlobal from '@testing-library/user-event';
import { PointerEventsCheckLevel } from '@testing-library/user-event';

import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import AlertsPage from './AlertsPage';
import type { AlertInstance, AlertInstanceState, AlertRule } from './types';

const mockDismiss = jest.fn<(...args: any[]) => Promise<any>>();
const mockInstances = jest.fn<(args: { states?: AlertInstanceState[] }) => { alertInstances: AlertInstance[] }>();

jest.mock('../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>('../../../common/utils/RoutingUtils'),
  useParams: () => ({ experimentId: '123' }),
}));

jest.mock('./hooks/useAlertRules', () => ({
  useAlertRules: () => ({ alertRules: [mockRule], isLoading: false }),
}));
jest.mock('./hooks/useAlertInstances', () => ({
  useAlertInstances: (args: { states?: AlertInstanceState[] }) => ({
    ...mockInstances(args),
    isLoading: false,
  }),
}));
jest.mock('./hooks/useDismissAlertInstance', () => ({
  useDismissAlertInstance: () => ({ dismissAlertInstanceAsync: mockDismiss, isDismissingAlertInstance: false }),
}));
jest.mock('./hooks/useDeleteAlertRule', () => ({
  useDeleteAlertRule: () => ({ deleteAlertRuleAsync: jest.fn() }),
}));
jest.mock('./hooks/useUpdateAlertRule', () => ({
  useUpdateAlertRule: () => ({ updateAlertRuleAsync: jest.fn(), isUpdatingAlertRule: false }),
}));

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const mockRule: AlertRule = {
  alert_rule_id: 'rule-1',
  experiment_id: 123,
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
};

const instance = (overrides: Partial<AlertInstance> = {}): AlertInstance => ({
  alert_instance_id: 'inst-1',
  alert_rule_id: 'rule-1',
  experiment_id: 123,
  state: 'FIRED',
  started_at_ms: 1_700_000_000_000,
  window_start_ms: 1,
  window_end_ms: 2,
  peak_value: 5_400_000,
  sample_count: 500,
  exemplar_trace_ids: [],
  ...overrides,
});

describe('AlertsPage', () => {
  beforeEach(() => {
    mockDismiss.mockReset();
    mockDismiss.mockImplementation(() => Promise.resolve({}));
    mockInstances.mockReset();
    mockInstances.mockReturnValue({ alertInstances: [] });
  });

  it('keeps a recovered alert in the active list, visually distinguished', () => {
    // The active list is the unacknowledged one, not the currently-breaching one:
    // an episode that recovered on its own is still someone's to look at, and the
    // tag is what says which of the two it is.
    mockInstances.mockImplementation(({ states }) => ({
      alertInstances: states
        ? []
        : [
            instance({ alert_instance_id: 'inst-inactive', state: 'INACTIVE', healthy_since_ms: 1_700_000_060_000 }),
            instance({ alert_instance_id: 'inst-fired', state: 'FIRED' }),
          ],
    }));

    renderWithDesignSystem(<AlertsPage />);

    // Both episodes are listed, each with its own tag. Before INACTIVE the
    // second one did not exist and the first read as still firing.
    expect(screen.getByText('Recovered')).toBeInTheDocument();
    // Twice: the firing episode's own tag, and the rule's status, which reports
    // the live episode rather than the recovered one.
    expect(screen.getAllByText('Alerting')).toHaveLength(2);
    // Recovered is still dismissible -- that is the only way it leaves the list.
    expect(screen.getAllByRole('button', { name: 'Dismiss' })).toHaveLength(2);
  });

  it('dismisses a recovered alert, which is how it leaves the list', () => {
    mockInstances.mockImplementation(({ states }) => ({
      alertInstances: states ? [] : [instance({ alert_instance_id: 'inst-inactive', state: 'INACTIVE' })],
    }));

    renderWithDesignSystem(<AlertsPage />);

    return userEvent.click(screen.getByRole('button', { name: 'Dismiss' })).then(() => {
      expect(mockDismiss).toHaveBeenCalledWith({
        alert_instance_id: 'inst-inactive',
        dismissed_by: 'default',
      });
    });
  });

  it('reports the rule as Recovered when its only open episode has recovered', () => {
    mockInstances.mockImplementation(({ states }) => ({
      alertInstances: states ? [] : [instance({ state: 'INACTIVE' })],
    }));

    renderWithDesignSystem(<AlertsPage />);

    // Once in the active list, once in the rules table's status column.
    expect(screen.getAllByText('Recovered')).toHaveLength(2);
    expect(screen.queryByText('Normal')).not.toBeInTheDocument();
  });
});
