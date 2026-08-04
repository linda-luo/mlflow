import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { fireEvent, waitFor } from '@testing-library/react';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import userEventGlobal from '@testing-library/user-event';
import React from 'react';

import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { CreateAlertModal } from './CreateAlertModal';
import type { AlertRule } from './types';

const CID = 'mlflow.experiment-alerts.create-rule';

const mockCreate = jest.fn<(...args: any[]) => Promise<any>>();
const mockUpdate = jest.fn<(...args: any[]) => Promise<any>>();
const mockDimensionValues = jest.fn<() => { dimensionValues: string[]; isLoading: boolean; error: Error | null }>();

jest.mock('./hooks/useCreateAlertRule', () => ({
  useCreateAlertRule: () => ({ createAlertRuleAsync: mockCreate, isCreatingAlertRule: false, error: null }),
}));
jest.mock('./hooks/useUpdateAlertRule', () => ({
  useUpdateAlertRule: () => ({ updateAlertRuleAsync: mockUpdate, isUpdatingAlertRule: false, error: null }),
}));
jest.mock('./hooks/useAlertDimensionValues', () => ({
  useAlertDimensionValues: () => mockDimensionValues(),
}));

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

/** The form's inputs are addressed by id; ids contain dots, so not via `#`. */
const inputById = (id: string) => {
  const input = document.querySelector<HTMLInputElement>(`[id="${id}"]`);
  if (!input) throw new Error(`No input with id "${id}"`);
  return input;
};

// `SimpleSelect` renders as a radix combobox button rather than a native
// `<select>`: open the trigger, then click the option.
const chooseOption = async (componentId: string, optionLabel: string | RegExp) => {
  const trigger = document.querySelector<HTMLElement>(`[data-component-id="${componentId}"]`);
  if (!trigger) throw new Error(`SimpleSelect "${componentId}" not found`);
  await userEvent.click(trigger);
  await userEvent.click(await screen.findByRole('option', { name: optionLabel }));
};

const submitButton = () => screen.getByRole('button', { name: /^(Create|Save)$/ });

const existingRule: AlertRule = {
  alert_rule_id: 'rule-1',
  experiment_id: 123,
  name: 'Slow checkout responses',
  metric_key: 'latency',
  dimension_key: 'SPAN_NAME',
  dimension_value: 'search_docs',
  aggregation: 'PERCENTILE',
  percentile_value: 99,
  comparator: 'GT',
  threshold: 45 * 60_000,
  window_seconds: 1800,
  evaluation_interval_seconds: 180,
  sustain_seconds: 600,
  min_sample_count: 1000,
  severity: 'HIGH',
  enabled: true,
};

const renderModal = (rule?: AlertRule) => {
  const onClose = jest.fn();
  renderWithDesignSystem(<CreateAlertModal experimentId="123" rule={rule} onClose={onClose} />);
  return { onClose };
};

describe('CreateAlertModal', () => {
  beforeEach(() => {
    mockCreate.mockReset();
    mockCreate.mockImplementation(() => Promise.resolve({ alert_rule: existingRule }));
    mockUpdate.mockReset();
    mockUpdate.mockImplementation(() => Promise.resolve({ alert_rule: existingRule }));
    mockDimensionValues.mockReset();
    mockDimensionValues.mockReturnValue({ dimensionValues: [], isLoading: false, error: null });
  });

  it('offers a trace status for latency, which is bucketed by it', () => {
    renderModal();
    expect(screen.getByText('Trace status')).toBeInTheDocument();
    expect(document.querySelector(`[data-component-id="${CID}.dimension-value"]`)).toBeInTheDocument();
  });

  it('offers no dimension value at all for whole-trace tokens', async () => {
    // `total_tokens` is summed across every trace status -- the aggregator writes
    // one series with an empty `dimension_value` -- so "tokens of OK traces" is a
    // rule the server would accept and nothing could ever match.
    renderModal();
    await chooseOption(`${CID}.metric`, 'Tokens');

    await waitFor(() => expect(screen.queryByText('Trace status')).not.toBeInTheDocument());
    expect(document.querySelector(`[data-component-id="${CID}.dimension-value"]`)).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Exact name, e.g. search_docs')).not.toBeInTheDocument();
    expect(document.body.textContent).toContain('aggregated across every trace status');
  });

  it('offers only Average for an error rate', async () => {
    // A ratio has no distribution, so PERCENTILE has nothing to bucket; SUM and
    // COUNT over the series would read as failures and invocations, which other
    // metrics already answer. The server rejects all three, so the form must not
    // offer them.
    renderModal();
    await chooseOption(`${CID}.metric`, 'Error rate');

    await waitFor(() => expect(screen.getByText('Average')).toBeInTheDocument());
    const trigger = document.querySelector<HTMLElement>(`[data-component-id="${CID}.aggregation"]`);
    await userEvent.click(trigger!);
    expect(await screen.findByRole('option', { name: 'Average' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Percentile of' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Total' })).not.toBeInTheDocument();
  });

  it('offers no dimension value for the whole-trace error rate', async () => {
    // The trace rate is one number per experiment. Slicing it by status would make
    // the error rate of ERROR traces 100% by construction.
    renderModal();
    await chooseOption(`${CID}.metric`, 'Error rate');

    await waitFor(() => expect(screen.queryByText('Trace status')).not.toBeInTheDocument());
    expect(document.querySelector(`[data-component-id="${CID}.dimension-value"]`)).not.toBeInTheDocument();
  });

  it('sends an error-rate threshold as a fraction of the percent typed', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Tool failures' } });
    await chooseOption(`${CID}.metric`, 'Error rate');
    await chooseOption(`${CID}.dimension`, 'Tool / span name');
    fireEvent.change(inputById(`${CID}.threshold-input`), { target: { value: '5' } });

    await userEvent.click(submitButton());
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    // Typed as 5%, stored as the fraction the evaluator compares against.
    expect(mockCreate.mock.calls[0][0]).toMatchObject({
      metric_key: 'error_rate',
      aggregation: 'AVG',
      threshold: 0.05,
    });
  });

  it('sends no dimension_value once the pair stops being sliceable', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Token budget' } });
    await chooseOption(`${CID}.dimension-value`, 'ERROR');
    await chooseOption(`${CID}.metric`, 'Tokens');

    await userEvent.click(submitButton());
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][0]).toMatchObject({ metric_key: 'total_tokens', dimension_value: undefined });
  });

  it('disables submit when the window is not a number', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Slow responses' } });
    expect(submitButton()).toBeEnabled();

    // `NaN` fails both ends of a bare range check, so without `Number.isFinite`
    // the field reads as valid and `JSON.stringify` puts `null` on the wire.
    fireEvent.change(inputById(`${CID}.window-input`), { target: { value: 'an hour' } });

    await waitFor(() => expect(submitButton()).toBeDisabled());
    expect(document.body.textContent).toContain('The window must be between 5 minutes and 3 days');
  });

  it('converts the window unit to seconds on the wire', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Weekly drift' } });
    fireEvent.change(inputById(`${CID}.window-input`), { target: { value: '3' } });
    await chooseOption(`${CID}.window-unit`, 'days');

    await userEvent.click(submitButton());
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][0]).toMatchObject({ window_seconds: 259_200 });
  });

  it('reopens a long window in the largest whole unit rather than in minutes', () => {
    renderModal({ ...existingRule, window_seconds: 259_200 });

    expect(inputById(`${CID}.window-input`).value).toBe('3');
    expect(inputById(`${CID}.window-unit-input`).value).toBe('days');
  });

  it('rejects a window past the raw data retention the ceiling is drawn from', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Weekly drift' } });
    fireEvent.change(inputById(`${CID}.window-input`), { target: { value: '4' } });
    await chooseOption(`${CID}.window-unit`, 'days');

    await waitFor(() => expect(submitButton()).toBeDisabled());
  });

  it('disables submit when the sustain period is not a number', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Slow responses' } });
    fireEvent.change(inputById(`${CID}.sustain-input`), { target: { value: 'ten' } });

    await waitFor(() => expect(submitButton()).toBeDisabled());
    expect(document.body.textContent).toContain('The sustain period must be a number of minutes');
  });

  it('disables submit when the sustain period is negative', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Slow responses' } });
    fireEvent.change(inputById(`${CID}.sustain-input`), { target: { value: '-5' } });

    await waitFor(() => expect(submitButton()).toBeDisabled());
  });

  it('warns about a value nothing has been seen under, without blocking submit', async () => {
    // A judge that has not run yet is a legitimate target: the rule is how you
    // find out it never ran.
    mockDimensionValues.mockReturnValue({ dimensionValues: ['safety'], isLoading: false, error: null });
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Freshness dropping' } });
    await chooseOption(`${CID}.metric`, 'Judge score');
    await chooseOption(`${CID}.dimension-value`, /Type a value/);
    fireEvent.change(screen.getByPlaceholderText('Exact name, e.g. search_docs'), {
      target: { value: 'freshness' },
    });

    await waitFor(() => expect(document.body.textContent).toContain('No judge "freshness" has been seen'));
    expect(submitButton()).toBeEnabled();
  });

  it('does not warn about a value that has been observed', async () => {
    mockDimensionValues.mockReturnValue({ dimensionValues: ['safety'], isLoading: false, error: null });
    renderModal();
    await chooseOption(`${CID}.metric`, 'Judge score');
    await chooseOption(`${CID}.dimension-value`, 'safety');

    await waitFor(() => expect(document.body.textContent).toContain('judge "safety"'));
    expect(document.body.textContent).not.toContain('has been seen in the last 24 hours');
  });

  it('says the check failed rather than that nothing was observed', async () => {
    // The query does not retry and is not cached, so a transient failure would
    // otherwise be indistinguishable from an experiment with no traffic.
    mockDimensionValues.mockReturnValue({ dimensionValues: [], isLoading: false, error: new Error('boom') });
    renderModal();
    await chooseOption(`${CID}.metric`, 'Judge score');

    await waitFor(() => expect(document.body.textContent).toContain('Could not load the values observed recently'));
    expect(document.body.textContent).not.toContain('Nothing observed yet');
  });

  it('prefills every field from an existing rule and patches it', async () => {
    renderModal(existingRule);

    expect(screen.getByText('Edit alert')).toBeInTheDocument();
    expect(inputById(`${CID}.name-input`)).toHaveValue('Slow checkout responses');
    // Stored scaled, shown in the unit it was typed in.
    expect(inputById(`${CID}.threshold-input`)).toHaveValue('45');
    expect(inputById(`${CID}.window-input`)).toHaveValue('30');
    expect(inputById(`${CID}.sustain-input`)).toHaveValue('10');
    expect(inputById(`${CID}.percentile-input`)).toHaveValue('99');

    await userEvent.click(submitButton());
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockCreate).not.toHaveBeenCalled();

    const { alert_rule_id: alertRuleId, updates } = mockUpdate.mock.calls[0][0] as any;
    expect(alertRuleId).toBe('rule-1');
    expect(updates).toMatchObject({
      name: 'Slow checkout responses',
      metric_key: 'latency',
      dimension_key: 'SPAN_NAME',
      dimension_value: 'search_docs',
      aggregation: 'PERCENTILE',
      percentile_value: 99,
      comparator: 'GT',
      threshold: 45 * 60_000,
      window_seconds: 1800,
      sustain_seconds: 600,
      severity: 'HIGH',
    });
    // A rule cannot move between experiments, and the store rejects the field.
    expect(updates).not.toHaveProperty('experiment_id');
  });

  it('clears percentile_value when an edited rule leaves PERCENTILE', async () => {
    // The server now accepts an explicit null here, so the stale percentile is
    // dropped rather than left sitting on a rule that no longer has one.
    renderModal(existingRule);
    await chooseOption(`${CID}.aggregation`, 'Average');

    await userEvent.click(submitButton());
    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const { updates } = mockUpdate.mock.calls[0][0] as any;
    expect(updates.aggregation).toBe('AVG');
    expect(updates.percentile_value).toBeNull();
  });

  it('seeds the sample floor from the percentile and follows it until edited', async () => {
    renderModal();
    const floor = inputById(`${CID}.min-samples-input`);
    expect(floor).toHaveValue('200'); // p95

    fireEvent.change(inputById(`${CID}.percentile-input`), { target: { value: '99' } });
    await waitFor(() => expect(floor).toHaveValue('1000'));

    // Once the user takes ownership of the number, the suggestion stops moving it.
    fireEvent.change(floor, { target: { value: '25' } });
    fireEvent.change(inputById(`${CID}.percentile-input`), { target: { value: '50' } });
    await waitFor(() => expect(inputById(`${CID}.percentile-input`)).toHaveValue('50'));
    expect(floor).toHaveValue('25');
  });

  it('seeds the sample floor from an edited rule rather than from its percentile', async () => {
    // The stored number is the user's, even where it disagrees with the suggestion.
    renderModal({ ...existingRule, percentile_value: 99, min_sample_count: 12 });
    expect(inputById(`${CID}.min-samples-input`)).toHaveValue('12');
  });

  it('sends the sample floor, including an explicit zero', async () => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'No floor' } });
    fireEvent.change(inputById(`${CID}.min-samples-input`), { target: { value: '0' } });

    await userEvent.click(submitButton());
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][0]).toMatchObject({ min_sample_count: 0 });
  });

  it.each(['-1', 'lots'])('disables submit when the sample floor is %p', async (bad) => {
    renderModal();
    fireEvent.change(inputById(`${CID}.name-input`), { target: { value: 'Slow responses' } });
    expect(submitButton()).toBeEnabled();

    fireEvent.change(inputById(`${CID}.min-samples-input`), { target: { value: bad } });

    await waitFor(() => expect(submitButton()).toBeDisabled());
    expect(document.body.textContent).toContain('must be a whole number of samples');
  });

  it('caps a typed dimension value at the width of the series column', async () => {
    renderModal();
    await chooseOption(`${CID}.metric`, 'Judge score');
    await chooseOption(`${CID}.dimension-value`, /Type a value/);
    expect(screen.getByPlaceholderText('Exact name, e.g. search_docs')).toHaveAttribute('maxlength', '250');
  });
});
