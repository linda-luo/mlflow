import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import type { AlertInstance, AlertRule, AlertSeries, CreateAlertRuleRequest, UpdateAlertRuleRequest } from './types';

/**
 * Base path for the alerting REST API. Hand-rolled JSON endpoints; the server
 * registers the `/api/3.0` twin of each of these for non-browser clients.
 */
export const ALERTS_API_BASE = 'ajax-api/3.0/mlflow/alerts';

export const listAlertRules = async (experimentId: string): Promise<{ alert_rules: AlertRule[] }> => {
  const params = new URLSearchParams({ experiment_id: experimentId });
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/rules?${params.toString()}`), {
    method: 'GET',
  })) as { alert_rules: AlertRule[] };
};

export const createAlertRule = async (request: CreateAlertRuleRequest): Promise<{ alert_rule: AlertRule }> => {
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/rules`), {
    method: 'POST',
    body: request,
  })) as { alert_rule: AlertRule };
};

/**
 * Patch a rule in place. Only the fields in `UpdateAlertRuleRequest` are
 * updatable -- `experiment_id` in particular is rejected with a 400, so this
 * cannot take a `Partial<CreateAlertRuleRequest>`.
 */
export const updateAlertRule = async (
  alertRuleId: string,
  updates: UpdateAlertRuleRequest,
): Promise<{ alert_rule: AlertRule }> => {
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/rules/${alertRuleId}`), {
    method: 'PATCH',
    body: updates,
  })) as { alert_rule: AlertRule };
};

/** Soft delete: the rule leaves the table, its firing history stays readable. */
export const deleteAlertRule = async (alertRuleId: string): Promise<void> => {
  await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/rules/${alertRuleId}`), { method: 'DELETE' });
};

export const listAlertInstances = async (
  experimentId: string,
  states?: string[],
): Promise<{ alert_instances: AlertInstance[] }> => {
  const params = new URLSearchParams({ experiment_id: experimentId });
  states?.forEach((state) => params.append('states', state));
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/instances?${params.toString()}`), {
    method: 'GET',
  })) as { alert_instances: AlertInstance[] };
};

export const dismissAlertInstance = async (
  alertInstanceId: string,
  dismissedBy: string,
): Promise<{ alert_instance: AlertInstance }> => {
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/instances/${alertInstanceId}/dismiss`), {
    method: 'POST',
    body: { dismissed_by: dismissedBy },
  })) as { alert_instance: AlertInstance };
};

/**
 * Values observed for a dimension, read from `metric_series`.
 *
 * Never hardcode these: `make_judge()` lets a user name a judge anything, and
 * tool names and exception types are open vocabularies for the same reason.
 */
export const listAlertDimensionValues = async (
  experimentId: string,
  metricKey: string,
  dimensionKey: string,
): Promise<{ dimension_values: string[] }> => {
  const params = new URLSearchParams({
    experiment_id: experimentId,
    metric: metricKey,
    dimension_key: dimensionKey,
  });
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/dimension-values?${params.toString()}`), {
    method: 'GET',
  })) as { dimension_values: string[] };
};

/**
 * The rule's metric over a range, as the rule itself measured it.
 *
 * Each point is a rolling `window_seconds` projection rather than a per-bucket
 * value, so the line and the alert cannot disagree about when the threshold was
 * crossed. The server folds it; see `mlflow/alerts/series.py`.
 */
export const getAlertRuleSeries = async (alertRuleId: string, startMs: number, endMs: number): Promise<AlertSeries> => {
  const params = new URLSearchParams({
    alert_rule_id: alertRuleId,
    start_ms: String(Math.round(startMs)),
    end_ms: String(Math.round(endMs)),
  });
  return (await fetchAPI(getAjaxUrl(`${ALERTS_API_BASE}/series?${params.toString()}`), {
    method: 'GET',
  })) as AlertSeries;
};
