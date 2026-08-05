import { useQuery } from '@databricks/web-shared/query-client';

import { getAlertRuleSeries } from '../api';
import type { AlertSeries } from '../types';

export const ALERT_RULE_SERIES_QUERY_KEY = 'ALERT_RULE_SERIES';

/**
 * The metric behind one rule, for the alert detail chart.
 *
 * The range is passed in rather than derived here so the caller can frame it on
 * the episode -- an alert is only legible next to the healthy stretch before it.
 */
export const useAlertRuleSeries = ({
  alertRuleId,
  startMs,
  endMs,
  enabled = true,
}: {
  alertRuleId: string;
  startMs: number;
  endMs: number;
  enabled?: boolean;
}) => {
  const { data, isLoading, error } = useQuery<AlertSeries, Error>({
    queryKey: [ALERT_RULE_SERIES_QUERY_KEY, alertRuleId, startMs, endMs],
    queryFn: () => getAlertRuleSeries(alertRuleId, startMs, endMs),
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(alertRuleId),
  });

  return { series: data, isLoading, error };
};
