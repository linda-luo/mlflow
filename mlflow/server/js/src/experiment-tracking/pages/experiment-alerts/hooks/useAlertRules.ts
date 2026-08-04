import { useQuery } from '@databricks/web-shared/query-client';

import { listAlertRules } from '../api';
import type { AlertRule } from '../types';

export const LIST_ALERT_RULES_QUERY_KEY = 'LIST_ALERT_RULES';

/** Every non-deleted rule for an experiment, newest first (server-side). */
export const useAlertRules = ({ experimentId, enabled = true }: { experimentId: string; enabled?: boolean }) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<{ alert_rules: AlertRule[] }, Error>({
    queryKey: [LIST_ALERT_RULES_QUERY_KEY, experimentId],
    queryFn: () => listAlertRules(experimentId),
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId),
  });

  return {
    alertRules: data?.alert_rules ?? [],
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
