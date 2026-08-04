import { useQuery } from '@databricks/web-shared/query-client';

import { listAlertInstances } from '../api';
import type { AlertInstance, AlertInstanceState } from '../types';

export const LIST_ALERT_INSTANCES_QUERY_KEY = 'LIST_ALERT_INSTANCES';

/**
 * Firing episodes for an experiment, most recent first.
 *
 * With no `states` the server returns the undismissed ones (PENDING, FIRED,
 * INACTIVE) -- the active-alerts view. Pass explicit states for history. An
 * instance stays listed until a person dismisses it, so this view means
 * unacknowledged rather than currently breaching: INACTIVE is exactly the case
 * where those two differ.
 */
export const useAlertInstances = ({
  experimentId,
  states,
  enabled = true,
}: {
  experimentId: string;
  states?: AlertInstanceState[];
  enabled?: boolean;
}) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<{ alert_instances: AlertInstance[] }, Error>({
    queryKey: [LIST_ALERT_INSTANCES_QUERY_KEY, experimentId, states?.join(',') ?? 'open'],
    queryFn: () => listAlertInstances(experimentId, states),
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId),
  });

  return {
    alertInstances: data?.alert_instances ?? [],
    isLoading,
    isFetching,
    refetch,
    error,
  };
};
