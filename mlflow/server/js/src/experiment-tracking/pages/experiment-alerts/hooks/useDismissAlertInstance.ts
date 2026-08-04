import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { dismissAlertInstance } from '../api';
import type { AlertInstance } from '../types';
import { LIST_ALERT_INSTANCES_QUERY_KEY } from './useAlertInstances';

/**
 * Acknowledge and close a firing episode.
 *
 * Alerts never remove themselves: a spike that recovered before anyone looked
 * still happened, so it goes INACTIVE rather than away, and this stays the only
 * way an instance leaves the active list.
 */
export const useDismissAlertInstance = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    { alert_instance: AlertInstance },
    Error,
    { alert_instance_id: string; dismissed_by: string }
  >({
    mutationFn: ({ alert_instance_id, dismissed_by }) => dismissAlertInstance(alert_instance_id, dismissed_by),
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_ALERT_INSTANCES_QUERY_KEY]);
    },
  });

  return {
    dismissAlertInstance: mutate,
    dismissAlertInstanceAsync: mutateAsync,
    isDismissingAlertInstance: isLoading,
    error,
  };
};
