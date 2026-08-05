import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { deleteAlertRule } from '../api';
import { LIST_ALERT_INSTANCES_QUERY_KEY } from './useAlertInstances';
import { LIST_ALERT_RULES_QUERY_KEY } from './useAlertRules';

/**
 * Soft-delete a rule. Its instances stay readable -- the record of what a rule
 * caught is exactly what a postmortem needs -- so the instance lists are
 * invalidated too rather than assumed unchanged.
 */
export const useDeleteAlertRule = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<void, Error, { alert_rule_id: string }>({
    mutationFn: ({ alert_rule_id }) => deleteAlertRule(alert_rule_id),
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_ALERT_RULES_QUERY_KEY]);
      queryClient.invalidateQueries([LIST_ALERT_INSTANCES_QUERY_KEY]);
    },
  });

  return {
    deleteAlertRule: mutate,
    deleteAlertRuleAsync: mutateAsync,
    isDeletingAlertRule: isLoading,
    error,
  };
};
