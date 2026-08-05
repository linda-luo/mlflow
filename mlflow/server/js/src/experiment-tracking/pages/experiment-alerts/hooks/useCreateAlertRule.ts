import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { createAlertRule } from '../api';
import type { AlertRule, CreateAlertRuleRequest } from '../types';
import { LIST_ALERT_RULES_QUERY_KEY } from './useAlertRules';

/**
 * Create an alert rule. The server derives the evaluation interval, the
 * minimum sample count and the jittered first evaluation time, so the request
 * carries only what the user actually chose.
 */
export const useCreateAlertRule = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    { alert_rule: AlertRule },
    Error,
    CreateAlertRuleRequest
  >({
    mutationFn: (request) => createAlertRule(request),
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_ALERT_RULES_QUERY_KEY]);
    },
  });

  return {
    createAlertRule: mutate,
    createAlertRuleAsync: mutateAsync,
    isCreatingAlertRule: isLoading,
    error,
  };
};
