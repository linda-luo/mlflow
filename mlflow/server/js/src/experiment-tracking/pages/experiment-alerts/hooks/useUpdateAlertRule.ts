import { useMutation, useQueryClient } from '@databricks/web-shared/query-client';

import { updateAlertRule } from '../api';
import type { AlertRule, UpdateAlertRuleRequest } from '../types';
import { LIST_ALERT_INSTANCES_QUERY_KEY } from './useAlertInstances';
import { LIST_ALERT_RULES_QUERY_KEY } from './useAlertRules';

/**
 * Patch a rule: an edit from the form, or the enable/disable toggle.
 *
 * The instance lists are invalidated as well as the rules, because a patch can
 * close instances server-side -- disabling a rule, or changing *what* it
 * measures, dismisses whatever it had open, since that episode now describes a
 * condition the rule no longer watches.
 */
export const useUpdateAlertRule = () => {
  const queryClient = useQueryClient();

  const { mutate, mutateAsync, isLoading, error } = useMutation<
    { alert_rule: AlertRule },
    Error,
    { alert_rule_id: string; updates: UpdateAlertRuleRequest }
  >({
    mutationFn: ({ alert_rule_id, updates }) => updateAlertRule(alert_rule_id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries([LIST_ALERT_RULES_QUERY_KEY]);
      queryClient.invalidateQueries([LIST_ALERT_INSTANCES_QUERY_KEY]);
    },
  });

  return {
    updateAlertRule: mutate,
    updateAlertRuleAsync: mutateAsync,
    isUpdatingAlertRule: isLoading,
    error,
  };
};
