import { useQuery } from '@databricks/web-shared/query-client';

import { listAlertDimensionValues } from '../api';

export const LIST_ALERT_DIMENSION_VALUES_QUERY_KEY = 'LIST_ALERT_DIMENSION_VALUES';

/**
 * The slices the rule editor can offer for a metric, read from the series that
 * have actually been observed. Judges, tools and exception types are open
 * vocabularies, so a hardcoded dropdown would be wrong the moment a user names
 * a judge something new.
 */
export const useAlertDimensionValues = ({
  experimentId,
  metricKey,
  dimensionKey,
  enabled = true,
}: {
  experimentId: string;
  metricKey?: string;
  dimensionKey?: string;
  enabled?: boolean;
}) => {
  const { data, isLoading, error } = useQuery<{ dimension_values: string[] }, Error>({
    queryKey: [LIST_ALERT_DIMENSION_VALUES_QUERY_KEY, experimentId, metricKey, dimensionKey],
    queryFn: () => listAlertDimensionValues(experimentId, metricKey ?? '', dimensionKey ?? ''),
    cacheTime: 0,
    refetchOnWindowFocus: false,
    retry: false,
    enabled: enabled && Boolean(experimentId) && Boolean(metricKey) && Boolean(dimensionKey),
  });

  return {
    dimensionValues: data?.dimension_values ?? [],
    isLoading,
    error,
  };
};
