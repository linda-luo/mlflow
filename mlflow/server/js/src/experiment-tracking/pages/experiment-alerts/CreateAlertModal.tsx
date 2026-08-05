import { useEffect, useMemo, useState } from 'react';

import {
  Alert,
  Button,
  FormUI,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, defineMessages, useIntl } from 'react-intl';
import type { MessageDescriptor } from 'react-intl';

import Utils from '../../../common/utils/Utils';
import { deriveMinSampleCount, describeAlertScope } from './alertStatus';
import { useAlertDimensionValues } from './hooks/useAlertDimensionValues';
import { useCreateAlertRule } from './hooks/useCreateAlertRule';
import { useUpdateAlertRule } from './hooks/useUpdateAlertRule';
import {
  ALERT_AGGREGATION_LABELS,
  ALERT_DIMENSION_META,
  ALERT_METRIC_CATALOGUE,
  ALERT_METRIC_KEYS,
  MAX_ALERT_DIMENSION_VALUE_LENGTH,
  WINDOW_MAX_SECONDS,
  WINDOW_MIN_SECONDS,
  WINDOW_UNIT_SECONDS,
  splitWindow,
  alertCountUnitFor,
  isAlertDimensionSliceable,
  isAlertMetricKey,
} from './types';
import type {
  AlertAggregation,
  AlertChannel,
  AlertComparator,
  AlertDimensionKey,
  AlertMetricKey,
  AlertMetricSpec,
  AlertRule,
  AlertSeverity,
  AlertThresholdUnit,
  WindowUnit,
} from './types';

const CID = 'mlflow.experiment-alerts.create-rule';

const COMPARATORS: { value: AlertComparator; label: string }[] = [
  { value: 'GT', label: '>' },
  { value: 'GTE', label: '>=' },
  { value: 'LT', label: '<' },
  { value: 'LTE', label: '<=' },
];

const SEVERITIES: AlertSeverity[] = ['LOW', 'MEDIUM', 'HIGH'];

/** Sentinel for "any value of this dimension" -- an empty string is a valid value. */
const ANY_VALUE = '__any__';
/** Sentinel that swaps the suggestion dropdown for a free-text box. */
const CUSTOM_VALUE = '__custom__';

const windowUnitOptions = defineMessages({
  minutes: { defaultMessage: 'minutes', description: 'Alert rule editor: window unit option' },
  hours: { defaultMessage: 'hours', description: 'Alert rule editor: window unit option' },
  days: { defaultMessage: 'days', description: 'Alert rule editor: window unit option' },
}) satisfies Record<WindowUnit, MessageDescriptor>;

const unitHints = defineMessages({
  minutes: { defaultMessage: 'minutes', description: 'Alert rule editor: unit of a latency threshold' },
  tokens: { defaultMessage: 'tokens', description: 'Alert rule editor: unit of a token count threshold' },
  usd: { defaultMessage: 'USD', description: 'Alert rule editor: unit of a spend threshold' },
  score: { defaultMessage: '0 – 1', description: 'Alert rule editor: range of a judge score threshold' },
  percent: { defaultMessage: 'percent', description: 'Alert rule editor: unit of an error-rate threshold' },
  requests: { defaultMessage: 'requests', description: 'Alert rule editor: unit of a threshold counting requests' },
  failures: { defaultMessage: 'failures', description: 'Alert rule editor: unit of a threshold counting failures' },
  verdicts: {
    defaultMessage: 'verdicts',
    description: 'Alert rule editor: unit of a threshold counting judge verdicts',
  },
}) satisfies Record<AlertThresholdUnit, MessageDescriptor>;

/**
 * The form's starting values, read from an existing rule when there is one.
 *
 * The threshold is un-scaled back into the unit the user typed it in, because
 * that is the only unit the form ever shows: a 45 minute latency rule is stored
 * as 2,700,000 and must come back as 45.
 */
const initialFormState = (rule?: AlertRule) => {
  const metricKey: AlertMetricKey = rule && isAlertMetricKey(rule.metric_key) ? rule.metric_key : 'latency';
  const aggregation = rule?.aggregation ?? 'PERCENTILE';
  // COUNT counts rows, so its threshold was never scaled on the way in either.
  const thresholdScale = aggregation === 'COUNT' ? 1 : ALERT_METRIC_CATALOGUE[metricKey].thresholdScale;
  return {
    name: rule?.name ?? '',
    severity: rule?.severity ?? ('MEDIUM' as AlertSeverity),
    metricKey,
    dimensionKey: rule?.dimension_key ?? ('TRACES' as AlertDimensionKey),
    aggregation,
    percentile: String(rule?.percentile_value ?? 95),
    dimensionValue: rule?.dimension_value ?? '',
    comparator: rule?.comparator ?? ('GT' as AlertComparator),
    threshold: rule ? String(rule.threshold / thresholdScale) : '45',
    // Read back in the largest unit that divides exactly, so a 3-day rule opens as
    // "3 days" rather than "4320 minutes".
    ...(() => {
      const { amount, unit } = splitWindow(rule?.window_seconds ?? 3_600);
      return { windowAmount: String(amount), windowUnit: unit };
    })(),
    sustainMinutes: rule?.sustain_seconds ? String(rule.sustain_seconds / 60) : '',
    // An existing rule's own number, whatever it is; a new rule starts at the
    // statistical suggestion for its percentile.
    minSamples: String(rule?.min_sample_count ?? deriveMinSampleCount(aggregation, rule?.percentile_value ?? 95)),
    channels: rule?.channels ?? [],
  };
};

/**
 * Build a rule from its parts, rather than pick from combinations someone
 * anticipated.
 *
 * The previous form offered nine preset rules, which covered the common cases and
 * nothing else: there was no way to ask for p99 rather than p95, or to scope a
 * latency rule to one tool, or to alert on a judge the presets did not name. Every
 * dropdown here is driven by `ALERT_METRIC_CATALOGUE`, which mirrors the server's
 * `METRIC_CATALOGUE`, so the form still cannot offer a combination the evaluator
 * would reject.
 *
 * The evaluation interval stays derived from the window and is shown read-only --
 * it is not a number anyone would want to pick. The minimum sample count is not:
 * it decides whether the rule can fire at all, so it is an ordinary input seeded
 * with the statistically sensible value for the chosen percentile. It used to be
 * derived and never shown, which left a p99 rule on a low-volume workload sitting
 * permanently in "not enough data" with nothing on screen naming the number
 * responsible.
 *
 * Passing `rule` turns this into an editor for that rule. One component rather
 * than two, because a create form and an edit form that drift apart is how a
 * field ends up settable but not changeable.
 */
export const CreateAlertModal = ({
  experimentId,
  rule,
  onClose,
}: {
  experimentId: string;
  rule?: AlertRule;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { createAlertRuleAsync, isCreatingAlertRule } = useCreateAlertRule();
  const { updateAlertRuleAsync, isUpdatingAlertRule } = useUpdateAlertRule();

  const initial = useMemo(() => initialFormState(rule), [rule]);

  const [name, setName] = useState(initial.name);
  const [severity, setSeverity] = useState<AlertSeverity>(initial.severity);

  const [metricKey, setMetricKey] = useState<AlertMetricKey>(initial.metricKey);
  const [dimensionKey, setDimensionKey] = useState<AlertDimensionKey>(initial.dimensionKey);
  const [aggregation, setAggregation] = useState<AlertAggregation>(initial.aggregation);
  const [percentile, setPercentile] = useState(initial.percentile);

  const [valueChoice, setValueChoice] = useState<string>(initial.dimensionValue || ANY_VALUE);
  const [customValue, setCustomValue] = useState('');

  const [comparator, setComparator] = useState<AlertComparator>(initial.comparator);
  const [threshold, setThreshold] = useState(initial.threshold);
  const [windowAmount, setWindowAmount] = useState(initial.windowAmount);
  const [windowUnit, setWindowUnit] = useState<WindowUnit>(initial.windowUnit);
  const [sustainMinutes, setSustainMinutes] = useState(initial.sustainMinutes);
  const [channels, setChannels] = useState<AlertChannel[]>(initial.channels);
  const [minSamples, setMinSamples] = useState(initial.minSamples);
  // Until the user takes ownership of the minimum, the field tracks the
  // suggestion as the percentile changes -- moving p95 to p99 should move the
  // floor with it. Once they type, it is theirs and nothing overwrites it.
  const [minSamplesTouched, setMinSamplesTouched] = useState(rule !== undefined);

  const isEditing = rule !== undefined;
  const isSaving = isCreatingAlertRule || isUpdatingAlertRule;

  const spec: AlertMetricSpec = ALERT_METRIC_CATALOGUE[metricKey];
  const dimensionMeta = ALERT_DIMENSION_META[dimensionKey];

  // Changing the metric can invalidate the dimension and the aggregation, since
  // the catalogue permits different sets for each. Snap them back to something
  // legal rather than letting the form hold a combination the server would reject.
  useEffect(() => {
    if (!spec.dimensions.includes(dimensionKey)) {
      setDimensionKey(spec.dimensions[0]);
      setValueChoice(ANY_VALUE);
      setCustomValue('');
    }
    if (!spec.aggregations.includes(aggregation)) {
      setAggregation(spec.aggregations[0]);
    }
  }, [metricKey, spec, dimensionKey, aggregation]);

  // Sliceability belongs to the (metric, dimension) pair, not the dimension:
  // latency is bucketed by trace status, but whole-trace token counts are summed
  // across every status, so `total_tokens` + `TRACES` + "OK" names a series the
  // aggregator never writes. Offering no value control at all is the only way the
  // form cannot express it.
  const isSliceable = isAlertDimensionSliceable(metricKey, dimensionKey);

  // Open vocabularies: judge names, tool names, model names and exception types
  // are whatever users called them, so the options come from what has actually
  // been observed rather than from a hardcoded list.
  const {
    dimensionValues,
    isLoading: isLoadingDimensionValues,
    error: dimensionValuesError,
  } = useAlertDimensionValues({
    experimentId,
    metricKey,
    dimensionKey,
    enabled: isSliceable,
  });

  const selectedValue = valueChoice === ANY_VALUE || valueChoice === CUSTOM_VALUE ? '' : valueChoice;

  const suggestions = useMemo(() => {
    const merged = new Set<string>([...(dimensionMeta.suggestions ?? []), ...dimensionValues]);
    // An edited rule may name something that has since stopped appearing in
    // traffic; the dropdown still has to be able to render its own value.
    if (selectedValue) {
      merged.add(selectedValue);
    }
    return Array.from(merged).sort();
  }, [dimensionMeta.suggestions, dimensionValues, selectedValue]);

  const dimensionValue = !isSliceable ? '' : valueChoice === CUSTOM_VALUE ? customValue.trim() : selectedValue;

  // A judge that has not run yet is a legitimate target, so this warns and never
  // blocks. It only fires against a non-empty list: an empty one cannot tell
  // "nothing named that" apart from "nothing at all yet".
  const valueNeverObserved =
    isSliceable &&
    dimensionValue.length > 0 &&
    !isLoadingDimensionValues &&
    !dimensionValuesError &&
    dimensionValues.length > 0 &&
    !dimensionValues.includes(dimensionValue);

  const isPercentile = aggregation === 'PERCENTILE';
  const percentileValue = Number(percentile);
  const percentileValid =
    !isPercentile || (Number.isFinite(percentileValue) && percentileValue > 0 && percentileValue < 100);

  // The suggestion, which seeds the field and is offered back as a shortcut --
  // not what gets sent. What gets sent is whatever is in the input.
  const suggestedMinSamples = deriveMinSampleCount(
    aggregation,
    isPercentile && percentileValid ? percentileValue : undefined,
  );
  const minSampleCount = minSamples.trim() ? Math.round(Number(minSamples)) : 0;
  const minSamplesValid = Number.isFinite(minSampleCount) && minSampleCount >= 0;

  // While the minimum is still the suggestion, keep it *being* the suggestion:
  // moving p95 to p99 raises the number of samples a percentile needs, and a
  // stale 200 sitting under a p99 rule would be a floor nobody chose. Stops the
  // moment the user types.
  useEffect(() => {
    if (!minSamplesTouched) {
      setMinSamples(String(suggestedMinSamples));
    }
  }, [suggestedMinSamples, minSamplesTouched]);
  // `Number.isFinite` rather than a bare range check: letters parse to NaN, and
  // NaN fails both comparisons, so a range check alone would call the field valid
  // and then send `null` over the wire.
  const windowSeconds = Math.round(Number(windowAmount) * WINDOW_UNIT_SECONDS[windowUnit]);
  const windowValid =
    Number.isFinite(windowSeconds) && windowSeconds >= WINDOW_MIN_SECONDS && windowSeconds <= WINDOW_MAX_SECONDS;
  // The derived numbers below still have to render while the window is being
  // typed, so they fall back to the floor rather than printing NaN.
  const effectiveWindowSeconds = windowValid ? windowSeconds : WINDOW_MIN_SECONDS;
  // Mirrors `derive_evaluation_interval_seconds`: ten evaluations per window,
  // clamped. Shown rather than asked, but shown because a user tuning `sustain`
  // needs to know what it rounds up to.
  const intervalSeconds = Math.max(60, Math.min(300, Math.floor(effectiveWindowSeconds / 10)));

  const sustainSeconds = sustainMinutes.trim() ? Math.round(Number(sustainMinutes) * 60) : 0;
  const sustainValid = Number.isFinite(sustainSeconds) && sustainSeconds >= 0;

  // COUNT counts rows, so its threshold is never scaled into the metric's unit --
  // "fewer than 5 requests" must not become 5 milliseconds -- and the noun comes
  // from the metric, because a COUNT of judge output is verdicts, not requests.
  const thresholdScale = aggregation === 'COUNT' ? 1 : spec.thresholdScale;
  const thresholdUnit = aggregation === 'COUNT' ? alertCountUnitFor(metricKey) : spec.thresholdUnit;

  const canSubmit =
    name.trim().length > 0 &&
    threshold.trim().length > 0 &&
    Number.isFinite(Number(threshold)) &&
    percentileValid &&
    windowValid &&
    sustainValid &&
    minSamplesValid &&
    (!isSliceable || valueChoice !== CUSTOM_VALUE || customValue.trim().length > 0) &&
    !isSaving;

  const handleSubmit = async () => {
    if (!canSubmit) {
      return;
    }
    try {
      if (rule) {
        await updateAlertRuleAsync({
          alert_rule_id: rule.alert_rule_id,
          updates: {
            name: name.trim(),
            metric_key: metricKey,
            dimension_key: dimensionKey,
            aggregation,
            comparator,
            threshold: Number(threshold) * thresholdScale,
            window_seconds: windowSeconds,
            dimension_value: dimensionValue,
            // Cleared rather than left stale when a rule leaves PERCENTILE.
            percentile_value: isPercentile ? percentileValue : null,
            // Empty list rather than undefined, so clearing a channel persists.
            channels: channels.filter((c) => c.type.trim()),
            sustain_seconds: sustainSeconds,
            // Always sent, which is also what stops the store re-deriving it
            // behind the user's back when the aggregation changes.
            min_sample_count: minSampleCount,
            severity,
          },
        });
      } else {
        await createAlertRuleAsync({
          experiment_id: experimentId,
          name: name.trim(),
          metric_key: metricKey,
          dimension_key: dimensionKey,
          aggregation,
          comparator,
          threshold: Number(threshold) * thresholdScale,
          window_seconds: windowSeconds,
          dimension_value: dimensionValue || undefined,
          percentile_value: isPercentile ? percentileValue : undefined,
          // Empty list rather than undefined, so clearing a channel persists.
          channels: channels.filter((c) => c.type.trim()),
          sustain_seconds: sustainSeconds,
          min_sample_count: minSampleCount,
          severity,
        });
      }
      onClose();
    } catch (e) {
      // The modal stays open so the condition can be corrected and retried --
      // a rejected catalogue triple or a duplicate name is a fixable mistake.
      const error = e instanceof Error ? e.message : String(e);
      Utils.displayGlobalErrorNotification(
        isEditing
          ? intl.formatMessage(
              {
                defaultMessage: 'Failed to save the alert: {error}',
                description: 'Edit alert rule: error toast shown when saving fails',
              },
              { error },
            )
          : intl.formatMessage(
              {
                defaultMessage: 'Failed to create the alert: {error}',
                description: 'Create alert rule: error toast shown when creation fails',
              },
              { error },
            ),
      );
    }
  };

  return (
    <Modal
      componentId={`${CID}.modal`}
      visible
      size="wide"
      title={
        isEditing ? (
          <FormattedMessage defaultMessage="Edit alert" description="Edit alert rule modal title" />
        ) : (
          <FormattedMessage defaultMessage="New alert" description="Create alert rule modal title" />
        )
      }
      okText={
        isEditing ? (
          <FormattedMessage defaultMessage="Save" description="Edit alert rule: confirm button" />
        ) : (
          <FormattedMessage defaultMessage="Create" description="Create alert rule: confirm button" />
        )
      }
      okButtonProps={{ disabled: !canSubmit, loading: isSaving }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Create alert rule: cancel button" />}
      onOk={handleSubmit}
      onCancel={onClose}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div css={{ display: 'flex', gap: theme.spacing.md }}>
          <div css={{ flex: 2 }}>
            <FormUI.Label htmlFor={`${CID}.name-input`}>
              <FormattedMessage defaultMessage="Name" description="Create alert rule: name field label" />
            </FormUI.Label>
            <Input
              componentId={`${CID}.name`}
              id={`${CID}.name-input`}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={intl.formatMessage({
                defaultMessage: 'e.g. Slow checkout responses',
                description: 'Create alert rule: name field placeholder',
              })}
            />
          </div>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor={`${CID}.severity-input`}>
              <FormattedMessage defaultMessage="Severity" description="Create alert rule: severity field label" />
            </FormUI.Label>
            <SimpleSelect
              componentId={`${CID}.severity`}
              id={`${CID}.severity-input`}
              value={severity}
              onChange={({ target }) => setSeverity(target.value as AlertSeverity)}
              width="100%"
            >
              {SEVERITIES.map((value) => (
                <SimpleSelectOption key={value} value={value}>
                  {value}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
        </div>

        {/* What to measure ------------------------------------------------- */}
        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Measure" description="Create alert rule: metric section label" />
          </FormUI.Label>
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              alignItems: 'center',
              marginTop: theme.spacing.xs,
              flexWrap: 'wrap',
            }}
          >
            <SimpleSelect
              componentId={`${CID}.aggregation`}
              id={`${CID}.aggregation-input`}
              value={aggregation}
              onChange={({ target }) => setAggregation(target.value as AlertAggregation)}
              width={200}
            >
              {spec.aggregations.map((value) => (
                <SimpleSelectOption key={value} value={value}>
                  {intl.formatMessage(ALERT_AGGREGATION_LABELS[value])}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>

            {/* Just the field. Presets beside a free-text box that accepted the
                same values were two controls for one number, and the buttons only
                covered five of the ninety-eight legal answers -- the placeholder
                says what to type without pretending some values are special. */}
            {isPercentile && (
              <Input
                componentId={`${CID}.percentile`}
                id={`${CID}.percentile-input`}
                css={{ width: 120 }}
                value={percentile}
                onChange={(e) => setPercentile(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'ex: 95, 99',
                  description: 'Create alert rule: example percentile values',
                })}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Percentile',
                  description: 'Create alert rule: accessible name of the percentile field',
                })}
              />
            )}

            <SimpleSelect
              componentId={`${CID}.metric`}
              id={`${CID}.metric-input`}
              value={metricKey}
              onChange={({ target }) => setMetricKey(target.value as AlertMetricKey)}
              width={180}
            >
              {ALERT_METRIC_KEYS.map((key) => (
                <SimpleSelectOption key={key} value={key}>
                  {intl.formatMessage(ALERT_METRIC_CATALOGUE[key].label)}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
          {!percentileValid && (
            <FormUI.Message
              type="error"
              message={
                <FormattedMessage
                  defaultMessage="The percentile must be between 0 and 100."
                  description="Create alert rule: percentile out of range error"
                />
              }
            />
          )}
        </div>

        {/* Which slice ------------------------------------------------------ */}
        <div css={{ display: 'flex', gap: theme.spacing.md }}>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor={`${CID}.dimension-input`}>
              <FormattedMessage defaultMessage="Broken down by" description="Create alert rule: dimension label" />
            </FormUI.Label>
            <SimpleSelect
              componentId={`${CID}.dimension`}
              id={`${CID}.dimension-input`}
              value={dimensionKey}
              onChange={({ target }) => {
                setDimensionKey(target.value as AlertDimensionKey);
                setValueChoice(ANY_VALUE);
                setCustomValue('');
              }}
              width="100%"
            >
              {spec.dimensions.map((value) => (
                <SimpleSelectOption key={value} value={value}>
                  {intl.formatMessage(ALERT_DIMENSION_META[value].label)}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
          <div css={{ flex: 1 }}>
            {isSliceable ? (
              <>
                <FormUI.Label htmlFor={`${CID}.dimension-value-input`}>
                  {intl.formatMessage(dimensionMeta.valueLabel)}
                </FormUI.Label>
                <SimpleSelect
                  componentId={`${CID}.dimension-value`}
                  id={`${CID}.dimension-value-input`}
                  value={valueChoice}
                  onChange={({ target }) => setValueChoice(target.value)}
                  width="100%"
                >
                  <SimpleSelectOption value={ANY_VALUE}>
                    {intl.formatMessage(dimensionMeta.allLabel)}
                  </SimpleSelectOption>
                  {suggestions.map((value) => (
                    <SimpleSelectOption key={value} value={value}>
                      {value}
                    </SimpleSelectOption>
                  ))}
                  {dimensionMeta.openVocabulary && (
                    <SimpleSelectOption value={CUSTOM_VALUE}>
                      {intl.formatMessage({
                        defaultMessage: 'Type a value…',
                        description: 'Create alert rule: option that reveals a free-text dimension value box',
                      })}
                    </SimpleSelectOption>
                  )}
                </SimpleSelect>
                {valueChoice === CUSTOM_VALUE && (
                  <Input
                    componentId={`${CID}.dimension-value-custom`}
                    css={{ marginTop: theme.spacing.xs }}
                    value={customValue}
                    onChange={(e) => setCustomValue(e.target.value)}
                    // The `metric_series` column is 250 wide and the aggregator
                    // truncates observed values to it, so a longer value could
                    // never match the series it names.
                    maxLength={MAX_ALERT_DIMENSION_VALUE_LENGTH}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Exact name, e.g. search_docs',
                      description: 'Create alert rule: free-text dimension value placeholder',
                    })}
                    aria-label={intl.formatMessage(dimensionMeta.valueLabel)}
                  />
                )}
                {dimensionValuesError ? (
                  <FormUI.Message
                    type="warning"
                    message={
                      <FormattedMessage
                        defaultMessage="Could not load the values observed recently, so this one has not been checked against them."
                        description="Create alert rule: warning when the observed dimension values could not be fetched"
                      />
                    }
                  />
                ) : valueNeverObserved ? (
                  <FormUI.Message
                    type="warning"
                    message={
                      <FormattedMessage
                        defaultMessage="No {scope} has been seen in the last 24 hours. The alert will not fire until one appears."
                        description="Create alert rule: warning when the chosen value has never been observed"
                        values={{ scope: describeAlertScope(dimensionKey, dimensionValue, intl) }}
                      />
                    }
                  />
                ) : (
                  dimensionMeta.openVocabulary &&
                  !isLoadingDimensionValues &&
                  dimensionValues.length === 0 && (
                    <FormUI.Hint>
                      <FormattedMessage
                        defaultMessage="Nothing observed yet — type a name to alert on something that has not appeared in traffic."
                        description="Create alert rule: hint when no dimension values have been observed"
                      />
                    </FormUI.Hint>
                  )
                )}
              </>
            ) : (
              <FormUI.Hint css={{ display: 'block', marginTop: theme.spacing.lg }}>
                <FormattedMessage
                  defaultMessage="This measurement is aggregated across every {label}, so it cannot be narrowed to one."
                  description="Create alert rule: explains why a dimension value cannot be chosen for this metric"
                  values={{ label: intl.formatMessage(dimensionMeta.valueLabel).toLowerCase() }}
                />
              </FormUI.Hint>
            )}
          </div>
        </div>

        {/* The predicate ---------------------------------------------------- */}
        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Alert me when it is" description="Create alert rule: predicate label" />
          </FormUI.Label>
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', marginTop: theme.spacing.xs }}>
            <SimpleSelect
              componentId={`${CID}.comparator`}
              id={`${CID}.comparator-input`}
              value={comparator}
              onChange={({ target }) => setComparator(target.value as AlertComparator)}
              width={90}
            >
              {COMPARATORS.map((option) => (
                <SimpleSelectOption key={option.value} value={option.value}>
                  {option.label}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
            <Input
              componentId={`${CID}.threshold`}
              id={`${CID}.threshold-input`}
              css={{ width: 120 }}
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
            />
            <Typography.Text color="secondary">{intl.formatMessage(unitHints[thresholdUnit])}</Typography.Text>
          </div>
        </div>

        {/* Timing ----------------------------------------------------------- */}
        <div css={{ display: 'flex', gap: theme.spacing.md }}>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor={`${CID}.window-input`}>
              <FormattedMessage
                defaultMessage="Measured over the last"
                description="Create alert rule: window field label"
              />
            </FormUI.Label>
            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Input
                componentId={`${CID}.window`}
                id={`${CID}.window-input`}
                value={windowAmount}
                onChange={(e) => setWindowAmount(e.target.value)}
                css={{ flex: 1 }}
              />
              <SimpleSelect
                componentId={`${CID}.window-unit`}
                id={`${CID}.window-unit-input`}
                value={windowUnit}
                onChange={(e) => setWindowUnit(e.target.value as WindowUnit)}
                css={{ flex: 1 }}
              >
                {(Object.keys(WINDOW_UNIT_SECONDS) as WindowUnit[]).map((unit) => (
                  <SimpleSelectOption key={unit} value={unit}>
                    {intl.formatMessage(windowUnitOptions[unit])}
                  </SimpleSelectOption>
                ))}
              </SimpleSelect>
            </div>
            {!windowValid && (
              <FormUI.Message
                type="error"
                message={
                  <FormattedMessage
                    defaultMessage="The window must be between 5 minutes and 3 days. Shorter windows leave gaps that no evaluation inspects; longer ones reach past the raw data's retention, so they cannot be measured."
                    description="Create alert rule: window out of range error"
                  />
                }
              />
            )}
          </div>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor={`${CID}.sustain-input`}>
              <FormattedMessage
                defaultMessage="Only if sustained for (minutes, optional)"
                description="Create alert rule: sustain field label"
              />
            </FormUI.Label>
            <Input
              componentId={`${CID}.sustain`}
              id={`${CID}.sustain-input`}
              value={sustainMinutes}
              onChange={(e) => setSustainMinutes(e.target.value)}
              placeholder="0"
            />
            {sustainValid ? (
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Rounded up to the {interval}s evaluation interval, which is derived from the window."
                  description="Create alert rule: sustain field hint naming the derived interval"
                  values={{ interval: intervalSeconds }}
                />
              </FormUI.Hint>
            ) : (
              <FormUI.Message
                type="error"
                message={
                  <FormattedMessage
                    defaultMessage="The sustain period must be a number of minutes, and cannot be negative. Leave it empty to alert on the first breaching evaluation."
                    description="Create alert rule: sustain period invalid error"
                  />
                }
              />
            )}
          </div>
          <div css={{ flex: 1 }}>
            <FormUI.Label htmlFor={`${CID}.min-samples-input`}>
              <FormattedMessage
                defaultMessage="Minimum samples to fire"
                description="Create alert rule: minimum sample count field label"
              />
            </FormUI.Label>
            <Input
              componentId={`${CID}.min-samples`}
              id={`${CID}.min-samples-input`}
              value={minSamples}
              onChange={(e) => {
                setMinSamplesTouched(true);
                setMinSamples(e.target.value);
              }}
              placeholder="0"
            />
            {minSamplesValid ? (
              <FormUI.Hint>
                {minSampleCount > 0 ? (
                  <FormattedMessage
                    defaultMessage="Below {count} samples the rule reports “not enough data” and can neither fire nor resolve — roughly {perMinute} per minute at this window. Set 0 for no minimum."
                    description="Create alert rule: minimum sample count hint when a floor is set"
                    values={{
                      count: minSampleCount,
                      perMinute: Math.ceil(minSampleCount / Math.max(1, effectiveWindowSeconds / 60)),
                    }}
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="No minimum: the rule fires on whatever the window holds. A percentile over very few samples is close to just its largest value."
                    description="Create alert rule: minimum sample count hint when there is no floor"
                  />
                )}
              </FormUI.Hint>
            ) : (
              <FormUI.Message
                type="error"
                message={
                  <FormattedMessage
                    defaultMessage="The minimum must be a whole number of samples, and cannot be negative. Use 0 for no minimum."
                    description="Create alert rule: minimum sample count invalid error"
                  />
                }
              />
            )}
            {minSamplesTouched && minSampleCount !== suggestedMinSamples && suggestedMinSamples > 0 && (
              <FormUI.Hint>
                <FormattedMessage
                  defaultMessage="Suggested for p{percentile}: {suggested}"
                  description="Create alert rule: the statistical suggestion, once the user has overridden it"
                  values={{ percentile: percentile, suggested: suggestedMinSamples }}
                />
              </FormUI.Hint>
            )}
          </div>
        </div>

        {/* Scope is always visible. A filter hidden inside a rule is worse than
            a prescriptive dropdown, because nothing on screen says what the
            number actually covers. */}
        {/* Preview: the wire format and the registry are real, but nothing
            validates a channel type on the way in -- an unregistered type is
            accepted here and only fails, silently, when the alert fires. Hence
            the tag, and hence the hint below rather than a picker that would
            imply a supported set. */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            {/* No htmlFor: this heads a repeating group, and the rows it labels
                do not exist until the user adds one. Same as "Measure" above. */}
            <FormUI.Label>
              <FormattedMessage
                defaultMessage="Notify elsewhere"
                description="Create alert rule: optional external notification channels"
              />
            </FormUI.Label>
            <Tag componentId={`${CID}.channels-preview`} color="turquoise">
              <FormattedMessage
                defaultMessage="Preview"
                description="Create alert rule: the channels field is not a finished feature"
              />
            </Tag>
          </div>

          {channels.map((channel, index) => (
            // eslint-disable-next-line react/no-array-index-key -- rows are positional
            <div key={index} css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Input
                componentId={`${CID}.channel-type`}
                id={`${CID}.channel-type-${index}`}
                css={{ flex: 1 }}
                value={channel.type}
                placeholder={intl.formatMessage({
                  defaultMessage: 'channel type, e.g. slack',
                  description: 'Create alert rule: channel type placeholder',
                })}
                onChange={(e) =>
                  setChannels(channels.map((c, i) => (i === index ? { ...c, type: e.target.value } : c)))
                }
              />
              <Input
                componentId={`${CID}.channel-target`}
                id={`${CID}.channel-target-${index}`}
                css={{ flex: 2 }}
                value={channel.target ?? ''}
                placeholder={intl.formatMessage({
                  defaultMessage: 'target — webhook URL, address, room…',
                  description: 'Create alert rule: channel target placeholder',
                })}
                onChange={(e) =>
                  setChannels(channels.map((c, i) => (i === index ? { ...c, target: e.target.value } : c)))
                }
              />
              <Button
                componentId={`${CID}.channel-remove`}
                onClick={() => setChannels(channels.filter((_, i) => i !== index))}
              >
                <FormattedMessage
                  defaultMessage="Remove"
                  description="Create alert rule: remove a notification channel"
                />
              </Button>
            </div>
          ))}

          <div>
            <Button
              componentId={`${CID}.channel-add`}
              onClick={() => setChannels([...channels, { type: '', target: '' }])}
            >
              <FormattedMessage
                defaultMessage="Add channel"
                description="Create alert rule: add a notification channel"
              />
            </Button>
          </div>

          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="The alert always appears in this list. A channel sends it somewhere else too, and only works if a plugin has registered that type — an unrecognised one is saved but never delivers."
              description="Create alert rule: what notification channels do and their limitation"
            />
          </FormUI.Hint>
        </div>

        <Alert
          componentId={`${CID}.scope`}
          type="info"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Scope: {scope}"
              description="Create alert rule: scope summary"
              values={{ scope: describeAlertScope(dimensionKey, dimensionValue, intl) }}
            />
          }
        />
      </div>
    </Modal>
  );
};
