import React, { useMemo } from 'react';

import { Button, Drawer, Empty, Spinner, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, defineMessages, useIntl } from 'react-intl';
import type { MessageDescriptor } from 'react-intl';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import Utils from '../../../common/utils/Utils';
import { describeAlertRule, formatAlertValue } from './alertStatus';
import { useAlertRuleSeries } from './hooks/useAlertRuleSeries';
import type { AlertInstance, AlertRule } from './types';

const CID = 'mlflow.experiment-alerts.detail';

/**
 * How much history to frame around an episode.
 *
 * An alert is only legible beside the healthy stretch before it -- a chart that
 * starts at the breach shows a line that was always over the threshold, which
 * tells you nothing about what changed.
 */
const LEAD_IN_WINDOWS = 6;
const MIN_LEAD_IN_MS = 30 * 60_000;

/**
 * Two different scopes live in this block, and saying so is the whole point.
 *
 * `peak_value` is the worst reading of the entire episode. `sample_count` and the
 * window bounds are rewritten by every evaluation that touches the instance, so
 * they describe only the most recent one -- on a live alert they move every cycle.
 * Labelling the latter "Window" read as "the period this alert covers", which is
 * the one thing it does not mean.
 */
const factLabels = defineMessages({
  worst: { defaultMessage: 'Worst value in episode', description: 'Alert detail: peak observed value' },
  samples: {
    defaultMessage: 'Samples in that window',
    description: 'Alert detail: observations in the most recently evaluated window',
  },
  window: {
    defaultMessage: 'Last evaluated window',
    description: 'Alert detail: the window of the most recent evaluation',
  },
  opened: { defaultMessage: 'Opened', description: 'Alert detail: when the episode opened' },
  fired: { defaultMessage: 'Fired', description: 'Alert detail: when the alert started firing' },
  recovered: {
    defaultMessage: 'Recovered',
    description: 'Alert detail: when the metric became healthy again',
  },
  closed: { defaultMessage: 'Closed', description: 'Alert detail: when the alert was acknowledged' },
});

interface Fact {
  label: MessageDescriptor;
  value: React.ReactNode;
}

/**
 * Rows of "label: value", the drawer's whole layout vocabulary.
 *
 * Labels are descriptors rather than rendered elements so a row is plain data --
 * arrays of JSX are awkward to key and hide their strings from extraction.
 */
const Facts = ({ rows }: { rows: Fact[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div css={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: `${theme.spacing.xs}px ${theme.spacing.lg}px` }}>
      {rows.map(({ label, value }) => (
        <React.Fragment key={String(label.defaultMessage)}>
          <Typography.Text color="secondary">{intl.formatMessage(label)}</Typography.Text>
          <Typography.Text>{value}</Typography.Text>
        </React.Fragment>
      ))}
    </div>
  );
};

export const AlertDetailDrawer = ({
  instance,
  rule,
  onClose,
  onDismiss,
  onEditRule,
}: {
  instance: AlertInstance;
  rule?: AlertRule;
  onClose: () => void;
  onDismiss?: (instance: AlertInstance) => void;
  onEditRule?: (rule: AlertRule) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Framed on the episode rather than on "now": an alert from last Tuesday must
  // open showing last Tuesday, not an empty chart of the last hour.
  const { startMs, endMs } = useMemo(() => {
    const windowMs = (rule?.window_seconds ?? 600) * 1000;
    const leadIn = Math.max(LEAD_IN_WINDOWS * windowMs, MIN_LEAD_IN_MS);
    const closed = instance.dismissed_at_ms ?? instance.healthy_since_ms;
    return {
      startMs: instance.started_at_ms - leadIn,
      endMs: (closed ?? Date.now()) + windowMs,
    };
  }, [instance, rule]);

  const { series, isLoading: seriesLoading } = useAlertRuleSeries({
    alertRuleId: instance.alert_rule_id,
    startMs,
    endMs,
    enabled: Boolean(rule),
  });

  const chartData = useMemo(
    () =>
      (series?.points ?? []).map((p) => ({
        t: p.timestamp_ms,
        // Scale to the unit the threshold is quoted in, so the axis and the
        // reference line share a scale. `null` survives as null: recharts breaks
        // the line there, which is the point.
        v: p.value,
        gap: p.is_gap,
      })),
    [series],
  );

  const timeline: Fact[] = [{ label: factLabels.opened, value: Utils.formatTimestamp(instance.started_at_ms, intl) }];
  if (instance.fired_at_ms) {
    timeline.push({ label: factLabels.fired, value: Utils.formatTimestamp(instance.fired_at_ms, intl) });
  }
  if (instance.state === 'INACTIVE' && instance.healthy_since_ms) {
    timeline.push({
      label: factLabels.recovered,
      value: Utils.formatTimestamp(instance.healthy_since_ms, intl),
    });
  }
  if (instance.dismissed_at_ms) {
    timeline.push({
      label: factLabels.closed,
      value: `${Utils.formatTimestamp(instance.dismissed_at_ms, intl)}${
        instance.dismissed_by ? ` — ${instance.dismissed_by}` : ''
      }`,
    });
  }

  return (
    <Drawer.Root open onOpenChange={(open) => !open && onClose()}>
      <Drawer.Content
        componentId={`${CID}.drawer`}
        width="52vw"
        title={
          <Typography.Title level={3} withoutMargins>
            {rule?.name ?? (
              <FormattedMessage
                defaultMessage="Deleted rule"
                description="Alert detail: title when the rule behind the alert has been deleted"
              />
            )}
          </Typography.Title>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {rule && <Typography.Text color="secondary">{describeAlertRule(rule, intl)}</Typography.Text>}

          <div css={{ height: 220 }}>
            {seriesLoading ? (
              <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Spinner />
              </div>
            ) : chartData.length === 0 ? (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No measurements are retained for this period."
                    description="Alert detail: chart empty state when the window has aged out of retention"
                  />
                }
                title={null}
              />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.borderDecorative} />
                  <XAxis
                    dataKey="t"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(t: number) => intl.formatTime(t)}
                    tick={{ fontSize: 11, fill: theme.colors.textSecondary }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: theme.colors.textSecondary }}
                    tickFormatter={(v: number) => (rule ? formatAlertValue(rule, v, intl) : String(v))}
                    width={70}
                  />
                  <Tooltip
                    labelFormatter={(t) => Utils.formatTimestamp(Number(t), intl)}
                    formatter={(v) => (rule ? formatAlertValue(rule, Number(v), intl) : String(v))}
                  />
                  {/* The threshold as the rule saw it, snapshotted at fire time --
                      not the rule's current value, which may since have been edited. */}
                  <ReferenceLine
                    y={instance.threshold ?? series?.threshold}
                    stroke={theme.colors.textValidationDanger}
                    strokeDasharray="4 4"
                  />
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke={theme.colors.primary}
                    dot={false}
                    isAnimationActive={false}
                    // Gaps arrive as `null` and must stay holes: joining across one
                    // draws a slope nobody measured.
                    connectNulls={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          <Facts
            rows={[
              {
                label: factLabels.worst,
                value: rule ? (
                  <>
                    {formatAlertValue(rule, instance.peak_value ?? instance.observed_value, intl)}{' '}
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="(threshold {threshold})"
                        description="Alert detail: the threshold this value breached"
                        values={{ threshold: formatAlertValue(rule, instance.threshold, intl) }}
                      />
                    </Typography.Text>
                  </>
                ) : (
                  String(instance.peak_value ?? '—')
                ),
              },
              {
                label: factLabels.window,
                value: `${Utils.formatTimestamp(instance.window_start_ms, intl)} – ${Utils.formatTimestamp(
                  instance.window_end_ms,
                  intl,
                )}`,
              },
              { label: factLabels.samples, value: String(instance.sample_count ?? 0) },
            ]}
          />

          <Facts rows={timeline} />

          <div>
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage
                defaultMessage="Example traces"
                description="Alert detail: heading for the traces that caused the alert"
              />
            </Typography.Title>
            {instance.exemplar_trace_ids?.length ? (
              <div
                css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, marginTop: theme.spacing.sm }}
              >
                {instance.exemplar_trace_ids.map((traceId) => (
                  <Tag componentId={`${CID}.trace`} key={traceId} color="charcoal">
                    {traceId}
                  </Tag>
                ))}
              </div>
            ) : (
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="None were captured for this alert."
                  description="Alert detail: empty state when an alert stored no exemplar traces"
                />
              </Typography.Text>
            )}
          </div>

          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            {onDismiss && instance.state !== 'DISMISSED' && (
              <Button componentId={`${CID}.dismiss`} onClick={() => onDismiss(instance)}>
                <FormattedMessage defaultMessage="Dismiss" description="Alert detail: acknowledge the alert" />
              </Button>
            )}
            {rule && onEditRule && (
              <Button componentId={`${CID}.edit-rule`} onClick={() => onEditRule(rule)}>
                <FormattedMessage defaultMessage="Edit rule" description="Alert detail: open the rule editor" />
              </Button>
            )}
          </div>
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
