import { useState } from 'react';

import {
  Button,
  Empty,
  NotificationIcon,
  PlusIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeleton,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useParams } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import { AlertDetailDrawer } from './AlertDetailDrawer';
import { CreateAlertModal } from './CreateAlertModal';
import {
  describeAlertRule,
  describeSampleShortfall,
  formatAlertValue,
  getAlertInstanceStatus,
  getAlertRuleStatus,
} from './alertStatus';
import type { AlertRuleStatus } from './alertStatus';
import { useAlertInstances } from './hooks/useAlertInstances';
import { useAlertRules } from './hooks/useAlertRules';
import { useDeleteAlertRule } from './hooks/useDeleteAlertRule';
import { useDismissAlertInstance } from './hooks/useDismissAlertInstance';
import { useUpdateAlertRule } from './hooks/useUpdateAlertRule';
import { ALL_ALERT_STATES } from './types';
import type { AlertInstance, AlertRule } from './types';

const CID = 'mlflow.experiment-alerts';

/**
 * How tall one list may get before it scrolls instead of growing.
 *
 * Relative to the viewport so a tall window shows more rows, and a short one is
 * not handed a list taller than the screen it has to live on.
 */
const MAX_LIST_HEIGHT = '45vh';

const STATUS_TAG_COLOR: Record<AlertRuleStatus, 'coral' | 'lemon' | 'lime' | 'charcoal' | 'turquoise'> = {
  ALERTING: 'coral',
  CONFIRMING: 'lemon',
  // Distinct from both the red of a live alert and the green of a healthy rule:
  // it is over, but it is still waiting for someone to look at it.
  RECOVERED: 'turquoise',
  NORMAL: 'lime',
  NO_DATA: 'charcoal',
  BELOW_MINIMUM: 'lemon',
  DISABLED: 'charcoal',
  CLOSED: 'charcoal',
};

/**
 * One section of the page: a heading that stays put, over a list that scrolls
 * once it gets long.
 *
 * Three competing pressures, and the shape here is what satisfies all of them:
 *
 * * **A long list must not crowd out the others.** Forty rules should not push
 *   recent activity off the bottom, so each list caps at `MAX_LIST_HEIGHT` and
 *   scrolls past it.
 * * **A short window must stay usable.** Giving the three sections a fixed share
 *   of the viewport squeezed each list to about two rows at 620px tall -- three
 *   slivers with three scrollbars. So the page scrolls, and sections take their
 *   natural height up to the cap.
 * * **The heading has to stay legible while its list moves.** `position: sticky`
 *   keeps it against the top of the scroll container as its own rows pass under.
 */
const Section = ({
  title,
  action,
  children,
}: {
  title: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: '0 0 auto', gap: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
          minHeight: theme.general.heightSm,
          position: 'sticky',
          top: 0,
          zIndex: 1,
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        <Typography.Title level={3} withoutMargins>
          {title}
        </Typography.Title>
        {action}
      </div>
      {/* `maxHeight` rather than a height: a short list takes only what it
          needs, and only a long one scrolls. */}
      <div css={{ maxHeight: MAX_LIST_HEIGHT, overflowY: 'auto' }}>{children}</div>
    </div>
  );
};

const StatusTag = ({ status }: { status: AlertRuleStatus }) => (
  <Tag componentId={`${CID}.status-tag`} color={STATUS_TAG_COLOR[status]}>
    {status === 'ALERTING' && (
      <FormattedMessage defaultMessage="Alerting" description="Alerts table: rule has an unacknowledged alert" />
    )}
    {status === 'CONFIRMING' && (
      <FormattedMessage
        defaultMessage="Confirming"
        description="Alerts table: rule is breaching but has not sustained yet"
      />
    )}
    {status === 'RECOVERED' && (
      <FormattedMessage
        defaultMessage="Recovered"
        description="Alerts table: the alert stopped firing on its own but nobody has acknowledged it yet"
      />
    )}
    {status === 'NORMAL' && <FormattedMessage defaultMessage="Normal" description="Alerts table: rule is healthy" />}
    {status === 'NO_DATA' && (
      <FormattedMessage defaultMessage="No data yet" description="Alerts table: rule has never been evaluated" />
    )}
    {status === 'BELOW_MINIMUM' && (
      <FormattedMessage
        defaultMessage="Below minimum"
        description="Alerts table: rule saw fewer samples than its own configured minimum, so it cannot fire"
      />
    )}
    {status === 'DISABLED' && (
      <FormattedMessage defaultMessage="Disabled" description="Alerts table: rule is not being evaluated" />
    )}
    {status === 'CLOSED' && (
      <FormattedMessage defaultMessage="Closed" description="Alerts history: the episode was acknowledged" />
    )}
  </Tag>
);

/**
 * The Alerts tab: active alerts, the rules that produce them, and recent
 * activity.
 *
 * A rule and a firing episode are different things -- one rule that fired in
 * January and again in March has two instances -- so the active list is
 * instances while the table below is rules. Status is the join of the two.
 */
const AlertsPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId } = useParams<{ experimentId: string }>();
  const [createOpen, setCreateOpen] = useState(false);
  const [editingRule, setEditingRule] = useState<AlertRule | undefined>(undefined);
  const [openedInstance, setOpenedInstance] = useState<AlertInstance | undefined>(undefined);

  const { alertRules, isLoading: rulesLoading } = useAlertRules({ experimentId: experimentId ?? '' });
  const { alertInstances: openInstances, isLoading: instancesLoading } = useAlertInstances({
    experimentId: experimentId ?? '',
  });
  const { alertInstances: history } = useAlertInstances({
    experimentId: experimentId ?? '',
    states: ALL_ALERT_STATES,
  });
  const { dismissAlertInstanceAsync, isDismissingAlertInstance } = useDismissAlertInstance();
  const { deleteAlertRuleAsync } = useDeleteAlertRule();
  const { updateAlertRuleAsync, isUpdatingAlertRule } = useUpdateAlertRule();

  const ruleById = new Map(alertRules.map((rule) => [rule.alert_rule_id, rule]));
  const ruleName = (instance: AlertInstance) =>
    ruleById.get(instance.alert_rule_id)?.name ??
    intl.formatMessage({
      defaultMessage: 'Deleted rule',
      description: 'Alerts page: label for an instance whose rule has been soft-deleted',
    });

  // An instance's numbers only mean something in its rule's unit; without the
  // rule (deleted) there is nothing to scale by, so show the raw figure.
  const formatAlertValueFor = (instance: AlertInstance, value: number | null | undefined) => {
    const rule = ruleById.get(instance.alert_rule_id);
    return rule ? formatAlertValue(rule, value, intl) : String(value ?? '—');
  };

  const handleDismiss = async (instance: AlertInstance) => {
    try {
      await dismissAlertInstanceAsync({
        alert_instance_id: instance.alert_instance_id,
        // The server stamps the authenticated user when there is one; this is
        // the no-auth fallback.
        dismissed_by: 'default',
      });
    } catch (e) {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Could not dismiss the alert: {error}',
            description: 'Alerts page: error toast when dismissing an alert fails',
          },
          { error: e instanceof Error ? e.message : String(e) },
        ),
      );
    }
  };

  const handleDelete = async (rule: AlertRule) => {
    try {
      await deleteAlertRuleAsync({ alert_rule_id: rule.alert_rule_id });
    } catch (e) {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Could not delete the alert rule: {error}',
            description: 'Alerts page: error toast when deleting a rule fails',
          },
          { error: e instanceof Error ? e.message : String(e) },
        ),
      );
    }
  };

  // Disabling is the reversible half of deleting: the rule stops being scheduled
  // and its open instances are closed server-side, but the definition survives so
  // a noisy alert can be silenced during an incident and brought back after.
  const handleToggleEnabled = async (rule: AlertRule) => {
    try {
      await updateAlertRuleAsync({
        alert_rule_id: rule.alert_rule_id,
        updates: { enabled: !rule.enabled },
      });
    } catch (e) {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Could not change whether the alert rule is enabled: {error}',
            description: 'Alerts page: error toast when enabling or disabling a rule fails',
          },
          { error: e instanceof Error ? e.message : String(e) },
        ),
      );
    }
  };

  const renderActiveAlerts = () => {
    if (instancesLoading) {
      return <TableSkeleton lines={3} />;
    }
    if (openInstances.length === 0) {
      return (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No unacknowledged alerts."
            description="Alerts page: empty state for the active alerts section"
          />
        </Typography.Text>
      );
    }
    return (
      <Table>
        <TableRow isHeader>
          <TableHeader componentId={`${CID}.active.rule`} css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Rule" description="Active alerts table: rule column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.active.state`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="State" description="Active alerts table: state column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.active.observed`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Worst value" description="Active alerts table: peak value column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.active.started`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Started" description="Active alerts table: start time column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.active.actions`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Actions" description="Active alerts table: actions column" />
          </TableHeader>
        </TableRow>
        {openInstances.map((instance) => (
          <TableRow
            key={instance.alert_instance_id}
            css={{ cursor: 'pointer' }}
            onClick={() => setOpenedInstance(instance)}
          >
            <TableCell css={{ flex: 2 }}>{ruleName(instance)}</TableCell>
            <TableCell css={{ flex: 1 }}>
              <StatusTag status={getAlertInstanceStatus(instance.state)} />
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              {formatAlertValueFor(instance, instance.peak_value ?? instance.observed_value)}
            </TableCell>
            <TableCell css={{ flex: 1 }}>{Utils.formatTimestamp(instance.started_at_ms, intl)}</TableCell>
            <TableCell css={{ flex: 1 }}>
              <Button
                componentId={`${CID}.dismiss`}
                size="small"
                disabled={isDismissingAlertInstance}
                // Stop the row's own handler: dismissing is not "tell me more".
                onClick={(e) => {
                  e.stopPropagation();
                  handleDismiss(instance);
                }}
              >
                <FormattedMessage defaultMessage="Dismiss" description="Active alerts table: dismiss button" />
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </Table>
    );
  };

  const renderRules = () => {
    if (rulesLoading) {
      return <TableSkeleton lines={4} />;
    }
    if (alertRules.length === 0) {
      return (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            minHeight: 400,
            width: '100%',
            '& > div': {
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
            },
          }}
        >
          <Empty
            image={<NotificationIcon />}
            title={
              <FormattedMessage
                defaultMessage="No alerts yet"
                description="Alerts page: empty state title when no rules exist"
              />
            }
            description={
              <FormattedMessage
                defaultMessage="An alert watches one metric over a rolling window and notifies you once when it crosses your threshold."
                description="Alerts page: empty state description when no rules exist"
              />
            }
            button={
              <Button
                componentId={`${CID}.empty-state-new-alert`}
                type="primary"
                icon={<PlusIcon />}
                onClick={() => setCreateOpen(true)}
              >
                <FormattedMessage
                  defaultMessage="New alert"
                  description="Alerts page: empty state button to create an alert"
                />
              </Button>
            }
          />
        </div>
      );
    }
    return (
      <Table>
        <TableRow isHeader>
          <TableHeader componentId={`${CID}.rules.name`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Name" description="Alert rules table: name column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.rules.condition`} css={{ flex: 3 }}>
            <FormattedMessage defaultMessage="Condition" description="Alert rules table: condition column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.rules.severity`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Severity" description="Alert rules table: severity column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.rules.status`} css={{ flex: 1 }}>
            <FormattedMessage defaultMessage="Status" description="Alert rules table: status column" />
          </TableHeader>
          <TableHeader componentId={`${CID}.rules.actions`} css={{ flex: 2 }}>
            <FormattedMessage defaultMessage="Actions" description="Alert rules table: actions column" />
          </TableHeader>
        </TableRow>
        {alertRules.map((rule) => (
          <TableRow key={rule.alert_rule_id}>
            <TableCell css={{ flex: 1 }}>{rule.name}</TableCell>
            <TableCell css={{ flex: 3 }}>
              <Typography.Text color="secondary">{describeAlertRule(rule, intl)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>{rule.severity}</TableCell>
            <TableCell css={{ flex: 1 }}>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <StatusTag status={getAlertRuleStatus(rule, openInstances)} />
                {/* The numbers, not just the verdict: a rule held back by its own
                    minimum is fixable, but only if the page says which minimum. */}
                {describeSampleShortfall(rule, intl) && (
                  <Typography.Text size="sm" color="secondary">
                    {describeSampleShortfall(rule, intl)}
                  </Typography.Text>
                )}
              </div>
            </TableCell>
            <TableCell css={{ flex: 2 }}>
              <div css={{ display: 'flex', gap: theme.spacing.sm }}>
                <Button componentId={`${CID}.edit-rule`} size="small" onClick={() => setEditingRule(rule)}>
                  <FormattedMessage defaultMessage="Edit" description="Alert rules table: edit button" />
                </Button>
                <Button
                  componentId={`${CID}.toggle-rule-enabled`}
                  size="small"
                  disabled={isUpdatingAlertRule}
                  onClick={() => handleToggleEnabled(rule)}
                >
                  {rule.enabled ? (
                    <FormattedMessage
                      defaultMessage="Disable"
                      description="Alert rules table: button that stops a rule being evaluated"
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="Enable"
                      description="Alert rules table: button that resumes evaluating a rule"
                    />
                  )}
                </Button>
                <Button componentId={`${CID}.delete-rule`} size="small" danger onClick={() => handleDelete(rule)}>
                  <FormattedMessage defaultMessage="Delete" description="Alert rules table: delete button" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </Table>
    );
  };

  const recentActivity = history.slice(0, 10);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: '100%',
        // The page scrolls; each section's heading sticks as its own rows pass
        // under it. Pinning all three sections to one viewport instead looked
        // right at 1080px and fell apart at 620px, where each list was squeezed
        // to about two rows.
        overflowY: 'auto',
        paddingRight: theme.spacing.md,
        paddingBottom: theme.spacing.md,
      }}
    >
      <Section
        title={
          <FormattedMessage defaultMessage="Active alerts" description="Alerts page: active alerts section title" />
        }
        action={
          <Button
            componentId={`${CID}.new-alert`}
            type="primary"
            icon={<PlusIcon />}
            onClick={() => setCreateOpen(true)}
          >
            <FormattedMessage defaultMessage="New alert" description="Alerts page: button to create an alert" />
          </Button>
        }
      >
        {renderActiveAlerts()}
      </Section>

      <Section title={<FormattedMessage defaultMessage="Rules" description="Alerts page: rules section title" />}>
        {renderRules()}
      </Section>

      {recentActivity.length > 0 && (
        <Section
          title={
            <FormattedMessage
              defaultMessage="Recent activity"
              description="Alerts page: recent activity section title"
            />
          }
        >
          <Table>
            <TableRow isHeader>
              <TableHeader componentId={`${CID}.activity.rule`} css={{ flex: 2 }}>
                <FormattedMessage defaultMessage="Rule" description="Recent activity table: rule column" />
              </TableHeader>
              <TableHeader componentId={`${CID}.activity.state`} css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="State" description="Recent activity table: state column" />
              </TableHeader>
              <TableHeader componentId={`${CID}.activity.started`} css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Started" description="Recent activity table: start time column" />
              </TableHeader>
              <TableHeader componentId={`${CID}.activity.closed-by`} css={{ flex: 1 }}>
                <FormattedMessage defaultMessage="Closed by" description="Recent activity table: closed-by column" />
              </TableHeader>
            </TableRow>
            {recentActivity.map((instance) => (
              <TableRow
                key={instance.alert_instance_id}
                css={{ cursor: 'pointer' }}
                onClick={() => setOpenedInstance(instance)}
              >
                <TableCell css={{ flex: 2 }}>{ruleName(instance)}</TableCell>
                <TableCell css={{ flex: 1 }}>
                  {/* A tag like every other list, rather than the raw enum. */}
                  <StatusTag status={getAlertInstanceStatus(instance.state)} />
                </TableCell>
                <TableCell css={{ flex: 1 }}>{Utils.formatTimestamp(instance.started_at_ms, intl)}</TableCell>
                <TableCell css={{ flex: 1 }}>{instance.dismissed_by ?? '—'}</TableCell>
              </TableRow>
            ))}
          </Table>
        </Section>
      )}

      {openedInstance && (
        <AlertDetailDrawer
          instance={openedInstance}
          rule={ruleById.get(openedInstance.alert_rule_id)}
          onClose={() => setOpenedInstance(undefined)}
          onDismiss={(instance) => {
            handleDismiss(instance);
            setOpenedInstance(undefined);
          }}
          onEditRule={(rule) => {
            setOpenedInstance(undefined);
            setEditingRule(rule);
          }}
        />
      )}

      {/* Keyed so the form remounts with fresh state when the target changes;
          every field is seeded from the rule at mount. */}
      {(createOpen || editingRule) && experimentId && (
        <CreateAlertModal
          key={editingRule?.alert_rule_id ?? 'new'}
          experimentId={experimentId}
          rule={editingRule}
          onClose={() => {
            setCreateOpen(false);
            setEditingRule(undefined);
          }}
        />
      )}
    </div>
  );
};

export default AlertsPage;
