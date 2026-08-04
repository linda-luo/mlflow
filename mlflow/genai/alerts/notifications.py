"""Notification channels for alert instances.

In-app only in v1: :class:`InAppChannel` does nothing, because the committed
``alert_instances`` row *is* the notification. The registry exists now so a
third party can add a channel without forking, mirroring
``ScorerStoreRegistry``.

**Durability comes from ordering** -- commit the instance, then dispatch. There
is deliberately no ``alert_notifications`` delivery table: with only an in-app
channel it would have nothing to track, and once a channel that can actually
fail arrives, adding one is a purely additive migration.
"""

import logging
import warnings
from abc import ABC, abstractmethod

from mlflow.exceptions import MlflowException
from mlflow.genai.alerts.entities import AlertInstance, AlertRule
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.plugins import get_entry_points

_logger = logging.getLogger(__name__)

IN_APP_CHANNEL_TYPE = "in_app"


class NotificationChannel(ABC):
    """Where an alert goes when it fires.

    ``send`` is called once per *transition*, never once per breaching
    evaluation: a three-hour incident produces one notification, not one every
    interval. The state machine owns that decision; a channel only has to
    deliver what it is handed.
    """

    @abstractmethod
    def send(self, rule: AlertRule, instance: AlertInstance) -> None:
        """Deliver one notification.

        Implementations must not raise for a delivery failure that a retry
        could fix -- the instance is already committed, and the dispatcher
        treats an exception as "this channel failed", not "the alert failed".
        """


class InAppChannel(NotificationChannel):
    """The default channel: a no-op.

    The alert's row in ``alert_instances`` is what the Alerts tab reads, so it
    is already the notification by the time this is called. Writing a second
    record here would only create something else that can disagree with it.
    """

    def send(self, rule: AlertRule, instance: AlertInstance) -> None:
        _logger.debug(
            "Alert rule %s fired instance %s (in-app; no delivery required)",
            rule.alert_rule_id,
            instance.alert_instance_id,
        )


class NotificationChannelRegistry:
    """Type-based registry of notification channels.

    Channels declared through the ``mlflow.alert_notification_channel``
    entry-point group are registered automatically by
    :meth:`register_entrypoints`, so a third party can add Slack or PagerDuty
    without forking MLflow.
    """

    def __init__(self):
        self._registry: dict[str, NotificationChannel] = {}
        self.group_name = "mlflow.alert_notification_channel"

    def register(self, channel_type: str, channel: NotificationChannel) -> None:
        self._registry[channel_type] = channel

    def register_entrypoints(self) -> None:
        """Register channels provided by other packages."""
        for entrypoint in get_entry_points(self.group_name):
            try:
                self.register(entrypoint.name, entrypoint.load()())
            except (AttributeError, ImportError, TypeError) as exc:
                warnings.warn(
                    f'Failure attempting to register notification channel "{entrypoint.name}": '
                    f"{exc}",
                    stacklevel=2,
                )

    def get_channel(self, channel_type: str) -> NotificationChannel:
        channel = self._registry.get(channel_type)
        if channel is None:
            raise MlflowException(
                f"Unknown notification channel type '{channel_type}'. Registered types: "
                f"{sorted(self._registry)}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return channel

    def registered_types(self) -> list[str]:
        return sorted(self._registry)


_registry = NotificationChannelRegistry()
_registry.register(IN_APP_CHANNEL_TYPE, InAppChannel())
_registry.register_entrypoints()


def get_notification_channel_registry() -> NotificationChannelRegistry:
    return _registry


def dispatch(rule: AlertRule, instance: AlertInstance) -> None:
    """Send one notification per configured channel.

    Call this *after* the instance is committed. A channel that raises is
    logged and skipped rather than propagated: the alert has already been
    recorded, and failing the evaluation cycle over an undeliverable Slack
    message would lose the next cycle's work too.

    A rule with no configured channels still notifies in-app, which is the
    only channel that cannot fail.
    """
    channels = rule.channels or [{"type": IN_APP_CHANNEL_TYPE}]
    for entry in channels:
        channel_type = entry.get("type", IN_APP_CHANNEL_TYPE)
        try:
            _registry.get_channel(channel_type).send(rule, instance)
        except Exception:
            _logger.exception(
                "Notification channel '%s' failed for alert instance %s; the instance is "
                "already recorded and remains visible in the Alerts tab.",
                channel_type,
                instance.alert_instance_id,
            )
