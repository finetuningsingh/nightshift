"""Shared exception types for Nightshift."""


class NightshiftError(Exception):
    """Base class for Nightshift errors."""


class ConfigError(NightshiftError):
    """Configuration is missing, malformed, or inconsistent."""


class LedgerError(NightshiftError):
    """The event ledger is missing or unreadable."""


class BudgetError(NightshiftError):
    """Budget-store state is inconsistent."""


class ReservationRejected(BudgetError):
    """A spend or rental reservation was refused before any provider call."""


class ResourceError(NightshiftError):
    """A resource operation was refused, for example an unowned rental."""


class ProviderError(NightshiftError):
    """A compute provider failed or is unavailable."""


class DataError(NightshiftError):
    """Data or manifest validation failed."""


class PolicyError(NightshiftError):
    """A pack, mount, privacy, or checkpoint policy check failed."""
