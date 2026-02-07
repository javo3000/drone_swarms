"""Communication modules for realistic network modeling."""

from swarm.comms.degradation import (
    CommsConfig,
    apply_range_dropout,
    apply_random_dropout,
    apply_jammer_zones,
    RealisticCommsModel,
    CurriculumCommsModel,
)

__all__ = [
    "CommsConfig",
    "apply_range_dropout",
    "apply_random_dropout",
    "apply_jammer_zones",
    "RealisticCommsModel",
    "CurriculumCommsModel",
]
