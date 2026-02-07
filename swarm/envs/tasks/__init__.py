"""Tasks module for swarm environments."""

from swarm.envs.tasks.coverage import (
    compute_coverage_reward,
    compute_voronoi_targets,
)
from swarm.envs.tasks.threat_response import (
    ThreatState,
    ThreatConfig,
    spawn_threats,
    update_threats,
    compute_threat_reward,
    ForbiddenZone,
)
from swarm.envs.tasks.gps_denied import (
    GPSJammer,
    GPSDeniedTask,
    apply_gps_jamming,
    create_gps_denied_scenario,
)

__all__ = [
    "compute_coverage_reward",
    "compute_voronoi_targets",
    "ThreatState",
    "ThreatConfig",
    "spawn_threats",
    "update_threats",
    "compute_threat_reward",
    "ForbiddenZone",
    "GPSJammer",
    "GPSDeniedTask",
    "apply_gps_jamming",
    "create_gps_denied_scenario",
]
