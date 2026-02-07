"""Perception modules for GNN-based collaborative sensing."""

from swarm.perception.graph_builder import (
    GraphData,
    build_radius_graph,
    build_knn_graph,
    apply_packet_dropout,
    CurriculumGraphBuilder,
)
from swarm.perception.gnn import (
    SwarmGNN,
    MessagePassingLayer,
    AttentionMessagePassing,
    create_gnn,
    init_gnn,
)
from swarm.perception.encoders import (
    StateEncoder,
    RelativeEncoder,
    ObservationEncoder,
)

__all__ = [
    "GraphData",
    "build_radius_graph",
    "build_knn_graph",
    "apply_packet_dropout",
    "CurriculumGraphBuilder",
    "SwarmGNN",
    "MessagePassingLayer",
    "AttentionMessagePassing",
    "create_gnn",
    "init_gnn",
    "StateEncoder",
    "RelativeEncoder",
    "ObservationEncoder",
]
