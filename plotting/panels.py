"""Named plotting adapters for all four experiment families.

These aliases expose one discoverable plotting surface without inserting an
extra call between a legacy kernel and Matplotlib. That matters because figure
compatibility includes artist insertion order, not only colors and dimensions.
"""

from traceesus.experiments.counterfactual.kernel import (
    plot_primary_figure as plot_counterfactual_primary,
)
from traceesus.experiments.endotype_discovery.kernel import (
    plot_control_figure as plot_endotype_controls,
    plot_example_patient as plot_endotype_example_patient,
    plot_primary_figure as plot_endotype_recovery,
)
from traceesus.experiments.model_comparison.kernel import (
    plot_comparison as plot_model_comparison,
)
from traceesus.experiments.transportability.kernel import (
    plot_ablation_figure as plot_transportability_ablation,
    plot_transport_controls as plot_transportability_controls,
    plot_transport_figure as plot_transportability_primary,
)

__all__ = [
    "plot_counterfactual_primary",
    "plot_endotype_controls",
    "plot_endotype_example_patient",
    "plot_endotype_recovery",
    "plot_model_comparison",
    "plot_transportability_ablation",
    "plot_transportability_controls",
    "plot_transportability_primary",
]
