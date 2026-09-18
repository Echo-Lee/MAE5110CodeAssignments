"""Plotting helpers for assignment analyses."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def save_standing_roa_plot(theta_values, omega_values, roa_map, output_file, data_file=None):
    """Save a region-of-attraction plot and, optionally, its grid data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(theta_values, omega_values, roa_map, shading="auto")
    ax.set(
        xlabel=r"$\theta_0$",
        ylabel=r"$\dot{\theta}_0$",
        title="Ankle Controller Region of Attraction",
    )
    ax.plot(0, 0, "kx", label="Upright equilibrium")
    ax.legend(
        handles=[
            Patch(facecolor="yellow", label="Inside RoA"),
            Patch(facecolor="purple", label="Outside RoA"),
            *ax.get_legend_handles_labels()[0],
        ],
        loc="upper right",
    )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    if data_file is not None:
        np.savez(
            data_file,
            theta_values=theta_values,
            omega_values=omega_values,
            roa_map=roa_map,
        )
    return fig


def save_grid_resolution_plot(n_omega_values, mean_errors, p95_errors, output_file):
    """Save the lookup-table state-grid resolution diagnostic."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_omega_values, mean_errors, "o-", label="Mean normalized error")
    ax.plot(n_omega_values, p95_errors, "s-", label="95th percentile normalized error")
    ax.set(
        xlabel="Number of omega grid points",
        ylabel="Normalized nearest-state error (%)",
        title="Poincare State Grid Resolution",
    )
    ax.grid(True)
    ax.legend()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    return fig


def save_policy_trajectory_plot(
    fastest_trajectory,
    fastest_crossings,
    longest_trajectory,
    longest_crossings,
    fastest_steps,
    longest_steps,
    output_file,
):
    """Save continuous phase-plane paths for fastest and longest walking plans."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.plot(*fastest_trajectory, color="C0", label=f"fastest policy ({fastest_steps} steps)")
    ax.plot(*longest_trajectory, color="C3", label=f"longest policy ({longest_steps} steps)")
    ax.plot(
        fastest_trajectory[0, fastest_crossings], fastest_trajectory[1, fastest_crossings], "o", color="C0"
    )
    ax.plot(
        longest_trajectory[0, longest_crossings], longest_trajectory[1, longest_crossings], "o", color="C3"
    )
    ax.plot(0, 0, "k*", ms=10, label="standing equilibrium")
    ax.set(xlabel=r"$\theta$ (rad)", ylabel=r"$\dot\theta$ (rad/s)", title="Phase-plane trajectories to standing")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.savefig(output_file, dpi=300)
    return fig


def save_steps_to_standstill_plot(omega_values, steps, chosen_index, output_file):
    """Save the lookup-table estimate of steps needed to enter the RoA."""
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    finite_steps = np.where(np.isfinite(steps), steps, np.nan)
    scatter = ax.scatter(omega_values, finite_steps, c=finite_steps, cmap="viridis", s=35)
    ax.scatter(omega_values[chosen_index], steps[chosen_index], color="crimson", marker="*", s=130)
    ax.set(
        xlabel=r"Initial $\dot\theta$ at $\theta=0$ (rad/s)",
        ylabel="Minimum steps to standing",
        title="Steps required to reach the standing RoA",
    )
    ax.set_yticks(np.arange(1, int(np.nanmax(finite_steps)) + 1))
    ax.grid(True)
    fig.colorbar(scatter, ax=ax, label="Steps")
    fig.savefig(output_file, dpi=300)
    return fig
