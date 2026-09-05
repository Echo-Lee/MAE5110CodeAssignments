import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from assignment_1 import analyze, params


# =========================
# Settings
# =========================

save_dir = "./sweep_figures"
os.makedirs(save_dir, exist_ok=True)

theta_points = 50
velocity_points = 70

# Fixed colors:
# 0 = Limit Cycle
# 1 = Backward
# 2 = Other
cmap = ListedColormap([
    "purple",
    "teal",
    "gold"
])

norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5],
    cmap.N
)


# =========================
# Sweep Slope
# =========================

slope_values = [0.05, 0.075, 0.10, 0.125, 0.15]

slope_results = []
slope_floquets = []

for gamma in slope_values:

    test_params = params.copy()
    test_params["slope"] = gamma

    (
        theta_grid,
        velocity_grid,
        roa,
        return_map_x,
        return_map_y,
        omega_star,
        floquet,
    ) = analyze(
        test_params,
        theta_points=theta_points,
        velocity_points=velocity_points
    )

    slope_results.append(
        (theta_grid, velocity_grid, roa)
    )

    slope_floquets.append(floquet)

    print(
        f"gamma={gamma:.3f}, "
        f"omega*={omega_star:.4f}, "
        f"Floquet={floquet:.4f}"
    )

    # Individual RoA figure
    plt.figure(figsize=(8, 6))

    plt.imshow(
        roa,
        origin="lower",
        extent=[
            theta_grid[0],
            theta_grid[-1],
            velocity_grid[0],
            velocity_grid[-1]
        ],
        aspect="auto",
        cmap=cmap,
        norm=norm
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\dot{\theta}_0$")
    plt.title(fr"RoA, $\gamma={gamma:.3f}$")

    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([
        "Limit Cycle",
        "Backward",
        "Other"
    ])

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"roa_slope_{gamma:.3f}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# Combined Slope RoA
# =========================

fig, axes = plt.subplots(
    1,
    len(slope_values),
    figsize=(18, 4),
    sharey=True
)

for ax, gamma, result in zip(
    axes,
    slope_values,
    slope_results
):

    theta_grid, velocity_grid, roa = result

    im = ax.imshow(
        roa,
        origin="lower",
        extent=[
            theta_grid[0],
            theta_grid[-1],
            velocity_grid[0],
            velocity_grid[-1]
        ],
        aspect="auto",
        cmap=cmap,
        norm=norm
    )

    ax.set_title(fr"$\gamma={gamma:.3f}$")
    ax.set_xlabel(r"$\theta_0$")

axes[0].set_ylabel(r"$\dot{\theta}_0$")

cbar = fig.colorbar(
    im,
    ax=axes,
    ticks=[0, 1, 2],
    fraction=0.02,
    pad=0.02
)

cbar.ax.set_yticklabels([
    "Limit Cycle",
    "Backward",
    "Other"
])

fig.savefig(
    os.path.join(
        save_dir,
        "slope_roa_combined.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# Slope vs Floquet
# =========================

plt.figure(figsize=(6, 4))

plt.plot(
    slope_values,
    slope_floquets,
    "o-"
)

plt.xlabel(r"Slope $\gamma$")
plt.ylabel("Floquet Multiplier")
plt.title("Effect of Slope on Local Convergence")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        save_dir,
        "slope_vs_floquet.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# Sweep Number of Spokes
# =========================

N_values = list(range(6, 13))

N_results = []
N_floquets = []

for N in N_values:

    test_params = params.copy()
    test_params["spoke_number"] = N

    (
        theta_grid,
        velocity_grid,
        roa,
        return_map_x,
        return_map_y,
        omega_star,
        floquet,
    ) = analyze(
        test_params,
        theta_points=theta_points,
        velocity_points=velocity_points
    )

    N_results.append(
        (theta_grid, velocity_grid, roa)
    )

    N_floquets.append(floquet)

    print(
        f"N={N}, "
        f"omega*={omega_star:.4f}, "
        f"Floquet={floquet:.4f}"
    )

    # Individual RoA figure
    plt.figure(figsize=(8, 6))

    plt.imshow(
        roa,
        origin="lower",
        extent=[
            theta_grid[0],
            theta_grid[-1],
            velocity_grid[0],
            velocity_grid[-1]
        ],
        aspect="auto",
        cmap=cmap,
        norm=norm
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\dot{\theta}_0$")
    plt.title(fr"RoA, $N={N}$")

    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([
        "Limit Cycle",
        "Backward",
        "Other"
    ])

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"roa_spokes_{N}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =========================
# Combined Spoke RoA
# =========================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 8),
    sharey=True
)

axes = axes.flatten()

for ax, N, result in zip(
    axes,
    N_values,
    N_results
):

    theta_grid, velocity_grid, roa = result

    im = ax.imshow(
        roa,
        origin="lower",
        extent=[
            theta_grid[0],
            theta_grid[-1],
            velocity_grid[0],
            velocity_grid[-1]
        ],
        aspect="auto",
        cmap=cmap,
        norm=norm
    )

    ax.set_title(fr"$N={N}$")
    ax.set_xlabel(r"$\theta_0$")

for ax in axes[::4]:
    ax.set_ylabel(r"$\dot{\theta}_0$")

# Last subplot unused
axes[-1].axis("off")

cbar = fig.colorbar(
    im,
    ax=axes,
    ticks=[0, 1, 2],
    fraction=0.02,
    pad=0.02
)

cbar.ax.set_yticklabels([
    "Limit Cycle",
    "Backward",
    "Other"
])

fig.savefig(
    os.path.join(
        save_dir,
        "spokes_roa_combined.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# Number of Spokes vs Floquet
# =========================

plt.figure(figsize=(6, 4))

plt.plot(
    N_values,
    N_floquets,
    "o-"
)

plt.xlabel("Number of Spokes")
plt.ylabel("Floquet Multiplier")
plt.title("Effect of Number of Spokes on Local Convergence")
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        save_dir,
        "spokes_vs_floquet.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()