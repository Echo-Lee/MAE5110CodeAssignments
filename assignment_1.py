import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import rk4 as integrator


params = {
    "gravity": 9.81,
    "length": 1,
    "mass": 1,
    "spoke_number": 8,
    "slope": 0.1,
    "damping_coeff": 0,
}

alpha = np.pi / params["spoke_number"]
gamma = params["slope"]

theta_grid = np.linspace(gamma - alpha, gamma + alpha, 80)
velocity_grid = np.linspace(-2.0, 2.0, 100)

timestep = 5e-4
sim_time = 20.0

roa = np.zeros((len(velocity_grid), len(theta_grid)), dtype=int)

return_map_x = []
return_map_y = []
fixed_point_estimates = []

attractor_map = {
    "limit_cycle": 0,
    "backward": 1,
    "other": 2,
}


def is_limit_cycle(impact_velocities, tol=1e-2, n_check=5):

    if len(impact_velocities) < n_check + 1:
        return False

    recent = impact_velocities[-(n_check + 1):]
    differences = np.abs(np.diff(recent))

    return np.all(differences < tol)


def simulate(initial_state, params, timestep=1e-5, sim_time=5.0):

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    impact_velocities = []
    is_backward_contact = False

    for step, t in enumerate(time_traj[:-1]):

        state_traj[:, step + 1] = integrator(
            t, state_traj[:, step], model.dynamics, timestep, params
        )

        contact = model.detect_impact(
            state_traj[:, step + 1], params
        )

        if contact == 1:

            state_traj[:, step + 1] = model.reset_state(
                state_traj[:, step + 1], params
            )

            impact_velocities.append(
                state_traj[1, step + 1]
            )

            if is_limit_cycle(
                np.array(impact_velocities)
            ):
                break

        elif contact == -1:

            is_backward_contact = True
            break

    return (
        time_traj[:step + 2],
        state_traj[:, :step + 2],
        np.array(impact_velocities),
        is_backward_contact,
    )


def classify_trajectory(
    impact_velocities,
    is_backward_contact,
    tol=1e-2,
    n_check=5,
):

    if is_backward_contact:
        return "backward"

    if is_limit_cycle(
        impact_velocities,
        tol=tol,
        n_check=n_check,
    ):
        return "limit_cycle"

    return "other"

def analyze(params, theta_points=80, velocity_points=100,
            timestep=5e-4, sim_time=20.0):

    alpha = np.pi / params["spoke_number"]
    gamma = params["slope"]

    theta_grid = np.linspace(
        gamma - alpha,
        gamma + alpha,
        theta_points
    )

    velocity_grid = np.linspace(
        -2.0,
        2.0,
        velocity_points
    )

    roa = np.zeros(
        (len(velocity_grid), len(theta_grid)),
        dtype=int
    )

    return_map_x = []
    return_map_y = []
    fixed_point_estimates = []

    for j, initial_theta in enumerate(theta_grid):
        for i, initial_velocity in enumerate(velocity_grid):

            initial_state = np.array([
                initial_theta,
                initial_velocity
            ])

            _, _, impact_velocities, backward = simulate(
                initial_state, params, timestep, sim_time
            )

            attractor = classify_trajectory(
                impact_velocities, backward
            )

            roa[i, j] = attractor_map[attractor]

            if len(impact_velocities) >= 2:
                return_map_x.extend(impact_velocities[:-1])
                return_map_y.extend(impact_velocities[1:])

            if attractor == "limit_cycle":
                fixed_point_estimates.append(
                    np.mean(impact_velocities[-5:])
                )

    return_map_x = np.array(return_map_x)
    return_map_y = np.array(return_map_y)

    if len(fixed_point_estimates) > 0:

        omega_star = np.median(fixed_point_estimates)

        mask = np.abs(return_map_x - omega_star) < 0.05

        if np.sum(mask) >= 2:
            floquet = np.polyfit(
                return_map_x[mask],
                return_map_y[mask],
                1
            )[0]
        else:
            floquet = np.nan

    else:
        omega_star = np.nan
        floquet = np.nan

    return (
        theta_grid,
        velocity_grid,
        roa,
        return_map_x,
        return_map_y,
        omega_star,
        floquet
    )




# =========================
# RoA + Return Map Sampling
# =========================
if __name__ == "__main__":

    theta_grid, velocity_grid, roa, \
    return_map_x, return_map_y, \
    omega_star, floquet = analyze(params)

    print(
        f"Estimated return-map fixed point: "
        f"{omega_star:.6f} rad/s"
    )

    print(
        f"Floquet multiplier: "
        f"{floquet:.6f}"
    )

    # =========================
    # RoA
    # =========================

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
        aspect="auto"
    )

    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\dot{\theta}_0$")
    plt.title("Region of Attraction")

    cbar = plt.colorbar(ticks=[0, 1, 2])

    cbar.ax.set_yticklabels([
        "Limit Cycle",
        "Backward",
        "Other"
    ])

    plt.tight_layout()

    plt.savefig(
        "rimless_wheel_roa.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # =========================
    # Return Map
    # =========================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        return_map_x,
        return_map_y,
        s=6,
        alpha=0.25,
        label="Return map samples"
    )

    min_val = min(
        return_map_x.min(),
        return_map_y.min()
    )

    max_val = max(
        return_map_x.max(),
        return_map_y.max()
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        label="Identity line"
    )

    plt.scatter(
        [omega_star],
        [omega_star],
        s=100,
        marker="x",
        label=fr"Fixed point $\omega^*={omega_star:.4f}$"
    )

    plt.xlabel(r"$\omega_k^+$")
    plt.ylabel(r"$\omega_{k+1}^+$")
    plt.title("Step-to-Step Return Map")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        "rimless_wheel_return_map.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    