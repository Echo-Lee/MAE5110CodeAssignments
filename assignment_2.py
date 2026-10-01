from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from models import inverted_pendulum_walker as model
from integrators import rk4

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}

def ankle_controller(state, params, kp, kd):
    theta, angular_velocity = state

    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]

    torque = (-mass * gravity * length * np.sin(theta) -mass * length**2 * (kp * theta + kd * angular_velocity))

    torque_min = -0.1 * mass * gravity * length
    torque_max = 0.05 * mass * gravity * length

    torque = np.clip(torque, torque_min, torque_max)

    return torque

# Controller gains
kp = 5.0
kd = 2.0

def rk4_step(t, state, timestep, model, params):
    time_traj = np.array([t, t + timestep])

    state_traj = np.zeros((2, 2))
    state_traj[:, 0] = state

    state_traj = rk4.integrate(
        time_traj,
        state_traj,
        timestep,
        model,
        params,
    )

    return state_traj[:, 1]

def simulate_one_step(initial_velocity, alpha, params, timestep=1e-3):
    state = np.array([0.0, initial_velocity])

    params["angle_of_attack"] = alpha
    params["ankle_torque"] = 0.0

    impact_occurred = False
    max_steps = 10000

    for step in range(max_steps):
        t = step * timestep

        next_state = rk4_step(
            t,
            state,
            timestep,
            model,
            params,
        )

        # Detect foot strike and apply impact dynamics
        if model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(
                next_state,
                params,
            )
            impact_occurred = True

        # After impact, detect next theta = 0 crossing
        elif (
            impact_occurred
            and state[0] < 0 <= next_state[0]
            and next_state[1] > 0
        ):
            return next_state[1]

        state = next_state

    # Did not successfully return to the Poincare section
    return np.nan

def compute_return_map(velocity_grid, alpha_grid, params):
    table = np.full(
        (len(velocity_grid), len(alpha_grid)),
        np.nan,
    )

    for i, angular_velocity in enumerate(velocity_grid):
        for j, alpha in enumerate(alpha_grid):
            table[i, j] = simulate_one_step(
                angular_velocity,
                alpha,
                params,
            )

    return table
#Testing one-step Poincare map; removed after testing was succesful
#test_velocity = 2.0
#test_alpha = np.pi / 8

#next_velocity = simulate_one_step(test_velocity, test_alpha, params,)

#print("Initial velocity:", test_velocity)
#print("Alpha:", test_alpha)
#print("Next velocity:", next_velocity)
##Results:
#Initial velocity: 2.0
#Alpha: 0.39269908169872414
#Next velocity: 1.3834668067010358

def test_stabilization(initial_state, params, kp, kd):
    state = initial_state.copy()

    timestep = 1e-4
    test_time = 3.0
    n_steps = int(test_time / timestep)

    for _ in range(n_steps):

        # Turn the ankle controller on for the stabilization test
        params["ankle_torque"] = ankle_controller(
            state, params, kp, kd
        )

        # rk4 integration
        state = rk4_step(0.0, state, timestep, model, params)

        # If the pendulum moves too far from upright,
        # consider this initial condition a failure.
        if abs(state[0]) > np.pi / 2:
            params["ankle_torque"] = 0.0
            return False

    # Check whether the pendulum finished close to upright
    stabilized = (
        abs(state[0]) < 0.01
        and abs(state[1]) < 0.05
    )

    # Reset torque after the test
    params["ankle_torque"] = 0.0

    return stabilized

# Grid of initial states around the upright equilibrium
theta_values = np.linspace(-np.pi / 7, np.pi / 7, 21)
velocity_values = np.linspace(-1.0, 1.0, 21)

# True = ankle controller can stabilize this initial condition
roa = np.zeros(
    (len(velocity_values), len(theta_values)),
    dtype=bool,
)

print("Computing region of attraction...")

for i, angular_velocity in enumerate(velocity_values):
    for j, theta in enumerate(theta_values):

        initial_state = np.array([
            theta,
            angular_velocity,
        ])

        roa[i, j] = test_stabilization(
            initial_state,
            params,
            kp,
            kd,
        )

print("Finished computing region of attraction.")


plt.figure(figsize=(7, 5))

# Red = failure, Green = successfully stabilized
roa_cmap = ListedColormap(["red", "green"])

plt.imshow(
    roa,
    origin="lower",
    extent=[
        theta_values[0],
        theta_values[-1],
        velocity_values[0],
        velocity_values[-1],
    ],
    aspect="auto",
    cmap=roa_cmap,
)

plt.xlabel("Theta (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Region of Attraction of Ankle Controller")
plt.savefig(output / "RoA.png", dpi=300, bbox_inches="tight")

plt.show()


def roa_event_guard(state, theta_values, velocity_values, roa):
    theta, angular_velocity = state

    # State must be inside the bounds of the grid
    if (
        theta < theta_values[0]
        or theta > theta_values[-1]
        or angular_velocity < velocity_values[0]
        or angular_velocity > velocity_values[-1]
    ):
        return False

    # Find the nearest point in the RoA grid
    theta_index = np.argmin(
        np.abs(theta_values - theta)
    )

    velocity_index = np.argmin(
        np.abs(velocity_values - angular_velocity)
    )

    return roa[velocity_index, theta_index]

# State-action grid for the step-to-step dynamics

gravity = params["gravity"]
length = params["length"]

# Maximum angular velocity corresponding to Froude number = 2
max_velocity = np.sqrt(2 * gravity / length)

# Grid of current angular velocities
poincare_velocity_values = np.linspace(
    0.0,
    max_velocity,
    21,
)

# Grid of allowable angle-of-attack controls
alpha_values = np.linspace(
    np.pi / 8,
    np.pi / 7,
    21,
)

# Table containing the next angular velocity
return_map = np.full(
    (len(poincare_velocity_values), len(alpha_values)),
    np.nan,
)

print("Computing step-to-step lookup table...")

for i, angular_velocity in enumerate(poincare_velocity_values):
    for j, alpha in enumerate(alpha_values):

        return_map[i, j] = simulate_one_step(
            angular_velocity,
            alpha,
            params,
        )

print("Finished computing step-to-step lookup table.")


# Test lookup-table grid resolution using nearest-neighbor error

grid_sizes = [11, 21, 31, 41]
mean_errors = []

for grid_size in grid_sizes:

    velocity_grid = np.linspace(
        0.0,
        max_velocity,
        grid_size,
    )

    alpha_grid = np.linspace(
        np.pi / 8,
        np.pi / 7,
        grid_size,
    )

    test_map = compute_return_map(
        velocity_grid,
        alpha_grid,
        params,
    )

    errors = []

    # Test velocities halfway between stored grid points
    for i in range(len(velocity_grid) - 1):

        test_velocity = (
            velocity_grid[i] + velocity_grid[i + 1]
        ) / 2

        # Find closest stored velocity
        nearest_i = np.argmin(
            np.abs(velocity_grid - test_velocity)
        )

        for j, alpha in enumerate(alpha_grid):

            # Prediction from lookup table
            predicted_velocity = test_map[
                nearest_i,
                j,
            ]

            # Actual simulation at the state between grid points
            actual_velocity = simulate_one_step(
                test_velocity,
                alpha,
                params,
            )

            if (
                not np.isnan(predicted_velocity)
                and not np.isnan(actual_velocity)
            ):
                errors.append(
                    abs(
                        actual_velocity
                        - predicted_velocity
                    )
                )

    mean_error = np.mean(errors)
    mean_errors.append(mean_error)

    print(
        f"Grid {grid_size}x{grid_size}: "
        f"mean nearest-neighbor error = "
        f"{mean_error:.6f} rad/s"
    )


plt.figure(figsize=(7, 5))

plt.plot(
    grid_sizes,
    mean_errors,
    "o-",
)

plt.axhline(
    0.05,
    linestyle="--",
    label="0.05 rad/s tolerance",
)

plt.xlabel("Grid size")
plt.ylabel("Mean nearest-neighbor error (rad/s)")
plt.title("Lookup Table Grid Convergence")
plt.legend()
plt.grid()

plt.savefig(output / "Grid Convergence.png", dpi=300, bbox_inches="tight")
plt.show()

# Select coarsest acceptable state-action grid


tolerance = 0.05
selected_grid_size = None

for grid_size, error in zip(grid_sizes, mean_errors):
    if error <= tolerance:
        selected_grid_size = grid_size
        break

print("Selected grid size:", selected_grid_size)


# Rebuild the lookup table using selected grid size

poincare_velocity_values = np.linspace(
    0.0,
    max_velocity,
    selected_grid_size,
)

alpha_values = np.linspace(
    np.pi / 8,
    np.pi / 7,
    selected_grid_size,
)

return_map = compute_return_map(
    poincare_velocity_values,
    alpha_values,
    params,
)


# Find Poincare states already inside standing-controller RoA


standing_states = np.zeros(
    len(poincare_velocity_values),
    dtype=bool,
)

for i, angular_velocity in enumerate(poincare_velocity_values):

    state = np.array([
        0.0,
        angular_velocity,
    ])

    standing_states[i] = roa_event_guard(
        state,
        theta_values,
        velocity_values,
        roa,
    )


# Backward search


steps_to_stand = np.full(
    len(poincare_velocity_values),
    -1,
    dtype=int,
)

best_action = np.full(
    len(poincare_velocity_values),
    np.nan,
)

# 0 means already inside standing-controller RoA
steps_to_stand[standing_states] = 0

max_steps = 20

for step_count in range(1, max_steps + 1):

    found_new_state = False

    for i in range(len(poincare_velocity_values)):

        # Already know how to stabilize this state
        if steps_to_stand[i] != -1:
            continue

        for j in range(len(alpha_values)):

            next_velocity = return_map[i, j]

            if np.isnan(next_velocity):
                continue

            # Find closest state on our velocity grid
            next_i = np.argmin(
                np.abs(
                    poincare_velocity_values
                    - next_velocity
                )
            )

            # Does this action move us one step closer?
            if steps_to_stand[next_i] == step_count - 1:

                steps_to_stand[i] = step_count
                best_action[i] = alpha_values[j]

                found_new_state = True
                break

    if not found_new_state:
        break

for i in range(len(poincare_velocity_values)):

    if steps_to_stand[i] == -1:

        print(
            "Unreachable velocity:",
            poincare_velocity_values[i]
        )

        print(
            "Next velocities:",
            return_map[i, :]
        )

# Compare minimum-step and maximum-step trajectories


# Pick an initial condition that requires at least 3 steps
initial_velocity = 3.0

initial_index = np.argmin(
    np.abs(poincare_velocity_values - initial_velocity)
)

print(
    "Initial velocity:",
    poincare_velocity_values[initial_index],
)

print(
    "Minimum steps to RoA:",
    steps_to_stand[initial_index],
)


# Find longest paths that still eventually reach the RoA


max_steps_before_roa = np.full(
    len(poincare_velocity_values),
    -1,
    dtype=int,
)

max_action = np.full(
    len(poincare_velocity_values),
    np.nan,
)

# States already in the RoA need zero steps
max_steps_before_roa[standing_states] = 0

# Search for longer paths
for iteration in range(100):

    changed = False

    for i in range(len(poincare_velocity_values)):

        # Skip states already in the RoA
        if standing_states[i]:
            continue

        for j in range(len(alpha_values)):

            next_velocity = return_map[i, j]

            if np.isnan(next_velocity):
                continue

            # Find nearest velocity grid point
            next_i = np.argmin(
                np.abs(
                    poincare_velocity_values
                    - next_velocity
                )
            )

            # Next state must already have a known path to RoA
            if max_steps_before_roa[next_i] >= 0:

                candidate_steps = (
                    1 + max_steps_before_roa[next_i]
                )

                if (
                    candidate_steps
                    > max_steps_before_roa[i]
                ):

                    max_steps_before_roa[i] = (
                        candidate_steps
                    )

                    max_action[i] = alpha_values[j]

                    changed = True

    if not changed:
        break


print(
    "Maximum steps to RoA:",
    max_steps_before_roa[initial_index],
)

# Minimum-step trajectory


min_trajectory = [
    poincare_velocity_values[initial_index]
]

current_i = initial_index

while steps_to_stand[current_i] > 0:

    alpha = best_action[current_i]

    j = np.argmin(
        np.abs(alpha_values - alpha)
    )

    next_velocity = return_map[current_i, j]

    current_i = np.argmin(
        np.abs(
            poincare_velocity_values
            - next_velocity
        )
    )

    min_trajectory.append(
        poincare_velocity_values[current_i]
    )


# Maximum-step trajectory


max_trajectory = [
    poincare_velocity_values[initial_index]
]

current_i = initial_index

for _ in range(100):

    # Stop when RoA is reached
    if standing_states[current_i]:
        break

    alpha = max_action[current_i]

    if np.isnan(alpha):
        break

    j = np.argmin(
        np.abs(alpha_values - alpha)
    )

    next_velocity = return_map[current_i, j]

    current_i = np.argmin(
        np.abs(
            poincare_velocity_values
            - next_velocity
        )
    )

    max_trajectory.append(
        poincare_velocity_values[current_i]
    )


# Plot minimum and maximum-step trajectories


plt.figure(figsize=(8, 5))

plt.plot(
    range(len(min_trajectory)),
    min_trajectory,
    "o-",
    label="Minimum-step trajectory",
)

plt.plot(
    range(len(max_trajectory)),
    max_trajectory,
    "o-",
    label="Maximum-step trajectory",
)

plt.xlabel("Step number")
plt.ylabel("Angular velocity (rad/s)")

plt.title(
    "Minimum and Maximum-Step Trajectories"
)

plt.legend()
plt.grid()

plt.savefig(output / "Min Max Trajectories.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot number of footsteps required to reach standing

reachable = steps_to_stand >= 0
unreachable = steps_to_stand == -1

plt.figure(figsize=(8, 5))

plt.scatter(
    poincare_velocity_values[reachable],
    steps_to_stand[reachable],
    label="Reachable",
)

plt.scatter(
    poincare_velocity_values[unreachable],
    np.zeros(np.sum(unreachable)),
    marker="x",
    s=100,
    label="No valid step found",
)

plt.xlabel("Angular velocity (rad/s)")
plt.ylabel("Steps to standing")
plt.title("Steps Required to Reach Standing Controller RoA")
plt.legend()
plt.grid()

plt.savefig(output / "Steps Required.png", dpi=300, bbox_inches="tight")
plt.show()

# Walker simulation / animation

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3


n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

# Simulation loop
reached_roa = False
impact_occurred = False

# Choose the first foot-placement action
initial_velocity = initial_state[1]

velocity_index = np.argmin(
    np.abs(poincare_velocity_values - initial_velocity)
)

if steps_to_stand[velocity_index] != -1:
    params["angle_of_attack"] = best_action[velocity_index]
else:
    print("Initial velocity is unreachable.")

for step, t in enumerate(time_traj[:-1]):

    state = state_traj[:, step]

    # Check if walker has entered the RoA
    if not reached_roa:

        if roa_event_guard(
            state,
            theta_values,
            velocity_values,
            roa,
        ):
            reached_roa = True

            print(
                f"Entered RoA at t = {t:.3f} s, "
                f"theta = {state[0]:.3f}, "
                f"velocity = {state[1]:.3f}"
            )

    # Ankle controller stays on after entering RoA
    if reached_roa:

        params["ankle_torque"] = ankle_controller(
            state,
            params,
            kp,
            kd,
        )

    else:

        params["ankle_torque"] = 0.0

    # Integrate one timestep
    next_state = rk4_step(
        t,
        state,
        timestep,
        model,
        params,
    )

    # Detect foot strike
    if (
        not reached_roa
        and model.event_guard(state, next_state, params)
    ):

        next_state = model.event_dynamics(
            next_state,
            params,
        )

        completed_steps += 1
        impact_occurred = True

    # Detect return to Poincare section
    elif (
        not reached_roa
        and impact_occurred
        and state[0] < 0 <= next_state[0]
        and next_state[1] > 0
    ):

        current_velocity = next_state[1]

        velocity_index = np.argmin(
            np.abs(
                poincare_velocity_values
                - current_velocity
            )
        )

        if steps_to_stand[velocity_index] > 0:

            params["angle_of_attack"] = (
                best_action[velocity_index]
            )

            print(
                f"Velocity = {current_velocity:.3f}, "
                f"alpha = {params['angle_of_attack']:.3f}, "
                f"steps remaining = "
                f"{steps_to_stand[velocity_index]}"
            )

        impact_occurred = False

    # IMPORTANT: store next state every timestep
    state_traj[:, step + 1] = next_state


# These stay OUTSIDE the for loop
time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)

animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()
