from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator
from controllers.ankle_controller import ankle_controller
from analysis.standing_roa import is_in_standing_roa_trial, roa_event_guard
from analysis.plotting import (
    save_grid_resolution_plot,
    save_policy_trajectory_plot,
    save_standing_roa_plot,
    save_steps_to_standstill_plot,
)
from analysis.policy_trajectory import simulate_planned_trajectory
from analysis.return_map import (
    action_plan,
    build_lookup_table,
    compute_lookup_errors,
    compute_steps_to_roa,
    longest_action_plan,
)


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

def simulation_step(t, state, params, timestep):
    next_state = integrator(t, state, model.dynamics, timestep, params)
    impacted = False

    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        impacted = True

    return next_state, impacted


def run_one_simulation(initial_state, params, timestep=1e-4, sim_time=3.0, desired_number_of_steps=3, roa_data=None):
    n_timesteps = round(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    completed_steps = 0

    torque_min = -0.1 * params["mass"] * params["gravity"] * params["length"]
    torque_max = 0.05 * params["mass"] * params["gravity"] * params["length"]

    # Simulation loop.
    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        # calculate ankle torque
        # If we have RoA map, and the state is in the map, then we turn on the controller
        controller_on = roa_event_guard(state, roa_data) if roa_data is not None else True

        # enforce torque bounds
        params["ankle_torque"] = np.clip(ankle_controller(state, params), torque_min, torque_max) if controller_on else 0.0

        next_state, impacted = simulation_step(t, state, params, timestep)
        if impacted:
            completed_steps += 1

        state_traj[:, step + 1] = next_state
        if completed_steps == desired_number_of_steps:
            break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]
    return time_traj, state_traj, completed_steps

# Draw initial RoA plot
theta_values = np.linspace(-0.3, 0.4, 50)
omega_values = np.linspace(-1.5, 1.5, 50)

roa_map = np.zeros((len(omega_values), len(theta_values)), dtype=bool)

for i, omega0 in enumerate(omega_values):
    for j, theta0 in enumerate(theta_values):
        initial_state = np.array([theta0, omega0])
        params_exp = params.copy()
        time_traj, state_traj, completed_steps = run_one_simulation(initial_state, params_exp)
    
        roa_map[i, j] = is_in_standing_roa_trial(state_traj, completed_steps)


save_standing_roa_plot(
    theta_values,
    omega_values,
    roa_map,
    output / "standing_roa.png",
    output / "standing_roa_data.npz",
)
plt.show()


data = np.load(
    output / "standing_roa_data.npz"
)

roa_data = {
    "theta_values": data["theta_values"],
    "omega_values": data["omega_values"],
    "roa_map": data["roa_map"],
}


# Choose theta = 0 as poincare, build look up table for different initial condition (theta_dot, alpha)

def return_function(omega_k, alpha_k, params, timestep=1e-4, sim_time=3.0, roa_data=None):
    # under initial poincare state, return the next omega when passing the vertical
    # Initial condition of alpha_k, omega_k
    params["angle_of_attack"] = alpha_k
    params["ankle_torque"] = 0.0
    state = np.array([0.0, omega_k])
    n_timesteps = round(sim_time / timestep)

    for step in range(n_timesteps):
        t = step * timestep

        if roa_data is not None and roa_event_guard(state, roa_data):
            return None, True # omega_next, reached_RoA

        next_state, _ = simulation_step(t, state, params, timestep)
        if model.event_transverse_guard(state, next_state, params):
            return next_state[1], False

        state = next_state

    return None, False


def evaluate_grid_resolution(n_omega_list, n_alpha, params, roa_data):
    omega_min = 0.0
    omega_max = np.sqrt(2 * params["gravity"] / params["length"])
    omega_range = omega_max - omega_min

    mean_error_percents = []
    p95_error_percents = []

    for n_omega in n_omega_list:
        omega_values, alpha_values, next_omega_table, reach_roa_table = build_lookup_table(
            n_omega, n_alpha, params, roa_data, return_function
        )

        errors = compute_lookup_errors(omega_values, next_omega_table)

        mean_error = np.mean(errors)
        p95_error = np.percentile(errors, 95)

        mean_error_percent = mean_error / omega_range * 100
        p95_error_percent = p95_error / omega_range * 100

        mean_error_percents.append(mean_error_percent)
        p95_error_percents.append(p95_error_percent)

        print(f"n_omega={n_omega}: mean error={mean_error:.4f} rad/s ({mean_error_percent:.2f}%), 95th={p95_error:.4f} rad/s ({p95_error_percent:.2f}%)")

    return np.array(mean_error_percents), np.array(p95_error_percents)

n_omega_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_alpha = 20

mean_error_percents, p95_error_percents = evaluate_grid_resolution(n_omega_list, n_alpha, params, roa_data)

save_grid_resolution_plot(
    n_omega_list,
    mean_error_percents,
    p95_error_percents,
    output / "grid_resolution_test.png",
)
plt.show()

# Use the best resolution to generate lookup table
n_omega = 100
n_alpha = 20

omega_values, alpha_values, next_omega_table, reach_roa_table = build_lookup_table(
    n_omega, n_alpha, params, roa_data, return_function
)

np.savez(
    output / "poincare_lookup_table.npz",
    omega_values=omega_values,
    alpha_values=alpha_values,
    next_omega_table=next_omega_table,
    reach_roa_table=reach_roa_table,
)


# Compute the minimum steps neede to enter RoA for each omega
steps_to_roa, fastest_action_by_state, next_state_index = compute_steps_to_roa(
    omega_values, next_omega_table, reach_roa_table
)
# Find the starting point we need
# This grid state gives distinct 3-step and 4-step plans, both of which
# converge under a continuous-time verification run.
initial_index = 87
# calculate fastest plan
fastest_plan = action_plan(
    initial_index,
    fastest_action_by_state,
    next_state_index,
    reach_roa_table,
)
# calculate longest plan
longest_steps, longest_plan = longest_action_plan(
    initial_index,
    next_state_index,
    next_omega_table,
    reach_roa_table,
)

longest_steps = int(longest_steps + 1)
initial_omega = omega_values[initial_index]

# simulating the fastest and longest trajectories
fastest_trajectory, fastest_crossings = simulate_planned_trajectory(
    initial_omega,
    alpha_values[fastest_plan],
    params,
    roa_data,
)

longest_trajectory, longest_crossings = simulate_planned_trajectory(
    initial_omega,
    alpha_values[longest_plan],
    params,
    roa_data,
)

save_policy_trajectory_plot(
    fastest_trajectory,
    fastest_crossings,
    longest_trajectory,
    longest_crossings,
    int(steps_to_roa[initial_index]),
    longest_steps,
    output / "three_step_trajectory.png",
)

save_steps_to_standstill_plot(
    omega_values, steps_to_roa, initial_index, output / "steps_to_standstill.png"
)
