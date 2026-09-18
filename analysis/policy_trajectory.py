"""Continuous phase-plane trajectories for planned walker footfalls."""

import numpy as np

from analysis.standing_roa import roa_event_guard
from controllers.ankle_controller import ankle_controller
from integrators import rk4
from models import inverted_pendulum_walker as model


def _advance_state(time, state, params, timestep):
    next_state = rk4(time, state, model.dynamics, timestep, params)
    if model.event_guard(state, next_state, params):
        return model.event_dynamics(next_state, params)
    return next_state


def simulate_planned_trajectory(initial_omega, alpha_plan, params, roa_data, timestep=1e-4):
    """Follow planned footfalls, then balance after the state enters the RoA."""
    params = params.copy()
    state = np.array([0.0, initial_omega])
    states = [state.copy()]
    crossing_indices = [0]
    time = 0.0

    for alpha in alpha_plan:
        params["angle_of_attack"] = alpha
        params["ankle_torque"] = 0.0
        for _ in range(round(3.0 / timestep)):
            if roa_event_guard(state, roa_data):
                break
            next_state = _advance_state(time, state, params, timestep)
            states.append(next_state.copy())
            time += timestep
            if model.event_transverse_guard(state, next_state, params):
                crossing_indices.append(len(states) - 1)
                state = next_state
                break
            state = next_state
        if roa_event_guard(state, roa_data):
            break

    torque_min = -0.1 * params["mass"] * params["gravity"] * params["length"]
    torque_max = 0.05 * params["mass"] * params["gravity"] * params["length"]
    for _ in range(round(2.0 / timestep)):
        params["ankle_torque"] = np.clip(
            ankle_controller(state, params), torque_min, torque_max
        )
        state = _advance_state(time, state, params, timestep)
        states.append(state.copy())
        time += timestep

    return np.asarray(states).T, crossing_indices
