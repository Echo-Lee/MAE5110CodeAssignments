# AI-assisted: test structure and the dE/dt derivation drafted with Claude.

import numpy as np

from integrators import rk4 as integrator
from models import pendulum as model

TIMESTEP = 0.01  # s
N_STEPS = 100


def simulate(initial_state, params):
    """Integrate N_STEPS with RK4 and return the (2, N_STEPS + 1) trajectory."""
    state_traj = np.zeros((2, N_STEPS + 1))
    state_traj[:, 0] = initial_state
    for step in range(N_STEPS):
        state_traj[:, step + 1] = integrator(
            model.dynamics, step * TIMESTEP, state_traj[:, step], TIMESTEP, params
        )
    return state_traj


def calculate_total_energy(state_traj, params):
    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    return kinetic_energy + potential_energy


def test_energy_is_conserved_without_torque_or_damping():
    params = model.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0
    initial_state = np.array([1.0, 0.0])  # away from both equilibria (0 and pi)

    total_energy = calculate_total_energy(simulate(initial_state, params), params)

    assert np.all(np.isclose(total_energy, total_energy[0]))


def test_torque_balancing_gravity_holds_pendulum_still():
    params = model.generate_params()
    params["damping_coeff"] = 0.0
    held_angle = 0.5  # rad
    # Cancel the gravity term m g l sin(theta), so angular acceleration is zero.
    params["torque"] = (
        -params["mass"] * params["gravity"] * params["length"] * np.sin(held_angle)
    )
    initial_state = np.array([held_angle, 0.0])

    state_traj = simulate(initial_state, params)

    assert np.all(np.isclose(state_traj[0], held_angle))
    assert np.all(np.isclose(state_traj[1], 0.0))


def test_damping_only_removes_energy():
    params = model.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.5  # kg m^2/s
    initial_state = np.array([1.0, 0.0])

    total_energy = calculate_total_energy(simulate(initial_state, params), params)

    assert np.all(np.diff(total_energy) <= 1e-9)  # never gains energy
    assert total_energy[-1] < total_energy[0]  # and loses some overall
