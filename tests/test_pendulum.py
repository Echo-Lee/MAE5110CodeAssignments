import numpy as np

from models import pendulum as model
from integrators import rk4 as integrator

def test_energy_constant():
    params = model.generate_params()
    params["damping_coeff"], params["torque"] = 0.0, 0.0

    state = np.array([0.1, 0.1])
    timestep = 0.01

    start_energy = np.sum(model.calculate_energy(state, params))

    for step in range(100):
        state = integrator(model.dynamics, step * timestep, state, timestep, params)
    
    end_energy = np.sum(model.calculate_energy(state, params))

    assert np.isclose(start_energy, end_energy)


def test_torque():
    params = model.generate_params()
    params["gravity"] = 0.0
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0
    state = np.array([0.0, 0.0])
    timestep = 0.01
    steps = 100
    final_time = steps * timestep

    for step in range(steps):
        state = integrator(model.dynamics, step * timestep, state, timestep, params)

    inertia = params["mass"] * params["length"] ** 2
    angular_acceleration = params["torque"] / inertia
    # With gravity and damping removed, theta_ddot = torque / inertia is constant.
    # Thus theta(t) = theta(0) + theta_dot(0)t + 0.5 * theta_ddot * t**2
    # and theta_dot(t) = theta_dot(0) + theta_ddot * t.
    expected_state = np.array(
        [0.5 * angular_acceleration * final_time**2, angular_acceleration * final_time]
    )

    assert np.allclose(state, expected_state, rtol=1e-6, atol=1e-8)


def test_damping():
    params = model.generate_params()
    params["gravity"] = 0.0
    params["torque"] = 0.0
    params["damping_coeff"] = 0.5
    initial_angular_velocity = 2.0
    state = np.array([0.0, initial_angular_velocity])
    timestep = 0.01
    steps = 100
    final_time = steps * timestep

    for step in range(steps):
        state = integrator(model.dynamics, step * timestep, state, timestep, params)

    inertia = params["mass"] * params["length"] ** 2
    decay_rate = params["damping_coeff"] / inertia
    # With gravity and torque removed, theta_ddot = -decay_rate * theta_dot.
    # The analytic solution is theta_dot(t) = theta_dot(0) * exp(-decay_rate * t)
    # and theta(t) = theta(0) + theta_dot(0) / decay_rate
    # * (1 - exp(-decay_rate * t)).
    expected_state = np.array(
        [
            initial_angular_velocity / decay_rate * (1 - np.exp(-decay_rate * final_time)),
            initial_angular_velocity * np.exp(-decay_rate * final_time),
        ]
    )

    assert np.allclose(state, expected_state, rtol=1e-6, atol=1e-8)
