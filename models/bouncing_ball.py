import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    height = state[0]
    speed = state[1] # positive when upwards

    acceleration = -gravity - damping_coeff / mass * speed

    state_derivative = np.array([speed, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0.1,  # damping coefficient (kg*m^2/s)
        "restitution_coeff": 0.8,  # coefficient of restitution (dimensionless)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    speed = state[1]

    kinetic_energy = 0.5 * mass * speed ** 2
    potential_energy = mass * gravity * height
    return kinetic_energy, potential_energy

def handle_collision(state, params):
    restitution_coeff = params["restitution_coeff"]
    height = state[0]
    speed = state[1]

    if height < 0:
        speed = -speed * restitution_coeff
        height = 0  # Reset height to ground level

    return np.array([height, speed])
