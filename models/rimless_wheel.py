import numpy as np

def generate_params():
    """Using base_params"""
    params = {
    "gravity": 9.81,
    "length": 1.0,
    "alpha": 0.3,
    "impact_angle": 0.3,
    "gamma": -0.05,
}

    return params


def dynamics(t, state, params):
    theta, velocity = state
    gravity = params["gravity"]
    length = params["length"]
    gamma = params["gamma"]
    acceleration = (gravity / length) *np.sin(theta - gamma)
    state_derivative = np.array([velocity, acceleration])
    return state_derivative

def generate_initial_condition():
    """Set an initial condition of (theta, angular_velocity) = (0.01, 0.01)"""
    return np.array([0.01, 0.01])

def impact_guard(state, params):
    theta, velocity = state
    return theta - params["impact_angle"]

def calculate_impact(t, state, params):
    theta, velocity = state
    alpha = params["alpha"]
    newtheta = theta - 2 *alpha
    newvelocity = velocity *np.cos(2*alpha)
    state = np.array([newtheta, newvelocity])
    return state