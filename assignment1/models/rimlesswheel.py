import numpy as np
def dynamics(t, state, params):
    theta, velocity = state
    gravity = params["gravity"]
    length = params["length"]
    gamma = params["gamma"]
    acceleration = (gravity / length) *np.sin(theta - gamma)
    state_derivative = np.array([velocity, acceleration])
    return state_derivative


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