import numpy as np

def ankle_controller(state, params):
    theta, omega = state

    m = params["mass"]
    g = params["gravity"]
    l = params["length"]

    k_theta = -5.0
    k_omega = -5.0

    torque = (
        -m * g * l * np.sin(theta)
        + k_theta * theta
        + k_omega * omega
    )

    return torque