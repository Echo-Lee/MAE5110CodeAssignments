import numpy as np

def dynamics(t, state, params):
    # It is the same as pendulum (inverted)
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity  # <-- DAMPING TERM
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative

def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at center (kg)
        "spoke_number": 8, # number of spokes
        "slope": np.pi / 10, # slope
        "damping_coeff": 0.1,  # damping coefficient (kg*m^2/s)
    }
    return params

def detect_impact(state, params):
    # When theta > alpha + gamma, it is indicated that a spoke has contacted the slope
    angle, angular_velocity = state[0], state[1]
    gamma = params["slope"]
    alpha = np.pi / params["spoke_number"]

    if angle >= gamma + alpha and angular_velocity > 0:
        # forward
        return 1

    elif angle <= gamma - alpha and angular_velocity < 0:
        return -1

    return 0

def reset_state(state, params):
    
    gamma = params["slope"]
    alpha = np.pi / params["spoke_number"]

    state[0] = gamma - alpha
    state[1] *= np.cos(2*alpha)
    return state
