import numpy as np

def is_in_standing_roa_trial(state_traj, completed_steps, theta_tol=1e-2, omega_tol=1e-2, check_last_n=100):
    # if hit the slope, then should not be stable
    if completed_steps > 0:
        return False

    # check whether last n stable
    recent_states = state_traj[:, -check_last_n:]
    theta_recent = recent_states[0, :]
    omega_recent = recent_states[1, :]

    theta_stable = np.all(np.abs(theta_recent) < theta_tol)
    omega_stable = np.all(np.abs(omega_recent) < omega_tol)

    return theta_stable and omega_stable

def roa_event_guard(state, roa_data):
    theta, omega = state

    theta_values = roa_data["theta_values"]
    omega_values = roa_data["omega_values"]
    roa_map = roa_data["roa_map"]

    if theta < theta_values[0] or theta > theta_values[-1]:
        return False

    if omega < omega_values[0] or omega > omega_values[-1]:
        return False

    j = np.argmin(np.abs(theta_values - theta))
    i = np.argmin(np.abs(omega_values - omega))

    return roa_map[i, j]

    