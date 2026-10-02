def explicit_euler(dynamics, t, state, timestep, params):
    state_derivative = dynamics(t, state, params)
    next_state = state + timestep * state_derivative
    return next_state

def rk4(dynamics, t, state, timestep, params):
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + 0.5 * timestep, state + 0.5 * timestep * k1, params)
    k3 = dynamics(t + 0.5 * timestep, state + 0.5 * timestep * k2, params)
    k4 = dynamics(t + timestep, state + timestep * k3, params)

    next_state = state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state