from models import bouncing_ball as model

def explicit_euler(t, state, timestep, params):
    state_derivative = model.dynamics(t, state, params)
    next_state = state + timestep * state_derivative
    next_state = model.handle_collision(next_state, params)
    return next_state

def rk4(t, state, timestep, params):
    k1 = model.dynamics(t, state, params)
    k2 = model.dynamics(t + 0.5 * timestep, state + 0.5 * timestep * k1, params)
    k3 = model.dynamics(t + 0.5 * timestep, state + 0.5 * timestep * k2, params)
    k4 = model.dynamics(t + timestep, state + timestep * k3, params)

    next_state = state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    next_state = model.handle_collision(next_state, params)
    return next_state