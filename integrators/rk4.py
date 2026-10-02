import numpy as np


def rk4(dynamics, t, state, timestep, params):
    k1 = dynamics(t, state, params)
    k2 = dynamics(
        t + timestep / 2,
        state + timestep * k1 / 2,
        params,
    )
    k3 = dynamics(
        t + timestep / 2,
        state + timestep * k2 / 2,
        params,
    )
    k4 = dynamics(
        t + timestep,
        state + timestep * k3,
        params,
    )

    state = state + timestep * (
        k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    )

    return state
