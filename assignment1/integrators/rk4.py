import numpy as np
def integrate(time_traj, state_traj, timestep, model, params):
    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]

        k1 = model.dynamics(t, state, params)
        k2 = model.dynamics(
            t + timestep / 2,
            state + timestep * k1 / 2,
            params,
        )
        k3 = model.dynamics(
            t + timestep / 2,
            state + timestep * k2 / 2,
            params,
        )
        k4 = model.dynamics(
            t + timestep,
            state + timestep * k3,
            params,
        )

        state_traj[:, step + 1] = state + timestep * (
            k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        )

    return state_traj