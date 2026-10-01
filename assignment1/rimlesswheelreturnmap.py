import numpy as np
import matplotlib.pyplot as plt

from integrators import rk4
from models import rimlesswheel
from pathlib import Path



#RK4 step
def rk4_step(t, state, dt, params):

    time_traj = np.array([t, t + dt])

    state_traj = np.zeros((2, 2))
    state_traj[:, 0] = state

    state_traj = rk4.integrate(
        time_traj,
        state_traj,
        dt,
        rimlesswheel,
        params,
    )

    return state_traj[:, 1]


#Find impact time and state using bisection
def find_impact(t, state, dt, params, tolerance=1e-8):

    left_time = 0.0
    right_time = dt

    while right_time - left_time > tolerance:

        middle_time = (left_time + right_time) / 2

        middle_state = rk4_step(
            t,
            state,
            middle_time,
            params,
        )

        guard = rimlesswheel.impact_guard(
            middle_state,
            params,
        )

        if guard < 0:
            left_time = middle_time
        else:
            right_time = middle_time

    impact_dt = right_time

    impact_state = rk4_step(
        t,
        state,
        impact_dt,
        params,
    )

    return impact_dt, impact_state


#Return map for the rimless wheel
def return_map(velocity, params):

    dt = 0.005
    max_time = 5.0

    #Start immediately after an impact
    state = np.array([
        -params["alpha"],
        velocity,
    ])

    t = 0.0

    while t < max_time:

        next_state = rk4_step(
            t,
            state,
            dt,
            params,
        )

        old_guard = rimlesswheel.impact_guard(
            state,
            params,
        )

        new_guard = rimlesswheel.impact_guard(
            next_state,
            params,
        )

        #Next foot contact
        if old_guard < 0 and new_guard >= 0:

            impact_dt, impact_state = find_impact(
                t,
                state,
                dt,
                params,
            )

            post_impact_state = rimlesswheel.calculate_impact(
                t + impact_dt,
                impact_state,
                params,
            )

            return post_impact_state[1]

        state = next_state
        t += dt

    # No next impact
    return np.nan

#Parameters for the rimless wheel
params = {
    "gravity": 9.81,
    "length": 1.0,
    "alpha": 0.3,
    "impact_angle": 0.3,
    "gamma": -0.05,
}


#Return map sweep
velocity_values = np.linspace(
    0.1,
    4.0,
    200,
)

next_velocity_values = np.zeros_like(
    velocity_values
)

for i, velocity in enumerate(velocity_values):

    next_velocity_values[i] = return_map(
        velocity,
        params,
    )


#Approximate fixed point by finding the closest point to the identity line
valid = ~np.isnan(next_velocity_values)

difference = np.abs(
    next_velocity_values[valid]
    - velocity_values[valid]
)

fixed_index = np.argmin(difference)

fixed_velocity = velocity_values[valid][
    fixed_index
]

fixed_next_velocity = next_velocity_values[valid][
    fixed_index
]

print(
    "Approximate fixed point:",
    fixed_velocity,
    "rad/s"
)



plt.figure()

plt.plot(
    velocity_values,
    next_velocity_values,
    label="Return map",
)

#Identity line
plt.plot(
    velocity_values,
    velocity_values,
    "--",
    label="Identity line",
)

#Mark the approximate fixed point
plt.plot(
    fixed_velocity,
    fixed_next_velocity,
    "o",
    label=f"Fixed point = {fixed_velocity:.3f} rad/s",
)

plt.xlabel(
    r"$\dot{\theta}_k$ [rad/s]"
)

plt.ylabel(
    r"$\dot{\theta}_{k+1}$ [rad/s]"
)

plt.title(
    rf"Step-to-Step Return Map, $\gamma={params['gamma']}$ rad"
)

plt.grid()
plt.legend()

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(figures_dir / "returnmap.png", dpi=300, bbox_inches="tight")

plt.show()

#Estimate the Floquet multiplier by finite differences

epsilon = 0.01

velocity_minus = fixed_velocity - epsilon
velocity_plus = fixed_velocity + epsilon

next_velocity_minus = return_map(
    velocity_minus,
    params,
)

next_velocity_plus = return_map(
    velocity_plus,
    params,
)

floquet_multiplier = (
    next_velocity_plus
    - next_velocity_minus
) / (
    velocity_plus
    - velocity_minus
)

print(
    "Fixed point:",
    fixed_velocity,
    "rad/s"
)

print(
    "Lower perturbation:",
    velocity_minus,
    "->",
    next_velocity_minus
)

print(
    "Upper perturbation:",
    velocity_plus,
    "->",
    next_velocity_plus
)

print(
    "Floquet multiplier:",
    floquet_multiplier
)

