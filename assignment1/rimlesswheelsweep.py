import numpy as np
import matplotlib.pyplot as plt

from integrators import rk4
from models import rimlesswheel
from pathlib import Path


#rk4 step
def rk4_step(t, state, dt, params):

    time_traj = np.array([
        t,
        t + dt
    ])

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


#find impact time and state using bisection
def find_impact(t, state, dt, params, tolerance=1e-8):

    left_time = 0.0
    right_time = dt

    while right_time - left_time > tolerance:

        middle_time = (
            left_time + right_time
        ) / 2

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


#simulate initial conditition 
def simulate(initial_state, params):

    dt = 0.005
    max_time = 20.0

    state = np.array(
        initial_state,
        dtype=float
    )

    t = 0.0

    post_impact_velocities = []

    last_impact_time = 0.0

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

        if old_guard < 0 and new_guard >= 0:

            impact_dt, impact_state = find_impact(
                t,
                state,
                dt,
                params,
            )

            state = rimlesswheel.calculate_impact(
                t + impact_dt,
                impact_state,
                params,
            )

            t += impact_dt

            post_impact_velocities.append(
                state[1]
            )

            last_impact_time = t

            # Check convergence to walking gait
            if len(post_impact_velocities) >= 6:

                recent = np.array(
                    post_impact_velocities[-6:]
                )

                if np.max(
                    np.abs(
                        np.diff(recent)
                    )
                ) < 1e-5:

                    return 1

        else:

            state = next_state
            t += dt

        # No impact for too long
        if t - last_impact_time > 5.0:
            return 0

        # Catch runaway states
        if np.any(
            np.abs(state) > 100
        ):
            return 0

    return 0


#return map
def return_map(velocity, params):

    dt = 0.005
    max_time = 5.0

    # State immediately after impact
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

        if old_guard < 0 and new_guard >= 0:

            impact_dt, impact_state = find_impact(
                t,
                state,
                dt,
                params,
            )

            post_impact_state = (
                rimlesswheel.calculate_impact(
                    t + impact_dt,
                    impact_state,
                    params,
                )
            )

            return post_impact_state[1]

        state = next_state
        t += dt

    return np.nan


#ROA fraction
def calculate_roa_fraction(params):

    theta_values = np.linspace(
        -params["alpha"],
        params["alpha"] - 1e-4,
        40,
    )

    velocity_values = np.linspace(
        -2.0,
        4.0,
        40,
    )

    successes = 0
    total = 0

    for velocity in velocity_values:

        for theta in theta_values:

            initial_state = np.array([
                theta,
                velocity,
            ])

            result = simulate(
                initial_state,
                params,
            )

            successes += result
            total += 1

    return successes / total


#fixed point finder
def find_fixed_point(params):

    velocities = np.linspace(
        0.1,
        5.0,
        200,
    )

    previous_velocity = None
    previous_difference = None

    for velocity in velocities:

        next_velocity = return_map(
            velocity,
            params,
        )

        if np.isnan(next_velocity):
            continue

        difference = (
            next_velocity - velocity
        )

        if previous_difference is not None:

            if (
                difference
                * previous_difference
                < 0
            ):

                left = previous_velocity
                right = velocity

                # Refine using bisection
                for _ in range(30):

                    middle = (
                        left + right
                    ) / 2

                    middle_next = return_map(
                        middle,
                        params,
                    )

                    middle_difference = (
                        middle_next - middle
                    )

                    left_next = return_map(
                        left,
                        params,
                    )

                    left_difference = (
                        left_next - left
                    )

                    if (
                        left_difference
                        * middle_difference
                        <= 0
                    ):
                        right = middle
                    else:
                        left = middle

                return (
                    left + right
                ) / 2

        previous_velocity = velocity
        previous_difference = difference

    return np.nan


#Floquet multiplier
def calculate_floquet(params):

    fixed_velocity = find_fixed_point(
        params
    )

    if np.isnan(fixed_velocity):
        return np.nan, np.nan

    epsilon = 0.01

    velocity_minus = (
        fixed_velocity - epsilon
    )

    velocity_plus = (
        fixed_velocity + epsilon
    )

    next_minus = return_map(
        velocity_minus,
        params,
    )

    next_plus = return_map(
        velocity_plus,
        params,
    )

    floquet = (
        next_plus - next_minus
    ) / (
        velocity_plus - velocity_minus
    )

    return fixed_velocity, floquet


#parameters for the rimless wheel
base_params = {
    "gravity": 9.81,
    "length": 1.0,
    "alpha": 0.3,
    "impact_angle": 0.3,
    "gamma": -0.05,
}


#Gamma sweep

gamma_degrees = np.arange(5, 46, 5)
gamma_values = -np.radians(gamma_degrees)

gamma_roa = []
gamma_floquet = []

for gamma in gamma_values:

    print(
        f"Testing gamma: {gamma:.3f} rad"
    )

    params = base_params.copy()

    params["gamma"] = gamma

    roa_fraction = calculate_roa_fraction(
        params
    )

    fixed_point, floquet = (
        calculate_floquet(
            params
        )
    )

    gamma_roa.append(
        roa_fraction
    )

    gamma_floquet.append(
        floquet
    )

    print(
        "RoA fraction:",
        roa_fraction
    )

    print(
        "Fixed point:",
        fixed_point
    )

    print(
        "Floquet multiplier:",
        floquet
    )


#Plot RoA vs gamma
plt.figure()

plt.plot(
    gamma_values,
    gamma_roa,
    "o-",
)

plt.xlabel(
    r"$\gamma$ [rad]"
)

plt.ylabel(
    "Fraction of state space that walks"
)

plt.title(
    "Region of Attraction vs Inclination"
)

plt.grid()

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(figures_dir / "gammavsroa.png", dpi=300, bbox_inches="tight")

plt.show()


#Plot floquet vs gamma
plt.figure()

plt.plot(
    gamma_values,
    gamma_floquet,
    "o-",
)

plt.axhline(
    1,
    linestyle="--",
)

plt.axhline(
    -1,
    linestyle="--",
)

plt.xlabel(
    r"$\gamma$ [rad]"
)

plt.ylabel(
    r"Floquet multiplier $\lambda$"
)

plt.title(
    "Floquet Multiplier vs Inclination"
)

plt.grid()

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(figures_dir / "gammavsfloquet.png", dpi=300, bbox_inches="tight")

plt.show()


#Sweep number of spokes

spoke_values = np.arange(
    6,
    13,
)

spoke_roa = []
spoke_floquet = []

for number_spokes in spoke_values:

    print(
        "Testing spokes:",
        number_spokes
    )

    params = base_params.copy()

    alpha = (
        np.pi / number_spokes
    )

    params["alpha"] = alpha

    params["impact_angle"] = alpha

    params["gamma"] = -0.05

    roa_fraction = calculate_roa_fraction(
        params
    )

    fixed_point, floquet = (
        calculate_floquet(
            params
        )
    )

    spoke_roa.append(
        roa_fraction
    )

    spoke_floquet.append(
        floquet
    )

    print(
        "RoA fraction:",
        roa_fraction
    )

    print(
        "Fixed point:",
        fixed_point
    )

    print(
        "Floquet multiplier:",
        floquet
    )


#Plot ROA vs spokes
plt.figure()

plt.plot(
    spoke_values,
    spoke_roa,
    "o-",
)

plt.xlabel(
    "Number of spokes"
)

plt.ylabel(
    "Fraction of state space that walks"
)

plt.title(
    "Region of Attraction vs Number of Spokes"
)

plt.grid()

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(figures_dir / "spokesvsroa.png", dpi=300, bbox_inches="tight")

plt.show()


#Plot floquet vs spokes
plt.figure()

plt.plot(
    spoke_values,
    spoke_floquet,
    "o-",
)

plt.axhline(
    1,
    linestyle="--",
)

plt.axhline(
    -1,
    linestyle="--",
)

plt.xlabel(
    "Number of spokes"
)

plt.ylabel(
    r"Floquet multiplier $\lambda$"
)

plt.title(
    "Floquet Multiplier vs Number of Spokes"
)

plt.grid()

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(figures_dir / "spokesvsfloquet.png", dpi=300, bbox_inches="tight")

plt.show()