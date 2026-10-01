import numpy as np
import matplotlib.pyplot as plt


from integrators import rk4
from models import rimlesswheel
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path



#RK4 step
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


#Find impact time and state using bisection
def find_impact(t, state, dt, params, tolerance=1e-8):

    left_time = 0.0
    right_time = dt

    # Bisection on the impact time
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


#Simulate the rimless wheel dynamics and check for convergence
def simulate(initial_state, params):

    dt = 0.005
    max_time = 20.0

    state = np.array(initial_state, dtype=float)

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

        #Impact detection
        if old_guard < 0 and new_guard >= 0:

            impact_dt, impact_state = find_impact(
                t,
                state,
                dt,
                params,
            )

            # Apply impact/reset
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

            #Check for convergence of post-impact velocities
            if len(post_impact_velocities) >= 6:

                recent = np.array(
                    post_impact_velocities[-6:]
                )

                if np.max(
                    np.abs(np.diff(recent))
                ) < 1e-5:

                    return 1

        else:

            state = next_state
            t += dt

       
        if t - last_impact_time > 5.0:
            return 0

        # Catch runaway numerical / physical failures
        if np.any(np.abs(state) > 100):
            return 0

    return 0


#Parameters for the rimless wheel
params = {
    "gravity": 9.81,
    "length": 1.0,

    "alpha": 0.3,
    "impact_angle": 0.3,

    "gamma": -0.05,
}


#Initial grid conditions
theta_values = np.linspace(
    -params["alpha"],
    params["alpha"] - 1e-4,
    60,
)

velocity_values = np.linspace(
    -2.0,
    4.0,
    60,
)

roa = np.zeros(
    (
        len(velocity_values),
        len(theta_values),
    )
)


#Brute force sweep of initial conditions
for i, velocity in enumerate(velocity_values):

    print(
        f"Row {i + 1}/{len(velocity_values)}"
    )

    for j, theta in enumerate(theta_values):

        initial_state = np.array([
            theta,
            velocity,
        ])

        roa[i, j] = simulate(
            initial_state,
            params,
        )


#Plot the region of attraction
cmap = ListedColormap(["red", "green"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

plt.figure()

image = plt.imshow(
    roa,
    origin="lower",
    extent=[
        theta_values[0],
        theta_values[-1],
        velocity_values[0],
        velocity_values[-1],
    ],
    aspect="auto",
    cmap=cmap,
    norm=norm,
)

plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")

plt.title(
    rf"Region of Attraction, $\gamma={params['gamma']}$ rad"
)

#Discrete colorbar with labels
colorbar = plt.colorbar(
    image,
    ticks=[0, 1],
    boundaries=[-0.5, 0.5, 1.5],
)

colorbar.ax.set_yticklabels([
    "Failure",
    "Stable walking",
])

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(figures_dir / "roa.png", dpi=300, bbox_inches="tight")

plt.show()