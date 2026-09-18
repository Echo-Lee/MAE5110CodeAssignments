from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator
from controllers.ankle_controller import ankle_controller
from analysis.standing_roa import is_in_standing_roa_trial, roa_event_guard


output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}


def run_one_simulation(initial_state, params, roa_data=None):
    timestep = 1e-4
    sim_time = 3.0
    desired_number_of_steps = 3

    n_timesteps = round(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    completed_steps = 0

    torque_min = -0.1 * params["mass"] * params["gravity"] * params["length"]
    torque_max = 0.05 * params["mass"] * params["gravity"] * params["length"]

    # Simulation loop. Replace this Euler step with your own integrator as needed.
    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        # calculate ankle torque
        if roa_data is None:
            # For parts of "standing RoA"
            controller_on = True
        else:
            # If we have RoA map, and the state is in the map, then we turn on the controller
            controller_on = roa_event_guard(state, roa_data)

        if controller_on:
            torque = ankle_controller(state, params)
            # enforce torque bounds
            params["ankle_torque"] = np.clip(torque, torque_min, torque_max)
        else:
            params["ankle_torque"] = 0.0

        next_state = integrator(t, state, model.dynamics, timestep, params)

        if model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(next_state, params)
            completed_steps += 1

        state_traj[:, step + 1] = next_state
        if completed_steps == desired_number_of_steps:
            break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]
    return time_traj, state_traj, completed_steps

# Draw initial RoA plot
theta_values = np.linspace(-0.3, 0.4, 50)
omega_values = np.linspace(-1.5, 1.5, 50)

roa_map = np.zeros((len(omega_values), len(theta_values)), dtype=bool)

for i, omega0 in enumerate(omega_values):
    for j, theta0 in enumerate(theta_values):
        initial_state = np.array([theta0, omega0])
        params_exp = params.copy()
        time_traj, state_traj, completed_steps = run_one_simulation(initial_state, params_exp)
    
        roa_map[i, j] = is_in_standing_roa_trial(state_traj, completed_steps)


plt.figure(figsize=(8, 6))

mesh = plt.pcolormesh(
    theta_values,
    omega_values,
    roa_map,
    shading="auto"
)

plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\dot{\theta}_0$")
plt.title("Ankle Controller Region of Attraction")

plt.plot(0, 0, "kx", label="Upright equilibrium")

legend_elements = [
    Patch(facecolor="yellow", label="Inside RoA"),
    Patch(facecolor="purple", label="Outside RoA"),
]

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles=legend_elements + handles,
    loc="upper right"
)

plt.savefig(
    output / "standing_roa.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

np.savez(
    output / "standing_roa_data.npz",
    theta_values=theta_values,
    omega_values=omega_values,
    roa_map=roa_map
)


data = np.load(
    output / "standing_roa_data.npz"
)

roa_data = {
    "theta_values": data["theta_values"],
    "omega_values": data["omega_values"],
    "roa_map": data["roa_map"],
}




# fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


# def draw_frame(index):
#     # The massless swing leg is repositioned instantaneously at each impact.
#     model.visualize(state_traj[:, index], params, ax=ax)
#     ax.set_title(f"t = {time_traj[index]:.2f} s")


# # Simulate at a small timestep, but render only 25 frames per second.
# fps = 25
# frame_stride = round(1 / (fps * timestep))
# frame_indices = list(range(0, time_traj.size, frame_stride))
# if frame_indices[-1] != time_traj.size - 1:
#     frame_indices.append(time_traj.size - 1)

# animation = FuncAnimation(
#     fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
# )
# output = Path("output/assignment_2")
# output.mkdir(parents=True, exist_ok=True)
# animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# # To save an MP4 instead, install FFmpeg and use:
# # animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
# print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
# plt.show()
