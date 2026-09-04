import numpy as np
import matplotlib.pyplot as plt
import timeit

# from models import pendulum as model
from models import bouncing_ball as model
from integrators import rk4 as integrator

# Basic simulation of the bouncing ball

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "mass": 0.2,  # point mass (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
    "restitution_coeff": 1,  # coefficient of restitution (dimensionless)
}


# some set-up
initial_state = np.array([1.0, 0.0])  # initial height and speed
ENERGY_CRITERIA = 1e-3

timestep = 1e-5
sim_time = 5.0

# simulation loop
# while True:
n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# t1 = timeit.default_timer()
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator(t, state_traj[:, step], timestep, params)
    
# t2 = timeit.default_timer()
# print(f"Timestep {timestep} took {t2 - t1:.6f} seconds")

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

# # Check if the total energy is conserved
total_energy = potential_energy + kinetic_energy
# if np.any(np.abs(total_energy - total_energy[0]) > ENERGY_CRITERIA):
#     print(f"Energy is not conserved for timestep {timestep}")
#     break
# else:
#     print(f"Energy is conserved for timestep {timestep}")
#     timestep *= 1.1

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing ball energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot
