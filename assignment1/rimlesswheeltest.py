import numpy as np

from models import rimlesswheel as model

state = np.array([0.1, 0.0])

params = {
    "gravity": 9.81,
    "length": 1.0,
    "gamma": 0.0,
    "alpha": np.pi / 6
}

result = model.dynamics(0, state, params)

print(result)

state = np.array([params["alpha"], -2.0])

new_state = model.calculate_impact(0, state, params)

print(new_state)