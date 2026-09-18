"""Lookup-table utilities for the walker Poincare return map."""

import numpy as np


def build_lookup_table(n_omega, n_alpha, params, roa_data, simulate_return):
    """Sample the return map over state-action space.

    ``simulate_return(omega, alpha, params, roa_data)`` returns
    ``(next_omega, reached_roa)`` for one state-action pair.
    """
    omega_values = np.linspace(0.0, np.sqrt(2 * params["gravity"] / params["length"]), n_omega)
    alpha_values = np.linspace(np.pi / 8, np.pi / 7, n_alpha)
    next_omega_table = np.full((n_omega, n_alpha), np.nan)
    reach_roa_table = np.zeros((n_omega, n_alpha), dtype=bool)

    for i, omega0 in enumerate(omega_values):
        for j, alpha0 in enumerate(alpha_values):
            omega_next, reached_roa = simulate_return(
                omega0, alpha0, params.copy(), roa_data=roa_data
            )
            if reached_roa:
                reach_roa_table[i, j] = True
            elif omega_next is not None:
                next_omega_table[i, j] = omega_next

    return omega_values, alpha_values, next_omega_table, reach_roa_table


def compute_lookup_errors(omega_values, next_omega_table):
    """Return nearest-grid-point errors for all valid return-map samples."""
    valid_next_omega = next_omega_table[np.isfinite(next_omega_table)]
    nearest_indices = np.abs(omega_values[:, None] - valid_next_omega).argmin(axis=0)
    return np.abs(valid_next_omega - omega_values[nearest_indices])


def compute_steps_to_roa(omega_values, next_omega_table, reach_roa_table):
    """Find the minimum number of lookup-table steps needed to reach the RoA."""
    next_indices = np.abs(omega_values[:, None, None] - next_omega_table[None]).argmin(axis=0)
    valid = np.isfinite(next_omega_table)
    steps = np.full(omega_values.size, np.inf)
    actions = np.full(omega_values.size, -1, dtype=int)
    direct = reach_roa_table.any(axis=1)
    steps[direct] = 1
    actions[direct] = reach_roa_table[direct].argmax(axis=1)

    while True:
        candidate_steps = np.where(valid, 1 + steps[next_indices], np.inf)
        candidate_actions = candidate_steps.argmin(axis=1)
        updated_steps = np.minimum(steps, candidate_steps[np.arange(steps.size), candidate_actions])
        changed = updated_steps < steps
        if not changed.any():
            return steps, actions, next_indices
        steps[changed] = updated_steps[changed]
        actions[changed] = candidate_actions[changed]


def action_plan(initial_index, actions, next_indices, reach_roa_table):
    """Return the fastest landing-angle indices from one grid state to the RoA."""
    direct = reach_roa_table.any(axis=1)
    plan = []
    index = initial_index
    while not direct[index]:
        action = actions[index]
        plan.append(action)
        index = next_indices[index, action]
    return plan


def longest_action_plan(initial_index, next_indices, next_omega_table, reach_roa_table):
    """Return a longest finite landing-angle plan to the RoA, if one exists."""
    valid = np.isfinite(next_omega_table)
    direct = reach_roa_table.any(axis=1)
    visiting, cache = set(), {}

    def visit(index):
        if direct[index]:
            return 0, []
        if index in cache:
            return cache[index]
        if index in visiting:
            return np.inf, []
        visiting.add(index)
        candidates = [
            (1 + length, [action] + plan)
            for action in np.flatnonzero(valid[index])
            for length, plan in [visit(next_indices[index, action])]
        ]
        visiting.remove(index)
        cache[index] = max(candidates, key=lambda candidate: candidate[0])
        return cache[index]

    return visit(initial_index)


def find_comparison_initial_condition(steps, next_indices, next_omega_table, reach_roa_table):
    """Find a >=3-step state whose shortest and longest policies differ."""
    for index in np.flatnonzero(steps >= 3):
        longest_steps, _ = longest_action_plan(
            index, next_indices, next_omega_table, reach_roa_table
        )
        # Add the final standing-controller step to match the ``steps`` convention.
        if np.isfinite(longest_steps) and longest_steps + 1 > steps[index]:
            return index
    raise ValueError("No finite comparison trajectory was found.")
