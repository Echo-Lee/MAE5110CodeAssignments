# Assignment 2 Report

To run the codes for assignment 2, please use:
```
uv run python assignment_2.py
```
## Sketch
![Sketch](../output/assignment_2/sketch.jpg)


## Region of attraction

The saturated feedback-linearizing ankle controller stabilizes the upright state
inside the sampled region below.

![Standing-controller RoA](../output/assignment_2/standing_roa.png)

## Poincare section and grid

I use the upward crossing of $\theta=0$ with $\dot\theta>0$. It is transverse
to the walking orbit and leaves $\dot\theta$ as the scalar return-map state.
The final lookup table uses 100 velocity points and 20 landing-angle points.

I choose 100 for $\theta_0$, because it gives an relative error rate of less than 0.5\%, while other smaller desolutions don't. And for the choice of $\alpha$, I just simply set it to a constant, for it does not influence the dynamics except for the time when touching the slope.

## Three-step trajectory and longest walk

The selected initial condition is $\theta_0=0$ and
$\dot\theta_0=3.893\ \mathrm{rad/s}$. The lookup policy reaches the standing
RoA in three steps. Enumerating every valid action from this initial grid cell
gives a maximum finite duration of four steps before entering the RoA; the
figure displays the minimum-step (blue) and maximum-duration (red) policies.

![Three-step trajectory](../output/assignment_2/three_step_trajectory.png)

## Steps to standstill

The plot below gives the minimum number of footstrikes to enter the standing RoA
for each initial velocity on the Poincare section. The red star marks the
three-step initial condition above.

![Steps to standing](../output/assignment_2/steps_to_standstill.png)
