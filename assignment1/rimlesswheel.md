# Rimless Wheel Assignment 1

## Explanation of sanity check
In rimlesswheeltest.py, I used the state space 0.1, 0.0 for theta and theta dot to represent the initial state of the wheel. When running the code, this produces an array of theta dot = 0, and theta double dot = 0.979. Manually plugging 0.1 into the equation theta double dot = 9.81*sin(0.1) also gives me 0.979.

Testing the impact reset, I initally plugged in pi/6 for alpha and -2.0 rad/s. The impact returned -0.525 and -1, which means the magnitude of the angle stayed the same while the signs switched as a new angle was define, while the angular velocity halved. This follows the given equation of θ˙+=θ˙−*cos(2α). (2α in this case is pi/3, cos(pi/3) = 0.5).

## A State-space plot showing RoA
Attached in 1roa.png under figures

## One-Dimensional return-map plot
Attached in 2returnmap.png under figures

## Visualization and discussion of how slope and number of spokes affect RoA, local convergence
Slope figures attached in 3gammavsfloquet.png and 3gammavsroa.png
Spokes figures attached in 4spokesvsfloquet.png and 4spokesvsroa.png

Explanation:
Taking a look at gamma vs floquet graph, the graphed result stays constant regardless of the gamma value, which concludes that the angle does not affect the floquest multiplier. However, the angle does affect the RoA, as increasing gamma leads to a larger number of initial condtions that the wheel will converge to stable walking. *Note, in my model my angle is negated due to how I set up the governing equations.*

Both the floquet multiplier and the chance of local convergence increases when the number of wheel spokes increases. However, in my # of spokes vs RoA, spokes # from 6-8 produce 0, which I am unsure about. I could see how having fewer spokes may make it more difficult to achieve stable walking, since in my mind less spokes equates to more "bumpy" and more impact losses.
