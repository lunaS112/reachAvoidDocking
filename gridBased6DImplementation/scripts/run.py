import numpy as np
np.random.seed(0)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dynamics import f
from utils.simulation import dock

from ComboControl import ComboController as Controller

s0 = [-6.0, 2.0, 0.0, 0.0, -np.pi, 0.0] # Initial state: [x, y, vx, vy, theta, omega]
controller = Controller()
final_time = (controller.final_time) * -1
dt = 0.1
nt = int(final_time/dt + 1)
f = f
#print('Save animation? (y/n): ', end='')
#save_animation = input() == 'y'
save_animation = True
animation_save_path = 'outputs/RA_animation.mp4'
controller.reset()

qualified, entered_failure, initial_in_brt = dock(s0, controller, dt, nt, f,
                                  save_animation, animation_save_path)

print()
print('###############')
print('### SUMMARY ###')
print('###############')
print()
print(f'Initial state within BRT:    {initial_in_brt}')
print(f'Entered failure:             {entered_failure}')
print(f'Finished in target:          {qualified}')



