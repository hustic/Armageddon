import numpy as np
from solver import Planet
import matplotlib.pyplot as plt
from analytical import anal_sol
import time
import scipy_test
import asteroid_par

earth = Planet(g=0, Cl=0, Ch=0)





'''sci_res = scipy_test.sci_sol(5, 10e3, 1200, 10e5, 45, num_scheme='RK45', fragmentation=False, g=0, C_L=0, C_H=0, dt=0.01)

df, out = earth.impact(5, 10e3, 1200, 10e5, 45, num_scheme='RK', fragmentation=False, dt=0.01, ensemble=False)
anal_res = anal_sol(df.altitude, 5, 10e3, 1200, 10e5, 45)

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.scatter(df.velocity, df.altitude, label='numeric', marker='.', color='r')
ax1.plot(sci_res.velocity, sci_res.altitude, label='scipy', color='b')
ax1.plot(anal_res.velocity, anal_res.altitude, label='anal', color='g')
ax1.set_ylabel('velocity')
ax1.set_xlabel('altitude')
ax1.set_ylim(0,1e5)
ax1.grid()
ax1.legend()

ax2.scatter(df.dedz, df.altitude, label='numeric', marker='.', color='r')
ax2.plot(sci_res.dedz, sci_res.altitude, label='scipy', color='b')
ax2.plot(anal_res.dedz, anal_res.altitude, label='anal', color='g')
ax2.set_ylabel('dedz')
ax2.set_xlabel('altitude')
ax2.set_ylim(0,1e5)
ax2.grid()
ax2.legend()

plt.show()'''