import numpy as np
from solver import Planet
import matplotlib.pyplot as plt
from analytical import anal_sol

earth = Planet(g=0, Cl=0, Ch=0)
df, out = earth.impact(1, 10e3, 3000, 10e5, 45, num_scheme='RK', fragmentation=False)

anal_df = anal_sol(np.array(df.altitude), 1, 10e3, 3000, 10e5, 45)

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.scatter(df.altitude, df.angle, label='numeric', marker='.', color='r')
ax1.plot(anal_df.altitude, anal_df.velocity, label='analytic', color='b')
ax1.set_ylabel('altitude')
ax1.set_xlabel('velocity')
ax1.grid()
ax1.legend()

ax2.scatter(df.altitude, df.dedz, label='numeric', marker='.', color='r')
ax2.plot(df.altitude, anal_df.dedz, label='analytic', color='b')
ax2.set_ylabel('altitude')
ax2.set_xlabel('dedz')
ax2.grid()
ax2.legend()

plt.show()