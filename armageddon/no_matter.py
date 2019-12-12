import numpy as np
from solver import Planet
import matplotlib.pyplot as plt
from analytical import anal_sol
import time
import scipy_test

radius = [5, 10]
velocity = [10e3, 20e3]
strength = [3000, 10e5]
angle = [30, 45]
density = [1200, 3000]

earth = Planet()
#df_vec, out_vec = earth.impact(radius, velocity, density, strength, angle, num_scheme='RK', fragmentation=True)

'''sci_res = scipy_test.sci_sol()
print(sci_res)'''

start_time = time.time()
df, out = earth.impact(10, 20e3, 3000, 10e5, 81, num_scheme='RK', fragmentation=True, dt=0.05, ensemble=False)
print(df, out)
print(time.time() - start_time)

#print(df.altitude)
#anal_df = anal_sol(np.array(df.altitude), 1, 10e3, 3000, 10e5, 45)

earth.plot_results(df)

'''fig = plt.figure(figsize=(8, 8))
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

plt.show()'''