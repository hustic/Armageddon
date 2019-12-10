from solver import Planet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analytical import anal_sol

earth = Planet(atmos_func='exponential', g=0, Cl=0, Ch=0)
df, dic = earth.impact(5, 10e3, 1200, 1e5, 30, 100e3, 0.001, False, False, 'runge kutta')
anal_df = anal_sol(5, 10e3, 1200, 1e5, 30, 100e3, False)
print(df)
fig = plt.figure(figsize=(7, 7))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(anal_df.velocity, anal_df.altitude, linestyle='-', label='analytical')
#ax1.set_xlabel('dedz', fontsize=14)
ax1.set_ylabel('altitude [m]', fontsize=14)
ax1.set_title("analytical")
ax1.grid()
ax1.legend()
#ax1.set_xlim((0, 35))

ax2.plot(df.velocity, df.altitude, linestyle=':', label='numerical')
ax2.set_xlabel('velocity', fontsize=14)
ax2.set_ylabel('altitude [m]', fontsize=14)
ax2.set_title("numerical")
ax2.grid()
ax2.legend()
#ax2.set_xlim((0, 35))

plt.show()

'''
0: velocity
1: mass
2: angle
3: altitude
4: distance
5: radius
'''
