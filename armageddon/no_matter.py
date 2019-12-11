from solver import Planet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analytical import anal_sol

earth = Planet(atmos_func='tabular', atmos_filename='../data/AltitudeDensityTable.csv')
df, out = earth.impact(10, 20e3, 3000, 3000, 45, num_scheme='RK', init_altitude=80000)
print(df)
print(out)
print(df.altitude)
schemes = [
            'EE',
            'IE',
            'MIE',
            'RK'
            ]

'''dts = [0.1, 0.05, 0.01, 0.005, 0.001]
dt_rms = []
for dt in dts:
    RMS_velocity = []
    RMS_dedz = []
    for scheme in schemes:
        df, dic = earth.impact(5, 10e3, 1200, 1e5, 30, 100e3, dt, False, False, scheme)
        H_plot = np.array(df.altitude)
        anal_df = anal_sol(H_plot, 5, 10e3, 1200, 1e5, 30, 100e3, False)
        RMS_velocity.append(np.sqrt(1/len(df) * sum((anal_df.velocity - df.velocity)**2)))
    dt_rms.append(RMS_velocity)

dt_rms = np.array(dt_rms).T
'''
fig = plt.figure(figsize=(7, 7))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(df.mass, df.altitude)
'''ax1.plot(dts, dt_rms[0], label='EE')
ax1.plot(dts, dt_rms[1], label='IE')
ax1.plot(dts, dt_rms[2], label='MIE')
ax1.plot(dts, dt_rms[3], label='RK')'''
ax1.set_xlabel('mass', fontsize=14)
ax1.set_ylabel('altitude', fontsize=14)
#ax1.set_title("RMSs")
#ax1.set_ylim(68.44, 68.46)
ax1.grid()
ax1.legend()

ax2.plot(df.dedz, df.altitude, linestyle=':', label='numerical')
ax2.set_xlabel('dedz', fontsize=14)
ax2.set_ylabel('altitude [m]', fontsize=14)
#ax2.set_title("numerical")
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
