from solver import Planet
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analytical import anal_sol

earth = Planet(atmos_func='exponential')
df, dic = earth.impact(10, 20e3, 3000, 1e5, 45)
#print(df)
#print(dic)
anal_df = anal_sol()
print(anal_df)
fig = plt.figure(figsize=(7,7))
plt.plot(df['dedz'], df['altitude'])
plt.xlabel('altitude')
plt.ylabel('time')
#plt.ylim(0,20000)
plt.grid()
plt.show()

'''
fig = plt.figure(figsize=(6, 6))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(v_h(H_plot), H_plot, 'r-')
ax1.set_xlabel('velocity', fontsize=14)
ax1.set_ylabel('Height', fontsize=14)
ax2.plot(de,H_plot[1:], 'bo', label='differentiation')
ax2.plot(dEdz(H_plot), H_plot,'r_', label='exact')
ax2.set_xlabel('Energy Change per Unit Height', fontsize=14)
ax2.set_ylabel('Height', fontsize=14)
ax2.legend(loc='best', fontsize=14)
plt.show()
'''
'''
0: velocity
1: mass
2: angle
3: altitude
4: distance
5: radius
'''
