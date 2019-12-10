import numpy as np
from scipy.integrate import solve_ivp


def dmove(t, Point):
    C_D, g, C_H, Q, C_L, R_p, alpha, rho_m, rho_0, H, Y = (1.,9.81,0.1,1e7,1e-3,6371000.,0.3,3000,1.2,8000.,1e5)
    v, m, theta, z, x, r = Point
    A = np.pi*(r**2)
    rho_a = rho_0*np.exp(-z/H)
    new_r = 0
    if rho_a * (v**2) >= Y:
        new_r = v*((7/2*alpha*rho_a/rho_m)**(1/2))
    return np.array([(-C_D*rho_a*A*(v**2))/(2*m) + g*np.sin(theta),
                     (-C_H*rho_a*A*(v**3))/(2*Q),
                     (g*np.cos(theta))/(v) - (C_L*rho_a*A*v)/(2*m) - (v*np.cos(theta))/(R_p + z),
                     -v*np.sin(theta),
                     (v*np.cos(theta))/(1 + z/R_p),
                     new_r
                    ])

tmax = 60
t = np.arange(0,tmax,0.05)
P1 = solve_ivp(dmove,(0,tmax),(21e3,4000*np.pi*(10**3),np.pi/4,100e3,0.,10.),method='RK45', t_eval=t)
energy_t = -(0.5*P1['y'][1, 1:]*(P1['y'][0, 1:]**2) - 0.5*P1['y'][1, :-1]*(P1['y'][0, :-1]**2))/(t[1:]-t[:-1])
energy_z = (0.5*P1['y'][1, 1:]*(P1['y'][0, 1:]**2) - 0.5*P1['y'][1, :-1]*(P1['y'][0, :-1]**2))/(P1['y'][3, 1:]-P1['y'][3, :-1])
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,12))
# ax = Axes3D(fig)
plt.subplot(331)
plt.plot(t,P1['y'][0]) # v-t
plt.title("v-t")
plt.subplot(332)
plt.plot(t,P1['y'][1]) # m-t
plt.title("m-t")
plt.subplot(333)
plt.plot(t,P1['y'][2]*180/np.pi) # theta-t
plt.title("$theta-t$")
plt.subplot(334)
plt.plot(t,P1['y'][3]) # z-t
plt.title("z-t")
plt.subplot(335)
plt.plot(t,P1['y'][4]) # x-t
plt.title("x-t")
plt.subplot(336)
plt.plot(t,P1['y'][5]) # r-t
plt.title("r-t")
plt.subplot(337)
plt.plot(t[1:],energy_t) # Energy-t
plt.title("Energy-t")
plt.subplot(338)
plt.plot(energy_z, P1['y'][3, 1:]) # Energy-z
plt.title("Energy-z")
plt.subplot(339)
plt.plot(P1['y'][4],P1['y'][3]) # z-x
plt.title("Trace")
plt.show()