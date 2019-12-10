import numpy as np

v = 20e3
r = 10
A = np.pi * r**2
Cd = 1
angle = 45
H = 8000
Z = 100000
rho = 3000
m = 4/3 * np.pi * r**3 * rho
vv = np.exp((-Cd * A * 1.2 *H / (2 * m * (np.sqrt(2) / 2))) * np.exp(-Z/H))
c = v / vv
def v_h(h):
    return c * np.exp((-Cd * A * 1.2 * H / (2 * m * (np.sqrt(2) / 2))) * np.exp(-h/H))
C2 = -Cd * A * 1.2 * H / (2 * m * np.sqrt(2) / 2)
def dEdz(z):
    return c * np.exp(C2 * np.exp(-z/H)) * C2 * np.exp(-z/H) * (-1/H) * m * v_h(z)
fig = plt.figure(figsize=(6, 6))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
H_plot = np.linspace(0, 100000, 200)
dh = 500
E = 0.5 * m * v_h(H_plot)**2
de = (E[1:] - E[0:-1]) / dh
ax1.plot(dEdz(H_plot), H_plot, 'r-')
ax1.set_xlabel('dedz', fontsize=14)
ax1.set_ylabel('Height', fontsize=14)
ax2.plot(de,H_plot[1:], 'bo', label='differentiation')
ax2.plot(dEdz(H_plot), H_plot,'r_', label='exact')
ax2.set_xlabel('Energy Change per Unit Height', fontsize=14)
ax2.set_ylabel('Height', fontsize=14)
ax2.legend(loc='best', fontsize=14)