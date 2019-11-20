import numpy as np
from math import atan2
from scipy.optimize import NonlinearConstraint, minimize

def dominant_region(x, xi, xds, a):
    for i, xd in enumerate(xds):
        if i == 0:
            inDR = a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - r)
        else:
            inDR = max(inDR, a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - r))
    return inDR 

def depth_in_target(xi, xds, target, a):
    def dr(x, xi=xi, xds=xds, a=a):
        return dominant_region(x, xi, xds, a)
    on_dr = NonlinearConstraint(dr, -np.inf, 0)
    sol = minimize(target, xi, constraints=(on_dr,))
    return sol.x

def deep_target_strategy(xi, xds, target, a):
    # print(xi, xds)
    
    xt = depth_in_target(xi, xds, target, a)
    # print(xt)
    IT = np.concatenate((xt - xi, np.zeros((1,))))
    DTs = []
    for xd in xds:
        DT = np.concatenate((xt - xd, np.zeros(1,)))
        DTs.append(DT)
    xaxis = np.array([1, 0, 0])

    psi = atan2(np.cross(xaxis, IT)[-1], np.dot(xaxis, IT))
    phis = []
    for DT in DTs:
        phi = atan2(np.cross(xaxis, DT)[-1], np.dot(xaxis, DT))
        phis.append(phi)

    return [psi], phis