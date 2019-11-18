import numpy as np
from math import atan2
from scipy.optimize import NonlinearConstraint, minimize
from geometries import dominant_region

def depth_in_target(xi, xds, target, a):
    def dr(x, xi=xi, xds=xds, a=a):
        return dominant_region(x, xi, xds, a)
    on_dr = NonlinearConstraint(dr, -np.inf, 0)
    sol = minimize(target, xi, constraints=(on_dr,))
    return sol.x

def deep_target_strategy(xi, xds, target, a):
    
    xt = depth_in_target(xi, xds, target, a)
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