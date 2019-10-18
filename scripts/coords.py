import numpy as np
from math import atan2, asin, cos, sin, pi

from Config import Config
r = Config.CAP_RANGE                                        # capture range

def get_vecs(x):
    D1 = np.concatenate((x['D1'], [0]))
    D2 = np.concatenate((x['D2'], [0]))
    I = np.concatenate((x['I'], [0]))
    
    D1_I = I - D1
    D2_I = I - D2
    D1_D2 = D2 - D1

    return D1_I, D2_I, D1_D2

def phy_to_thtalpha(x):

    D1_I, D2_I, D1_D2 = get_vecs(x)

    alpha_1 = asin(np.clip(r/np.linalg.norm(D1_I), -1, 1))      # state: alpha_1
    alpha_2 = asin(np.clip(r/np.linalg.norm(D2_I), -1, 1))      # state: alpha_2

    k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I)) # angle between D1_D2 to D1_I
    k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I)) # angle between D1_D2 to D2_I

    tht = k2 - k1
    
    if k1 < 0:
        tht += 2*pi

    return np.array([alpha_1, alpha_2, tht])

def phy_to_thtd(x):

    D1_I, D2_I, D1_D2 = get_vecs(x)

    d1 = max(np.linalg.norm(D1_I), r)
    d2 = max(np.linalg.norm(D2_I), r)

    k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I)) # angle between D1_D2 to D1_I
    k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I)) # angle between D1_D2 to D2_I

    tht = k2 - k1
    
    if k1 < 0:
        tht += 2*pi

    return np.array([d1, d2, tht]), [D1_I, D2_I]

def phy_to_xyz(x):

    D1_I, D2_I, D1_D2 = get_vecs(x)
    
    z = np.linalg.norm(D1_D2)/2
    
    x = -np.cross(D1_D2, D1_I)[-1]/(2*z)
    y =  np.dot(D1_D2, D1_I)/(2*z) - z
    
    return np.array([x, y, z])

def phy_to_hds(xs):
    D1 = np.array([xs[0, 0], xs[0, 1], 0])
    D2 = np.array([xs[0, 2], xs[0, 3], 0])
    I = np.array([xs[0, 4], xs[0, 5], 0])

    xaxis = np.array([1, 0, 0])
    
    D1_D2 = D2 - D1

    O = 0.5*(D1 + D2)[:2]

    c = np.dot(D1_D2, xaxis)
    s = np.cross(xaxis, D1_D2)[-1]
    tht = atan2(s, c)

    C = np.array([[cos(tht), sin(tht)], [-sin(tht), cos(tht)]])

    return O, C


def traj_to_hds(traj, O, C):

    traj_ = np.zeros(np.shape(traj))

    for i in range(len(traj)):
        traj_[i,0:2] = C.dot(traj[i,0:2]-O)
        traj_[i,2:4] = C.dot(traj[i,2:4]-O)
        traj_[i,4:6] = C.dot(traj[i,4:6]-O)

    return traj_

def circ_to_hds(cs, O, C):
    cs_ = np.zeros(np.shape(cs))

    for i in range(len(cs)):
        cs_[i] = C.dot(cs[i]-O)

    return cs_
