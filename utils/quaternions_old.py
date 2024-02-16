import numpy as np
def quat_mul(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
def quaternion_inverse(quaternion0):
    w0, x0, y0, z0 = quaternion0
    return np.array([-w0, x0, y0, z0], dtype=np.float64)
def quat_inv_mul(quaternion1, quaternion0):
    return quat_mul(quaternion_inverse(quaternion1), quaternion0)
def quat_abs(quaternion0):
    if quaternion0[0] < 0:
        return -quaternion0
    else:
        return quaternion0
def quat_log(quaternion0, eps=1e-8):
    v = quaternion0[1:]
    w0 = quaternion0[0]
    magnitude = np.sqrt(np.dot(quaternion0, quaternion0))
    if np.sqrt(np.dot(v,v)) < eps :
        return np.concatenate(([0],v),axis=0)
    z = (v / np.sqrt(np.dot(v,v))) * np.arccos(w0 / magnitude)
    r = np.concatenate(([np.log(magnitude)],z),axis=0)
    return r
    
def quat_exp(quaternion0, eps=1e-8):
    v = quaternion0[1:]
    vn = np.sqrt(np.dot(v,v))
    w0 = quaternion0[0]
    if (vn < eps):
        return np.concatenate(([1.0],v),axis=0)
    r = np.exp(w0)* np.concatenate(([np.cos(vn)],(np.sin(vn)/vn)*v),axis=0)
    return r