import torch
def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a, b):

    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion):

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quat_inv_mul(quaternion1, quaternion0):
    return quaternion_multiply(quaternion_invert(quaternion1), quaternion0)

def quat_log(quaternion0, eps=1e-8):
    v = quaternion0[1:]
    w0 = quaternion0[0]
    magnitude = torch.sqrt(torch.dot(quaternion0, quaternion0))
    
    if torch.sqrt(torch.dot(v,v)) < eps :
        return torch.cat((torch.tensor([0]),v),axis=0)
    z = (v / torch.sqrt(torch.dot(v,v))) * torch.arccos(w0 / magnitude)
    r = torch.cat((torch.log(magnitude).unsqueeze(0),z),axis=0)
    return r
    
def quat_exp(quaternion0, eps=1e-8):
    v = quaternion0[1:]
    vn = torch.sqrt(torch.dot(v,v))
    w0 = quaternion0[0]
    if (vn < eps):
        return torch.cat((torch.tensor([1.0]),v),axis=0)
    r = torch.exp(w0)* torch.cat((torch.cos(vn).unsqueeze(0),(torch.sin(vn)/vn)*v),axis=0)
    return r