import numpy as np
import random
from utils.rotation_conversions import axis_angle_to_quaternion, quaternion_to_axis_angle
from utils.quaternions import *
import torch 

def bone_rotations_weighted_average_log_ref(reference_rotations, sample_rotations, sample_weights):

    # Number of quaternions to average
    M = sample_rotations.shape[0]
    N = sample_rotations.shape[1]
    accum_rotations = np.zeros((55, 4))
    blended_rotations= np.zeros((55, 3))
    assert(sample_rotations.shape[0] == len(sample_weights))
    assert(sample_rotations.shape[1] == len(reference_rotations))
    for i in range(0,M):
        for j in range(0,N):
            sample_rotation = axis_angle_to_quaternion(torch.FloatTensor(sample_rotations[i, j]))
            accum_rotations[j] += sample_weights[i] * quat_log(quat_abs(quat_inv_mul(reference_rotations[j], sample_rotation)))
    for j in range(0,N):
        accum_rotations[j] = quat_abs(quat_mul(reference_rotations[j], quat_exp(accum_rotations[j])))
        blended_rotations[j] = quaternion_to_axis_angle(torch.from_numpy(accum_rotations[j])).numpy()
        print(blended_rotations[j])
    return blended_rotations


