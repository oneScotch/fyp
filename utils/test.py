import random
from utils.rotation_conversions import axis_angle_to_quaternion, quaternion_to_axis_angle
from utils.quaternions import *
import torch 

import smplx
def bone_rotations_weighted_average_log_ref(reference_rotations, sample_rotations, sample_weights):

    # Number of quaternions to average
    M = sample_rotations.shape[0]
    N = sample_rotations.shape[1]
    accum_rotations = torch.zeros((54, 4))
    blended_rotations= torch.zeros((54, 3))
    assert(sample_rotations.shape[0] == len(sample_weights))
    assert(sample_rotations.shape[1] == len(reference_rotations))
    for i in range(0,M):
        for j in range(0,N):
            sample_rotation = axis_angle_to_quaternion(torch.from_numpy(sample_rotations[i, j]))
            # print(ample_rotatio)
            accum_rotations[j] += sample_weights[i] * quat_log(quat_inv_mul(reference_rotations[j], sample_rotation))
    for j in range(0,N):
        accum_rotations[j] = quaternion_multiply(reference_rotations[j], quat_exp(accum_rotations[j]))
        blended_rotations[j] = quaternion_to_axis_angle(accum_rotations[j])
        # print(blended_rotations[j])
    return blended_rotations

def calculate_sample_weights():

    kwargs = dict(gender='neutral',
        num_betas=10,
        use_face_contour=True,
        flat_hand_mean=args.flat_hand_mean,
        use_pca=False,
        batch_size=1)

    smplx_model = smplx.create(
        '../common/utils/human_model_files', 'smplx', 
        **kwargs).to(args.device)