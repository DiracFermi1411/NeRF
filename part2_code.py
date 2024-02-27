import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import math

def get_rays(height, width, intrinsics, Rcw, Tcw): # Rwc, Twc
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from camera to world coordinates.
    Tcw: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    
    #############################  TODO 2.1 BEGIN  ########################## 
    
    # Compute the ray directions
    v, u = torch.meshgrid(torch.arange(width, dtype=torch.float32, device = device),
                          torch.arange(height, dtype=torch.float32, device = device))
    uv1 = torch.stack((u.flatten(), v.flatten(), torch.ones_like(u.flatten())))
    Kinv = torch.inverse(intrinsics)
    dir_c = Kinv @ uv1
    dir_w = (Rcw @ dir_c).T
    ray_directions = dir_w.reshape(height, width, 3)
    

    # Compute the ray origins
    # Twc = - Rcw.T @ Tcw
    ray_origins = torch.tile(Tcw.reshape(1, 3), (height, height, 1))
    
    return ray_origins, ray_directions


def stratified_sampling(ray_origins, ray_directions, near, far, samples):
    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
    height, width = ray_origins.shape[:2]

    # Generate stratified samples
    t_samples = (torch.rand(height, width, samples, device=ray_origins.device) + torch.arange(samples, device=ray_origins.device)) / (samples + 1e-6)
    t_samples = t_samples.expand(height, width, samples)
    t_values = near * (1.0 - t_samples) + far * t_samples

    # Compute 3D coordinates
    ray_points = ray_origins[..., None, :] + t_values[..., None] * ray_directions[..., None, :]

    # Compute depth values
    depth_points = t_values

    #############################  TODO 2.2 END  ############################

    return ray_points, depth_points


    
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper. 
    """
    def positional_encoding(x, num_frequencies=6, incl_input=True):
        
        """
        Apply positional encoding to the input.
        
        Args:
        x (torch.Tensor): Input tensor to be positionally encoded. 
        The dimension of x is [N, D], where N is the number of input coordinates,
        and D is the dimension of the input coordinate.
        num_frequencies (optional, int): The number of frequencies used in
        the positional encoding (default: 6).
        incl_input (optional, bool): If True, concatenate the input with the 
            computed positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor. 
        """
        
        results = []
        if incl_input:
            results.append(x)
        #############################  TODO 1(a) BEGIN  ############################
        # encode input tensor and append the encoded tensor to the list of results.
        for i in range(num_frequencies):
            results.append(torch.sin(x * (2**(i)) * math.pi))
            results.append(torch.cos(x * (2**(i)) * math.pi)) 
        

        #############################  TODO 1(a) END  ##############################
        return torch.cat(results, dim=-1)
    
    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
    # Calculate the sizes of gamma_x and gamma_d
        gamma_x_size = 3 + 2 * 3 * num_x_frequencies  # (2DL: D = dimension of the input, L = number of frequencies)
        gamma_d_size = 3 + 2 * 3 * num_d_frequencies

        # Regular layers:
        self.layer1 = nn.Linear(gamma_x_size, filter_size)
        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, filter_size)
        self.layer4 = nn.Linear(filter_size, filter_size)
        self.layer5 = nn.Linear(filter_size, filter_size)
        self.layer6 = nn.Linear(filter_size + gamma_x_size, filter_size)
        self.layer7 = nn.Linear(filter_size, filter_size)
        self.layer8 = nn.Linear(filter_size, filter_size)
        self.layer9 = nn.Linear(filter_size, 1)
        self.layer10 = nn.Linear(filter_size, filter_size)
        self.layer11 = nn.Linear(filter_size + gamma_d_size, 128)
        self.layer12 = nn.Linear(128, 3)
        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################

        layer1 = F.relu(self.layer1(x))
        layer2 = F.relu(self.layer2(layer1))
        layer3 = F.relu(self.layer3(layer2))
        layer4 = F.relu(self.layer4(layer3))
        layer5 = F.relu(self.layer5(layer4))
        layer5_cat = torch.cat([layer5, x], dim=-1)
        layer6 = F.relu(self.layer6(layer5_cat))
        layer7 = F.relu(self.layer7(layer6))
        layer8 = F.relu(self.layer8(layer7))
        sigma = self.layer9(layer8)
        layer10 = self.layer10(layer8)
        layer10_cat = torch.cat([layer10, d], dim=-1)
        layer11 = F.relu(self.layer11(layer10_cat))
        rgb = torch.sigmoid(self.layer12(layer11))

        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
  
    device = ray_points.device

    # Normalize ray directions
    ray_directions_norm = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # Repeat directions for every point
    ray_directions_pop = ray_directions_norm.unsqueeze(-2).repeat(1, 1, ray_points.shape[2], 1)

    # Flatten vectors
    ray_points_flat = ray_points.view(-1, 3)  # (H * W * N_samples, 3)
    ray_directions_flat = ray_directions_pop.view(-1, 3)  # (H * W * N_samples, 3)

    # Apply positional encoding
    en_ray_directions_flat = nerf_model.positional_encoding(ray_directions_flat, num_frequencies=num_d_frequencies)
    en_ray_points_flat = nerf_model.positional_encoding(ray_points_flat, num_frequencies=num_x_frequencies)

    # Split ray batches into chunks to avoid memory errors
    ray_points_batches = [en_ray_points_flat[i:i+ 2**15] for i in range(0, en_ray_points_flat.shape[0], 2**15)]
    ray_directions_batches = [en_ray_directions_flat[i:i+ 2**15] for i in range(0, en_ray_directions_flat.shape[0], 2**15)]

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches


def volumetric_rendering(rgb, sigma, depth_points):
    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    device = rgb.device

    # Calculate the transmittance for each ray
    delta_depth_finalmat = (10**9) * torch.ones_like(depth_points).to(device)
    # print(delta_depth_finalmat.shape)
    delta_depth_finalmat[..., :-1] = torch.diff(depth_points, dim=-1)
    sigma_deltas = - F.relu(sigma) * delta_depth_finalmat.reshape_as(sigma)
    # print(sigma_deltas.shape)
    # print(sigma_deltas)
    T = torch.cumprod(torch.exp(sigma_deltas), dim = -1)
    T = torch.roll(T, 1, dims=-1)
    C = ((T * (1 - torch.exp(sigma_deltas)))[..., None]) * rgb
    rec_image = torch.sum(C, dim=-2)
    # sigma_delta_N = (10**9) * torch.ones(depth_points.shape[0], depth_points.shape[1], 1)


    # print(sigma_delta_N)


    # sigma_deltas = torch.cat([sigma_deltas, sigma_delta_N], dim = -1)
    # print(torch.exp(sigma_deltas))
    # print(torch.exp(-sigma_deltas))

    # T_0 = torch.ones(T.shape[0], T.shape[1], 1, device = device)
    # # print(T_0.shape)
    # T = torch.cat([T_0, T], dim = -1)


    # print(T.shape)
    # print(sigma_deltas)
    # print(delta_depth.shape)
    # print((T * delta_depth)[..., None].shape)    
    # print((T * delta_depth)[..., None])
    # print(rgb.shape)
    # Apply the volumetric rendering equation to each ray
    

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    
    #############################  TODO 2.5 BEGIN  ############################

    #compute all the rays from the image


    #sample the points from the rays


    #divide data into batches to avoid memory errors


    #forward pass the batches and concatenate the outputs at the end
    





    # Apply volumetric rendering to obtain the reconstructed image


    #############################  TODO 2.5 END  ############################

    return rec_image