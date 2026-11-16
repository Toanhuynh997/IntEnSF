import torch
import numpy as np
import torch.nn.functional as F

def gen_weight_grid(block_res, expand_ratio):
    if expand_ratio == 1:
        return torch.ones((1, 1, block_res, block_res))
    
    grid_size = expand_ratio*2 - 1
    total_res =  block_res * expand_ratio

    # get 1d basis function template 
    x_vec = torch.linspace(0, np.pi, block_res)
    b_1d_left = (torch.cos(x_vec) + 1) / 2
    b_1d_mid = -(torch.cos(x_vec*2) + 1) / 2 + 1
    b_1d_right = torch.flip(b_1d_left, dims=[0])

    # get 1d basis function grid 
    b_1d_grid = [b_1d_left]
    for i in range(grid_size-2):
        b_1d_grid.append(b_1d_mid)    
    b_1d_grid.append(b_1d_right)
    

    # Create a 2D grid of basis functions by taking outer products of the 1D basis functions
    b_2d_grid = []
    for i, b_1d_x in enumerate(b_1d_grid):
        temp = []
        for j, b_1d_y in enumerate(b_1d_grid):
            b_2d = torch.outer(b_1d_x, b_1d_y)
            temp.append(b_2d)
        b_2d_grid.append(temp)


    # pad into full grid
    b_2d_grid_padded = []
    for i in range(grid_size):
        temp = []
        for j in range(grid_size):
            b_2d_padded = torch.zeros((total_res, total_res))
            loc_x = i*block_res//2
            loc_y = j*block_res//2
            b_2d_padded[loc_x:loc_x+block_res, loc_y:loc_y+block_res] = b_2d_grid[i][j]
            temp.append(b_2d_padded)

        b_2d_grid_padded.append(temp)

    # calculate the total weight for each padded block
    # this is used to normalize the weight of each padded block
    w_total = torch.zeros((total_res, total_res))
    for i in range(grid_size):
        for j in range(grid_size):
            w_total += b_2d_grid_padded[i][j]



    # adjust weight to all padded weight 
    b_2d_grid_padded_adjusted = []
    for i in range(grid_size):
        temp = []
        for j in range(grid_size):
            temp.append(b_2d_grid_padded[i][j] / w_total)
        b_2d_grid_padded_adjusted.append(temp)


    # get back the original block res weight 
    b_2d_grid_adjusted = []
    for i in range(grid_size):
        temp = []
        for j in range(grid_size):
            loc_x = i*block_res//2
            loc_y = j*block_res//2
            temp.append(b_2d_grid_padded_adjusted[i][j][loc_x:loc_x+block_res, loc_y:loc_y+block_res])

        b_2d_grid_adjusted.append(temp)

    b_2d_grid_adjusted = torch.stack([torch.stack(row) for row in b_2d_grid_adjusted])
    return b_2d_grid_adjusted.contiguous()



def f2p(fields, patch_size: int = 32, stride: int = 16):
    B, C, H, W = fields.shape

    # Step 1: Extract patches
    patches = F.unfold(fields, kernel_size=patch_size, stride=stride)  # (B, C*ps*ps, num_patches)
    num_patches = patches.shape[-1]
    grid_h = grid_w = int(num_patches**0.5)
    assert grid_h * grid_w == num_patches, "Image must produce square patch grid"

    # Step 2: Reshape to (B * num_patches, C, patch_size, patch_size)
    patches = patches.transpose(1, 2).reshape(B * num_patches, C, patch_size, patch_size)
    return patches



def p2f(processed_patches, weight_grid_flat):
    num_patches = weight_grid_flat.shape[0]
    patch_size = weight_grid_flat.shape[2]
    C = processed_patches.shape[1]
    B = processed_patches.shape[0] // num_patches
    expand_ratio = (np.sqrt(num_patches) + 1).astype(int) // 2
    H = expand_ratio * patch_size
    W = expand_ratio * patch_size
    stride = patch_size // 2
    
    # Step 4: Multiply with shared weight grid
    # weights: (grid_h, grid_w, patch_size, patch_size) -> (num_patches, 1, patch_size, patch_size)
    # weight_grid = weight_grid.reshape(num_patches, 1, patch_size, patch_size)

    # Expand and apply weights
    processed_patches = processed_patches.view(B, num_patches, C, patch_size, patch_size)
    weighted = processed_patches * weight_grid_flat.unsqueeze(0)  # (B, num_patches, C, patch_size, patch_size)
    weighted = weighted.view(B * num_patches, C, patch_size, patch_size)

    # Step 5: Reshape to (B, C * patch_size * patch_size, num_patches)
    weighted = weighted.view(B, num_patches, C * patch_size * patch_size).transpose(1, 2)

    # Step 6: Fold back to image
    reconstructed = F.fold(weighted, output_size=(H, W), kernel_size=patch_size, stride=stride)

    return reconstructed
