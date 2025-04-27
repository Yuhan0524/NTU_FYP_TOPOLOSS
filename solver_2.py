import os
import numpy as np
import torchvision
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AE_Net, FreeU_Net
from tqdm import tqdm
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from comp_pearson_corr import pearson_correlation
######################################topo loss#################################################

from PIL import Image
import numpy as np
from gtda.images import Binarizer, RadialFiltration, HeightFiltration
from gtda.homology import CubicalPersistence
from gudhi.hera import wasserstein_distance
import torch
import gudhi

binarizer = Binarizer(threshold=0.5)  
radial_filtration_1 = HeightFiltration(direction=np.array([0,-1])) 
radial_filtration_2 = HeightFiltration(direction=np.array([1,0]))
radial_filtration_3 = HeightFiltration(direction=np.array([0,1])) 
radial_filtration_4 = HeightFiltration(direction=np.array([-1,0])) 

def process_image(image, binarizer, radial_filtration_1,radial_filtration_2):
    im_binarized = binarizer.fit_transform(image[None, :, :])
    im_filtration_inverse = radial_filtration_1.fit_transform(im_binarized)
    im_binarized = np.where(im_binarized == 0, 1, 0)
    im_filtration = radial_filtration_2.fit_transform(im_binarized)
    im_filtration[im_filtration == 256] = im_filtration_inverse[im_filtration == 256]+256
    cubical_persistence = gudhi.CubicalComplex(
        top_dimensional_cells=im_filtration,
    )
    cubical_persistence.compute_persistence()
    diag = cubical_persistence.persistence_intervals_in_dimension(1)
    diag_0 = cubical_persistence.persistence_intervals_in_dimension(0)

    return diag,diag_0

'''
def topoloss_radial(predictions, targets):
    
    global binarizer, radial_filtration_1, radial_filtration_2
    
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    pred_diag, pred_diag_0 = process_image(predictions, binarizer, radial_filtration_1, radial_filtration_2) 
    targ_diag, targ_diag_0 = process_image(targets, binarizer, radial_filtration_1, radial_filtration_2) 
    #return (wasserstein_distance(pred_diag, targ_diag, order=1, internal_p=2)+wasserstein_distance(pred_diag_0, targ_diag_0, order=1, internal_p=2))/2
    return wasserstein_distance(pred_diag, targ_diag, order=1, internal_p=2)


def topoloss_radius(predictions, targets):
    
    global binarizer, radial_filtration_3, radial_filtration_4
    
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    
    pred_diag, pred_diag_0 = process_image(predictions, binarizer, radial_filtration_3, radial_filtration_4) 
    targ_diag, targ_diag_0 = process_image(targets, binarizer, radial_filtration_3, radial_filtration_4) 
    #return (wasserstein_distance(pred_diag, targ_diag, order=1, internal_p=2)+wasserstein_distance(pred_diag_0, targ_diag_0, order=1, internal_p=2))/2
    return wasserstein_distance(pred_diag, targ_diag, order=1, internal_p=2)


def topo_loss(SR, GT):
    batch_size = SR.shape[0]
    total_loss = 0.0 
    for i in range(batch_size): 
        loss = topoloss_radial(SR[i][0], GT[i][0]) 
        total_loss += loss 
        return torch.tensor(total_loss / batch_size, requires_grad=True)

def topo_loss_radius(SR, GT):
    batch_size = SR.shape[0]
    total_loss = 0.0 
    for i in range(batch_size): 
        loss = topoloss_radius(SR[i][0], GT[i][0]) 
        total_loss += loss 
    return torch.tensor(total_loss / batch_size, requires_grad=True)
    

class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_topological=0.005,weight_topological_radius=0.005):
        super(CombinedLoss, self).__init__()
        self.weight_topological = weight_topological
        self.weight_topological_radius = weight_topological_radius

    def forward(self, predictions, targets,predictions_2, targets_2):
        bce_loss = F.binary_cross_entropy(predictions_2, targets_2)
        topoloss = topo_loss(predictions, targets)
        topoloss_radius = topo_loss_radius(predictions, targets)
        combined_loss = bce_loss + self.weight_topological * topoloss + self.weight_topological_radius * topoloss_radius
        return bce_loss, combined_loss
'''

from concurrent.futures import ThreadPoolExecutor

# Assumes process_image is defined elsewhere

def compute_topological_losses(SR, GT, binarizer, filtrations):
    radial_filtration_1, radial_filtration_2, radial_filtration_3, radial_filtration_4 = filtrations

    SR = SR.cpu().detach().numpy()
    GT = GT.cpu().detach().numpy()

    def process_pair(sr, gt):
        # Radial filtration
        pred_diag_radial, pred_diag_radial_0 = process_image(sr, binarizer, radial_filtration_1, radial_filtration_2)
        targ_diag_radial, targ_diag_radial_0 = process_image(gt, binarizer, radial_filtration_1, radial_filtration_2)
        radial_loss = wasserstein_distance(pred_diag_radial, targ_diag_radial, order=1, internal_p=2) 

        # Radius filtration
        pred_diag_radius, pred_diag_radius_0 = process_image(sr, binarizer, radial_filtration_3, radial_filtration_4)
        targ_diag_radius, targ_diag_radius_0 = process_image(gt, binarizer, radial_filtration_3, radial_filtration_4)
        radius_loss = wasserstein_distance(pred_diag_radius, targ_diag_radius, order=1, internal_p=2)

        return radial_loss, radius_loss

    # Parallel processing with ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pair, SR[:, 0], GT[:, 0]))

    radial_losses, radius_losses = zip(*results)
    mean_radial_loss = np.mean(radial_losses)
    mean_radius_loss = np.mean(radius_losses)

    return torch.tensor((mean_radial_loss + mean_radius_loss) / 2, requires_grad=True)


class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_topological=0.0005, weight_topological_radius=0.0005):
        super(CombinedLoss, self).__init__()
        self.weight_topological = weight_topological
        self.weight_topological_radius = weight_topological_radius
        self.binarizer = binarizer
        self.filtrations = (
            radial_filtration_1,
            radial_filtration_2,
            radial_filtration_3,
            radial_filtration_4,
        )

    def forward(self, predictions, targets, predictions_2, targets_2):
        bce_loss = F.binary_cross_entropy(predictions_2, targets_2)
        topological_loss = compute_topological_losses(predictions, targets, self.binarizer, self.filtrations)
        combined_loss = bce_loss + self.weight_topological * topological_loss
        return bce_loss, combined_loss



########################################################################

'''
from PIL import Image
import numpy as np
from gtda.images import Binarizer
from gtda.images import RadialFiltration
from gtda.images import HeightFiltration
from gtda.homology import CubicalPersistence
from gudhi.hera import wasserstein_distance
import gudhi
import cv2 as cv
import math

binarizer = Binarizer(threshold=0.5)  
radial_filtration_1 = HeightFiltration(direction=np.array([0,-1])) 
radial_filtration_3 = HeightFiltration(direction=np.array([0,1])) 

radial_filtration_2 = HeightFiltration(direction=np.array([1,0])) 
radial_filtration_4 = HeightFiltration(direction=np.array([-1,0])) 
import numpy as np

def wasserstein(targ_diag_0):
    targ_diag_0 = np.array(targ_diag_0)    
    valid_pairs = targ_diag_0[targ_diag_0[:, 1] != np.inf]    
    midpoints = (valid_pairs[:, 0] + valid_pairs[:, 1]) / 2    
    p1_mid = (valid_pairs[:, 0] - midpoints) / 2
    q1_mid = (valid_pairs[:, 1] - midpoints) / 2    
    dist = np.sum(np.sqrt(p1_mid ** 2 + q1_mid ** 2))
    return dist


def process_image(image, binarizer, radial_filtration_1, radial_filtration_2):
    im_binarized = binarizer.fit_transform(image[None, :, :])
    im_filtration_inverse = radial_filtration_1.fit_transform(im_binarized)
    im_binarized = np.where(im_binarized == 0, 1, 0)
    im_filtration = radial_filtration_2.fit_transform(im_binarized)
    im_filtration[im_filtration == 256] = im_filtration_inverse[im_filtration == 256]+256
    result = im_filtration + image[None, :, :]
    cubical_persistence = gudhi.CubicalComplex(
        top_dimensional_cells=result,
    )
    cubical_persistence.compute_persistence()
    diag_0 = cubical_persistence.persistence_intervals_in_dimension(0)
    diag_1 = cubical_persistence.persistence_intervals_in_dimension(1)
    return result, diag_0, diag_1

def topoloss_LB(predictions, targets):
    """
    Compute the Wasserstein distance between the persistent diagrams of two images.
    
    Args:
    predictions (torch.Tensor): Predicted binary images.
    targets (torch.Tensor): Ground truth binary images.
    threshold (float): Threshold for binarization.
    center (list): Center coordinates for radial filtration.

    Returns:
    float: Wasserstein distance between the persistent diagrams.
    """
    global binarizer,radial_filtration_1,radial_filtration_2
    
    #predictions = predictions.cpu().detach().numpy()
    #targets = targets.cpu().detach().numpy()
    SR, pred_diag_0, pred_diag_1= process_image(predictions, binarizer, radial_filtration_1, radial_filtration_2) 
    GT, targ_diag_0, targ_diag_1 = process_image(targets, binarizer, radial_filtration_1, radial_filtration_2) 
    if np.array_equal(pred_diag_0, np.array([[0, np.inf]])) :
        distance = wasserstein(targ_diag_0)
        matching = [[-1,i] for i in range(len(targ_diag_0))]
        return SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, [distance, matching],wasserstein_distance(pred_diag_1, targ_diag_1, order=1, internal_p=2, matching = True)    
    elif np.array_equal(targ_diag_0, np.array([[0, np.inf]])):
        distance = wasserstein(pred_diag_0)
        matching = [[i, -1] for i in range(len(pred_diag_0))]
        return SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, [distance, matching],wasserstein_distance(pred_diag_1, targ_diag_1, order=1, internal_p=2, matching = True)    
    return SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, wasserstein_distance(pred_diag_0, targ_diag_0, order=1, internal_p=2, matching = False),wasserstein_distance(pred_diag_1, targ_diag_1, order=1, internal_p=2, matching = False)    

def topoloss_RT(predictions, targets):
    """
    Compute the Wasserstein distance between the persistent diagrams of two images.
    
    Args:
    predictions (torch.Tensor): Predicted binary images.
    targets (torch.Tensor): Ground truth binary images.
    threshold (float): Threshold for binarization.
    center (list): Center coordinates for radial filtration.

    Returns:
    float: Wasserstein distance between the persistent diagrams.
    """
    global binarizer, radial_filtration_3, radial_filtration_4
    
    #predictions = predictions.cpu().detach().numpy()
    #targets = targets.cpu().detach().numpy()
    SR, pred_diag_0, pred_diag_1 = process_image(predictions, binarizer, radial_filtration_3, radial_filtration_4) 
    GT, targ_diag_0, targ_diag_1 = process_image(targets, binarizer, radial_filtration_3, radial_filtration_4) 
    if np.array_equal(pred_diag_0, np.array([[0, np.inf]])) :
        distance = wasserstein(targ_diag_0)
        matching = [[-1,i] for i in range(len(targ_diag_0))]
        return SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, [distance, matching],wasserstein_distance(pred_diag_1, targ_diag_1, order=1, internal_p=2, matching = True)    
    elif np.array_equal(targ_diag_0, np.array([[0, np.inf]])):
        distance = wasserstein(pred_diag_0)
        matching = [[i, -1] for i in range(len(pred_diag_0))]
        return SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, [distance, matching],wasserstein_distance(pred_diag_1, targ_diag_1, order=1, internal_p=2, matching = True)    
    return SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, wasserstein_distance(pred_diag_0, targ_diag_0, order=1, internal_p=2, matching = True),wasserstein_distance(pred_diag_1, targ_diag_1, order=1, internal_p=2, matching = True)    

'''
'''
def optimized_gradient(SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0, matching_1):
    gradient_matrix = np.zeros_like(SR, dtype=float)
    def calculate_gradient(birth_SR, death_SR, birth_GT, death_GT):
        denominator = np.sqrt((birth_SR - birth_GT)**2 + (death_SR - death_GT)**2)
        if denominator == 0:
            return 0, 0
        gradient_birth = (birth_SR - birth_GT) / denominator
        gradient_death = (death_SR - death_GT) / denominator
        return gradient_birth, gradient_death
    def update_gradient(critical_points, gradient_value):
        for i, j in critical_points:
            abs_grad = np.abs(gradient_value)
            if abs_grad >= np.abs(gradient_matrix[i, j]):
                gradient_matrix[i, j] = gradient_value
    for pair in matching_0:
        if pair[0] != -1 and pair[1] != -1:
            if pred_diag_0[pair[0]][1] == np.inf or targ_diag_0[pair[1]][1] == np.inf:
                continue 
            birth_SR = pred_diag_0[pair[0]][0]
            death_SR = pred_diag_0[pair[0]][1]
            birth_GT = targ_diag_0[pair[1]][0]
            death_GT = targ_diag_0[pair[1]][1]
        elif pair[1] == -1:
            if pred_diag_0[pair[0]][1] == np.inf:
                continue
            birth_SR = pred_diag_0[pair[0]][0]
            death_SR = pred_diag_0[pair[0]][1]
            birth_GT = death_GT = (birth_SR + death_SR) / 2

        gradient_birth, gradient_death = calculate_gradient(birth_SR, death_SR, birth_GT, death_GT)
        critical_birth = np.argwhere(SR == birth_SR)
        critical_death = np.argwhere(SR == death_SR)
        update_gradient(critical_birth, gradient_birth)
        update_gradient(critical_death, gradient_death)
    for pair in matching_1:
        if pair[0] != -1 and pair[1] != -1:
            birth_SR = pred_diag_1[pair[0]][0]
            death_SR = pred_diag_1[pair[0]][1]
            birth_GT = targ_diag_1[pair[1]][0]
            death_GT = targ_diag_1[pair[1]][1]
        elif pair[1] == -1:
            birth_SR = pred_diag_1[pair[0]][0]
            death_SR = pred_diag_1[pair[0]][1]
            birth_GT = death_GT = (birth_SR + death_SR) / 2

        gradient_birth, gradient_death = calculate_gradient(birth_SR, death_SR, birth_GT, death_GT)
        critical_birth = np.argwhere(SR == birth_SR)
        critical_death = np.argwhere(SR == death_SR)
        update_gradient(critical_birth, gradient_birth)
        update_gradient(critical_death, gradient_death)

    return gradient_matrix
'''
'''
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def calculate_gradient(birth_SR, death_SR, birth_GT, death_GT):
    # Vectorized gradient computation
    denominator = np.sqrt((birth_SR - birth_GT)**2 + (death_SR - death_GT)**2)
    
    # Avoid division by zero by replacing 0 with a small epsilon value
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    gradient_birth = (birth_SR - birth_GT) / denominator
    gradient_death = (death_SR - death_GT) / denominator
    return gradient_birth, gradient_death


def update_gradient(critical_points, gradient_value, gradient_matrix):
    # Efficiently update the gradient matrix
    i, j = critical_points.T  # Unpack the points
    abs_grad = np.abs(gradient_value)
    abs_grad_matrix = np.abs(gradient_matrix[i, j])
    mask = abs_grad >= abs_grad_matrix  # Mask where the gradient should be updated
    gradient_matrix[i[mask], j[mask]] = gradient_value

def optimized_gradient(SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0, matching_1):
    gradient_matrix = np.zeros_like(SR, dtype=float)

    # Pre-filter matching pairs and handle edge cases
    matching_0_filtered = [
        pair for pair in matching_0 if pair[0] != -1 and pair[1] != -1 and not (pred_diag_0[pair[0]][1] == np.inf or targ_diag_0[pair[1]][1] == np.inf)
    ]
    matching_1_filtered = [
        pair for pair in matching_1 if pair[0] != -1 and pair[1] != -1
    ]

    def process_matching(pair, pred_diag, targ_diag, SR, gradient_matrix):
        # Process a single pair of matching diagrams
        birth_SR = pred_diag[pair[0]][0]
        death_SR = pred_diag[pair[0]][1]
        birth_GT = targ_diag[pair[1]][0]
        death_GT = targ_diag[pair[1]][1]

        gradient_birth, gradient_death = calculate_gradient(birth_SR, death_SR, birth_GT, death_GT)
        
        # Find critical points (indices of birth and death values)
        critical_birth = np.argwhere(SR == birth_SR)
        critical_death = np.argwhere(SR == death_SR)

        # Update gradients
        update_gradient(critical_birth, gradient_birth, gradient_matrix)
        update_gradient(critical_death, gradient_death, gradient_matrix)

    # Use ThreadPoolExecutor to parallelize the gradient calculation
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks to process filtered matching pairs
        futures = []
        for pair in matching_0_filtered:
            futures.append(executor.submit(process_matching, pair, pred_diag_0, targ_diag_0, SR, gradient_matrix))

        for pair in matching_1_filtered:
            futures.append(executor.submit(process_matching, pair, pred_diag_1, targ_diag_1, SR, gradient_matrix))

        # Wait for all futures to complete
        for future in futures:
            future.result()

    return gradient_matrix



def optimized_combine_gradients(gradient_LB, gradient_RT):
    return np.where(np.abs(gradient_LB) >= np.abs(gradient_RT), gradient_LB, gradient_RT)


def loss_gradient(image_array_SR, image_array_GT):
    topoloss = 0
    SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0, matching_1 = topoloss_LB(image_array_SR, image_array_GT)
    gradient_LB = optimized_gradient(SR[0], GT[0], pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0[1], matching_1[1])
    topoloss = (matching_0[0]+matching_1[0])/2
    SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0, matching_1 = topoloss_RT(image_array_SR, image_array_GT)
    gradient_RT = optimized_gradient(SR[0], GT[0], pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0[1], matching_1[1])
    gradient_all = optimized_combine_gradients(gradient_LB, gradient_RT)
    #gradient_all = gradient_LB +gradient_RT
    topoloss = topoloss + (matching_0[0]+matching_1[0])/2
    return topoloss, gradient_all


'''
'''
import torch.multiprocessing as mp

# Ensure the start method is 'spawn' to prevent issues with non-leaf tensors
mp.set_start_method('spawn', force=True)

def compute_topoloss_and_gradient(i, SR_probs, GT, device, return_dict):
    # Detach tensors to remove gradient tracking for multiprocessing
    SR_prob_detached = SR_probs[i][0].detach()
    GT_detached = GT[i][0].detach()

    topoloss, grad_all = loss_gradient(SR_prob_detached, GT_detached)
    return_dict[i] = (topoloss, grad_all)

def combined_loss_gradient(self, SR_probs, GT, SR_flat, GT_flat):
    total_topoloss = 0
    total_gradients = torch.zeros_like(SR_probs)

    manager = mp.Manager()
    return_dict = manager.dict()

    # Ensure the multiprocessing uses 'spawn'
    processes = []

    for i in range(SR_probs.size(0)):
        p = mp.Process(target=compute_topoloss_and_gradient, args=(i, SR_probs, GT, self.device, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for i in range(SR_probs.size(0)):
        topoloss, grad_all = return_dict[i]
        total_topoloss += topoloss
        total_gradients += 0.01 * torch.from_numpy(grad_all).to(self.device)

    batch_topoloss = total_topoloss / SR_probs.size(0)
    bce_loss = self.criterion(SR_flat, GT_flat)
    combined_loss = bce_loss + 0.01 * batch_topoloss
    print(combined_loss)

    return total_gradients, combined_loss, batch_topoloss


'''
'''
def combined_loss_gradient(self, SR_probs, GT, SR_flat, GT_flat):
    total_topoloss = 0
    # Initialize total_gradients as a NumPy array, matching the shape of SR_probs.
    total_gradients = np.zeros_like(SR_probs.cpu().detach().numpy())  # Ensure it's a NumPy array on the CPU

    for i in range(SR_probs.size(0)): 
        topoloss, grad_all = loss_gradient(SR_probs[i][0], GT[i][0])
        total_topoloss += topoloss
        total_gradients += 0.01 * grad_all 

    # Convert total_gradients to a PyTorch tensor and move it to the correct device
    total_gradients = torch.tensor(total_gradients, dtype=torch.float32).to(self.device)

    batch_topoloss = total_topoloss / SR_probs.size(0)
    bce_loss = self.criterion(SR_flat, GT_flat)
    combined_loss = bce_loss + 0.01 * batch_topoloss
    return total_gradients, combined_loss, batch_topoloss
'''
'''
import numpy as np
import multiprocessing as mp

def loss(image_array_SR, image_array_GT):
    topoloss = 0
    SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0, matching_1 = topoloss_LB(image_array_SR, image_array_GT)
    topoloss = (matching_0[0]+matching_1[0])/2
    SR, GT, pred_diag_0, pred_diag_1, targ_diag_0, targ_diag_1, matching_0, matching_1 = topoloss_RT(image_array_SR, image_array_GT)
    topoloss = topoloss + (matching_0[0]+matching_1[0])/2
    return topoloss

def compute_loss(i, SR_probs, GT):
    topoloss = loss(SR_probs[i][0], GT[i][0])
    return topoloss


def combined(self, SR_probs, GT, SR_flat, GT_flat):
    SR_probs = SR_probs.cpu().detach().numpy()
    GT = GT.cpu().detach().numpy()
    total_topoloss = 0
    total_gradients = np.zeros_like(SR_probs)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(compute_loss_and_gradients, [(i, SR_probs, GT) for i in range(SR_probs.shape[0])])
    
    for topoloss, grad_all in results:
        total_topoloss += topoloss
        total_gradients += 0.01 * grad_all
    
    total_gradients = torch.tensor(total_gradients, dtype=torch.float32).to(self.device)

    batch_topoloss = 0.01 * total_topoloss / SR_probs.shape[0]
    bce_loss = self.criterion(SR_flat, GT_flat)
    combined_loss = bce_loss + batch_topoloss

    return total_gradients, combined_loss, batch_topoloss



def compute_loss_and_gradients(i, SR_probs, GT):
    topoloss, grad_all = loss_gradient(SR_probs[i][0], GT[i][0])
    return topoloss, grad_all

def combined_loss_gradient(self, SR_probs, GT, SR_flat, GT_flat):
    SR_probs = SR_probs.cpu().detach().numpy()
    GT = GT.cpu().detach().numpy()
    total_topoloss = 0
    total_gradients = np.zeros_like(SR_probs)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(compute_loss_and_gradients, [(i, SR_probs, GT) for i in range(SR_probs.shape[0])])
    
    for topoloss, grad_all in results:
        total_topoloss += topoloss
        total_gradients += 0.01 * grad_all
    
    total_gradients = torch.tensor(total_gradients, dtype=torch.float32).to(self.device)

    batch_topoloss = 0.01 * total_topoloss / SR_probs.shape[0]
    bce_loss = self.criterion(SR_flat, GT_flat)
    combined_loss = bce_loss + batch_topoloss

    return total_gradients, combined_loss, batch_topoloss

'''


########################################################################

def f_beta_loss(preds, labels, beta=1, threshold=0.5):
    epsilon = 1e-7  # used to prevent division by zero
    preds = (preds > threshold).float()

    true_positives = torch.sum(preds * labels)
    false_positives = torch.sum(preds * (1 - labels))
    false_negatives = torch.sum((1 - preds) * labels)

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)

    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (precision * recall) / \
                   ((beta_squared * precision) + recall + epsilon)

    f_beta_loss = 1. / (f_beta_score.mean() + epsilon)

    return f_beta_loss

def concat_images(image_paths, output_path):

    images = [Image.open(x) for x in image_paths]
    width, height = images[0].size
    total_width = width * len(images)
    new_image = Image.new('RGBA', (total_width, height))
    x_offset = 0

    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += width
    new_image.save(output_path)


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        self.config = config
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.focus_beta = config.focus_beta

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path + config.special_save_folder_name
        self.result_path = config.result_path
        self.mode = config.mode

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{config.cuda_idx}')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

        self.epoch = 0  # Initialize the starting epoch
        self.start_epoch = 0
        self.checkpoint_dir = "/raid/crp.dssi/volume_Kubernetes/Benquan/Topological_Network/Network/QRUNET_60_BCE_0424"



    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


    def load_checkpoint(self, checkpoint_path):
        """
        Loads the model and optimizer state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
            return start_epoch
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")


    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            # self.unet = U_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet = U_Net(img_ch=1, output_ch=self.config.output_ch)

            # ===================== try double ====================================== #
            # self.unet = self.unet.double()
            # ====================================================================== #


        elif self.model_type =='FreeU_Net':
            self.unet = FreeU_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)
        elif self.model_type == 'AE_Net_step1':
            self.unet = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet.load_state_dict(torch.load(
                '/home/benquan/AE_Net_step1-250-0.0000-114-0.3052.pkl'))
        elif self.model_type == 'AE_Net_step2':
            self.unet = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet_mask = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet_mask.load_state_dict(torch.load(
                '/home/benquan/AE_Net_step1-250-0.0000-114-0.3052.pkl'))
            self.unet_mask.to(self.device)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=self.config.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=self.config.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=self.config.output_ch, t=self.t)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2], weight_decay=self.config.wd)
        self.unet.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self, start_epoch=0):
        # ====================================== Training ===========================================#
        # ===========================================================================================#

        special_save_name = self.config.special_save_name
        unet_path = os.path.join(self.model_path,
                                 f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                                 f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')

        if not os.path.exists(os.path.join(self.model_path, 'train_valid_records')):
            os.makedirs(os.path.join(self.model_path, 'train_valid_records'))
        writer = SummaryWriter(os.path.join(self.model_path, 'train_valid_records',
                                            f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                                            f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}'))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            print('No training is executed.')
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            
            
            train_outputs = []
            valid_outputs = []
            for epoch in range(start_epoch, self.num_epochs):
                

                self.unet.train(True)
                epoch_loss = 0
                epoch_focus_loss = 0

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                CR = 0
                length = 0

           

                '''

                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.train_loader)):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = self.unet(images)
                    SR_probs = torch.sigmoid(SR)                     
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    total_gradients, combined_loss, batch_topoloss = combined_loss_gradient(self, SR_probs, GT, SR_flat, GT_flat)

                    bce_grad = SR_probs - GT  # BCE gradient is (SR - GT) * Sigmoid'(SR), which simplifies as SR_probs - GT
                    bce_grad = bce_grad.view_as(SR)
                    #grad_all_tensor = torch.tensor(total_gradients, dtype=bce_grad.dtype, device=bce_grad.device)  # Ensure type and device match
                    grad_all_tensor = total_gradients.clone().detach().requires_grad_(True)
                                        
                    def custom_gradient_hook(grad):
                        return bce_grad + 0.5 * grad_all_tensor
                        #return bce_grad + grad_all_tensor


                    # Attach the hook to the output layer
                    SR.register_hook(custom_gradient_hook)

                    # Perform backpropagation as usual
                    self.reset_grad()
                    combined_loss.backward()
                    self.optimizer.step()

                    epoch_loss += combined_loss.item()
                    epoch_focus_loss += batch_topoloss.item()

                    acc += get_accuracy(SR_probs, GT)
                    SE += get_sensitivity(SR_probs, GT)
                    SP += get_specificity(SR_probs, GT)
                    PC += get_precision(SR_probs, GT)
                    F1 += get_F1(SR_probs, GT)
                    JS += get_JS(SR_probs, GT)
                    DC += get_DC(SR_probs, GT)
                    CR += pearson_correlation(GT, SR_probs)
                    # length += images.size(0)
                    length += 1
                '''

                '''
                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.train_loader)):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = self.unet(images)
                    SR_probs = torch.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    loss_fn = CombinedLoss()
                    bce_loss, total_loss = loss_fn(SR,GT,SR_flat, GT_flat)  
                    self.reset_grad()
                    total_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)


                    self.optimizer.step()
                    epoch_loss += total_loss.item()
                    epoch_focus_loss += bce_loss
                    acc += get_accuracy(SR_probs, GT)
                    SE += get_sensitivity(SR_probs, GT)
                    SP += get_specificity(SR_probs, GT)
                    PC += get_precision(SR_probs, GT)
                    F1 += get_F1(SR_probs, GT)
                    JS += get_JS(SR_probs, GT)
                    DC += get_DC(SR_probs, GT)
                    CR += pearson_correlation(GT, SR_probs)
                    # length += images.size(0)
                    length += 1
                '''
                
                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.train_loader)):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = self.unet(images)
                    
                    SR_probs = torch.sigmoid(SR)
                    
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)

                    bce_loss = self.criterion(SR_flat, GT_flat)  # Base BCE loss
                    
                    epoch_loss += bce_loss.item()
                    
                    # Backpropagate only BCE loss
                    self.reset_grad()
                    #self.optimizer.zero_grad()

                    bce_loss.backward()  # Only backpropagate BCE loss
                    
                    # Optimizer step to update model parameters
                    self.optimizer.step()

                    epoch_focus_loss += bce_loss.item()
                    acc += get_accuracy(SR_probs, GT)
                    SE += get_sensitivity(SR_probs, GT)
                    SP += get_specificity(SR_probs, GT)
                    PC += get_precision(SR_probs, GT)
                    F1 += get_F1(SR_probs, GT)
                    JS += get_JS(SR_probs, GT)
                    DC += get_DC(SR_probs, GT)
                    CR += pearson_correlation(GT, SR_probs)
                    # length += images.size(0)
                    length += 1
                    
                # break
                
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                CR = CR / length

                # Print the log info
                print(
                    'Epoch [%d/%d], Total_Loss: %.4f, BCE_loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, CR: %.4f' % (
                        epoch + 1, self.num_epochs, \
                        epoch_loss / length, epoch_focus_loss / length,   \
                        acc, SE, SP, PC, F1, JS, DC, CR))
                train_outputs.append([epoch + 1, epoch_loss / length, epoch_focus_loss / length, acc, SE, SP, PC, F1, JS, DC, CR])
                writer.add_scalar('Train_epoch_loss', epoch_loss / length, epoch)
                writer.add_scalar('Train_SE', SE, epoch)

                torchvision.utils.save_image(images.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_train_%d_image.png' % (
                                                              self.model_type, epoch + 1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                          '%s_train_%d_SR.png' % (
                                                              self.model_type, epoch + 1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_train_%d_GT.png' % (
                                                              self.model_type, epoch + 1)))
                
                if epoch % 100 == 0:
                    state = {
                        'epoch': epoch,
                        'model_state_dict': self.unet.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                    torch.save(state, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")
                    
                    

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    # if current epoch is in the last few epochs
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                with torch.no_grad():
                    self.unet.train(False)
                    self.unet.eval()

                    acc = 0.  # Accuracy
                    SE = 0.  # Sensitivity (Recall)
                    SP = 0.  # Specificity
                    PC = 0.  # Precision
                    F1 = 0.  # F1 Score
                    JS = 0.  # Jaccard Similarity
                    DC = 0.  # Dice Coefficient
                    CR = 0
                    length = 0
                    for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.valid_loader)):
                        images = images.to(self.device)
                        GT = GT.to(self.device)
                        SR = torch.sigmoid(self.unet(images))
                        acc += get_accuracy(SR, GT)
                        SE += get_sensitivity(SR, GT)
                        SP += get_specificity(SR, GT)
                        PC += get_precision(SR, GT)
                        F1 += get_F1(SR, GT)
                        JS += get_JS(SR, GT)
                        DC += get_DC(SR, GT)
                        CR += pearson_correlation(GT, SR)
                        # length += images.size(0)
                        length += 1
                    # break

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                CR = CR / length
                # unet_score = JS + DC
                # unet_score = acc
                unet_score = SE

                print(
                    '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, DC: %.4f' % (
                        acc, SE, SP, PC, F1, JS, DC, CR))

                # writer.add_scalar('Valid_acc', acc, epoch)
                writer.add_scalar('Valid_SE', SE, epoch)
                valid_outputs.append([epoch + 1, acc, SE, SP, PC, F1, JS, DC, CR])

                # use threshold to preprocess SR img
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)

                #torchvision.utils.save_image(images.data.cpu(),
                #                             os.path.join(self.result_path,
                #                                          '%s_valid_%d_image.png' % (
                #                                          self.model_type, epoch + 1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_valid_%d_SR.png' % (
                                                          self.model_type, epoch + 1)))
                #torchvision.utils.save_image(SR_075.data.cpu(),
                #                             os.path.join(self.result_path,
                #                                          '%s_valid_%d_SR_075.png' % (
                #                                          self.model_type, epoch + 1)))
                #torchvision.utils.save_image(SR_050.data.cpu(),
                #                             os.path.join(self.result_path,
                #                                          '%s_valid_%d_SR_050.png' % (
                #                                          self.model_type, epoch + 1)))
                #torchvision.utils.save_image(SR_030.data.cpu(),
                #                             os.path.join(self.result_path,
                #                                          '%s_valid_%d_SR_030.png' % (
                #                                          self.model_type, epoch + 1)))
                #torchvision.utils.save_image(SR_010.data.cpu(),
                #                             os.path.join(self.result_path,
                #                                          '%s_valid_%d_SR_010.png' % (
                #                                          self.model_type, epoch + 1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.result_path,
                                                          '%s_valid_%d_GT.png' % (
                                                          self.model_type, epoch + 1)))

                # Save Best U-Net model
                if unet_score >= best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    # unet_path = os.path.join(self.model_path,
                    #                          f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                    #                          f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')
                    # add in epoch name in the unet path
                    unet_path = os.path.join(self.model_path,
                                             f'{self.model_type}_epoch_{epoch + 1}_lr_{self.lr}_'
                                             f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')
                    torch.save(best_unet, unet_path)
                torch.cuda.empty_cache()


                if epoch % 50 == 0:
                    best_unet = self.unet.state_dict()
                    # unet_path = os.path.join(self.model_path,
                    #                          f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                    #                          f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')
                    # add in epoch name in the unet path
                    unet_path = os.path.join(self.model_path,
                                             f'{self.model_type}_epoch_{epoch + 1}_lr_{self.lr}_'
                                             f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')
                    torch.save(best_unet, unet_path)
                torch.cuda.empty_cache()


                if epoch % 50 == 0:
                    file_path = os.path.join(self.result_path, f"outputs{epoch}.txt")
                    
                    with open(file_path, 'w') as f:
                        f.write("Epoch\tTrain Output\tValid Output\n")
                        for i, (train_out, val_out) in enumerate(zip(train_outputs, valid_outputs)):  # Avoid overwriting 'epoch'
                            f.write(f"{i}\t{train_out}\t{val_out}\n")

                    print(f"Outputs saved to {file_path}")
        # ===================================== Test ====================================#
        # del self.unet
        # del best_unet
        # self.build_model()
        # self.unet.load_state_dict(torch.load(unet_path))
        #
        # self.unet.train(False)
        # self.unet.eval()
        #
        # acc = 0.	# Accuracy
        # SE = 0.		# Sensitivity (Recall)
        # SP = 0.		# Specificity
        # PC = 0. 	# Precision
        # F1 = 0.		# F1 Score
        # JS = 0.		# Jaccard Similarity
        # DC = 0.		# Dice Coefficient
        # length=0
        # for i, (images, GT) in enumerate(self.valid_loader):
        #
        # 	images = images.to(self.device)
        # 	GT = GT.to(self.device)
        # 	SR = F.sigmoid(self.unet(images))
        # 	acc += get_accuracy(SR,GT)
        # 	SE += get_sensitivity(SR,GT)
        # 	SP += get_specificity(SR,GT)
        # 	PC += get_precision(SR,GT)
        # 	F1 += get_F1(SR,GT)
        # 	JS += get_JS(SR,GT)
        # 	DC += get_DC(SR,GT)
        #
        # 	length += images.size(0)
        #
        # acc = acc/length
        # SE = SE/length
        # SP = SP/length
        # PC = PC/length
        # F1 = F1/length
        # JS = JS/length
        # DC = DC/length
        # unet_score = JS + DC
        #
        #
        # f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
        # wr = csv.writer(f)
        # wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
        # f.close()
        #
        


    def generate_test_result(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay,
            self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

            # =================================== generate test results one by one ==================================#
            with torch.no_grad():
                self.unet.train(False)
                self.unet.eval()

                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.valid_loader)):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = torch.sigmoid(self.unet(images))

                    # # use threshold to preprocess SR img
                    SR_050 = torch.where(SR > 0.5, 1., 0.)

                    torchvision.utils.save_image(SR_050.data.cpu(),
                                                 os.path.join(self.result_path,
                                                              f'{gt_paths[17:-4]}_test.tif'))

    def test(self, pretrain_path=None):
        unet_path = pretrain_path
        #self.test_result_path = os.path.join(self.result_path,
        #                                     f'test_result_{self.config.selected_test_fold[0]}')
        
        self.test_result_path = self.result_path

        #print(
        #    f'test on fold {self.config.selected_test_fold[0]} and save results to {self.test_result_path}')
        if not os.path.exists(self.test_result_path):
            os.makedirs(self.test_result_path)

        self.unet.load_state_dict(torch.load(unet_path))
        print(f'{self.model_type} is Successfully Loaded from {unet_path}')
        # print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        '''
        # ======================= preparing data for computing correlation value ======================= #
        data_dir = os.path.join(self.config.test_path, self.config.selected_test_fold[0])
        #group_info = np.load(os.path.join(data_dir, 'group.npy'), allow_pickle=True)
        files = os.listdir(data_dir)
        txt = [txt for txt in files if txt.endswith('.txt')]
        for txt in ['truth_count.txt']:
            truth_count = np.loadtxt(os.path.join(data_dir, txt))
        for txt in ['truth_x.txt']:
            truth_x = np.loadtxt(os.path.join(data_dir, txt))
        for txt in ['truth_y.txt']:
            truth_y = np.loadtxt(os.path.join(data_dir, txt))

        datasize_all = np.size(truth_count)  # truth number

        truthsel_x = []
        truthsel_y = []

        best_threshold_distribution = [0, 0, 0, 0]  # 4 thresholds

        for i in range(datasize_all):
            number = truth_count[i]
            inu = int(number)
            truthsel_x.insert(i, truth_x[i][:inu] * 1e6)
            truthsel_y.insert(i, truth_y[i][:inu] * 1e6)

        x_all = truthsel_x
        y_all = truthsel_y
        '''

        '''
        # ===================================== Testing ====================================#
        with torch.no_grad():
            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            epoch_loss = 0.
            auc_roc = 0.
            length = 0
            CR = 0. # Pearson correlation

            accs = []
            SEs = []
            PCs = []
            F1s = []
            auc_rocs = []
            CRs = []

            # create a dictionary to save the results of this subfolder
            '''
        '''
            correlation_result_dict = {
                '2 nanoholes': [],
                '3 nanoholes': [],
                '4 nanoholes': [],
                '5 nanoholes': [],
                '6 nanoholes': [],
                '7 nanoholes': [],
                '8 nanoholes': [],
                '9 nanoholes': [],
                '10 nanoholes': [],
            }
            '''
        '''
            correlation_result_dict = {
                'A4':[]
            }



            for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.test_loader)):
                images = images.to(self.device)
                GT = GT.to(self.device)
                predict_image = self.unet(images)
                predict_image_flatten = predict_image.flatten()

                SR = torch.sigmoid(self.unet(images))

                SR_flat = SR.view(SR.size(0), -1)

                GT_flat = GT.view(GT.size(0), -1)
                loss = self.criterion(SR_flat, GT_flat)
                epoch_loss += loss.item()

                point_acc = get_accuracy(SR, GT)
                point_SE = get_sensitivity(SR, GT)
                point_SP = get_specificity(SR, GT)
                point_PC = get_precision(SR, GT)
                point_F1 = get_F1(SR, GT)
                point_JS = get_JS(SR, GT)
                point_DC = get_DC(SR, GT)
                # GT_binary = np.round(GT.flatten().cpu().numpy())
                # point_auc_roc = roc_auc_score(y_true=GT_binary.flatten(),
                #                               y_score=SR.flatten().cpu().numpy())
                point_CR = pearson_correlation(GT_flat, SR_flat)




                acc += point_acc
                SE += point_SE
                SP += point_SP
                PC += point_PC
                F1 += point_F1
                JS += point_JS
                DC += point_DC
                # auc_roc += point_auc_roc
                CR += point_CR

                accs.append(point_acc)
                SEs.append(point_SE)
                PCs.append(point_PC)
                F1s.append(point_F1)
                # auc_rocs.append(point_auc_roc)
                CRs.append(point_CR)

                # length += images.size(0)
                length += 1

                # use threshold to preprocess SR img
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)



                #print(SR.size(), GT.size())
                # point_raw_correlation = cros_correlation_benquan.cross_correlation(predict=SR.cpu().numpy(), gt=GT.cpu().numpy())
                # point_thres_correlation = cros_correlation_benquan.cross_correlation(predict=SR_050.cpu().numpy(), gt=GT.cpu().numpy())

                # print(f'raw corr: {point_raw_correlation}')
                # print(f'thres corr: {point_thres_correlation}')

                # cor_raw += point_raw_correlation
                # cor_thresh += point_thres_correlation
                # cors_raw.append(point_raw_correlation)
                # cors_thres.append(point_thres_correlation)

                output_list = (SR_075, SR_050, SR_030, SR_010)
                output_SEs = []

                # new_file_path = os.path.join(self.test_result_path,
                #                              'test_%s_idx_%d_oringinal_image.tif' % (
                #                                  self.model_type, i))
                # concat_images(image_path, new_file_path)

                # torchvision.utils.save_image(images.data.cpu(),
                #                              os.path.join(self.test_result_path,
                #                                           'test_%s_idx_%d_image.png' % (
                #                                               self.model_type, i)))

                torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.test_result_path,
                                                          'test_%s_idx_%d_SR.png' % (
                                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_075.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_075.png' % (
                #                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_050.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_050.png' % (
                #                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_030.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_030.png' % (
                #                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_010.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_010.png' % (
                #                                          self.model_type, i)))
                torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.test_result_path,
                                                          'test_%s_idx_%d_GT.png' % (
                                                          self.model_type, i)))

                
                # iterate over all the output images and compute the SE
                for output in output_list:
                    se = get_sensitivity_no_threshold(output, GT)
                    output_SEs.append(se)
                # update thresh distribution
                # select the best threshold index
                #best_threshold_idx = np.argmax(output_SEs)
                # update the best threshold distribution
                #best_threshold_distribution[best_threshold_idx] += 1

                # =========================== compute correlation ============================= #
                # compute correlation in 512*512 size image (!!! 512 512 512 512 !!!)
                # rayleigh_region, center_info = get_pixels_in_circles(i, x_all, y_all, group_info)
                image = cv2.imread(os.path.join(self.test_result_path,
                                                'test_%s_idx_%d_SR_050.png' % (self.model_type, i)))
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_gray = cv2.resize(image_gray, (512, 512))
                gt = cv2.imread(GT_path[0])
                gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

                '''
        '''
                correlation = region_correlation_single_image(rayleigh_region, image_gray, gt_gray)
                for key, value in correlation.items():  # key is x nanoholes
                    if key in correlation_result_dict:
                        correlation_result_dict[key].append(value)
                    if key in two_nanoholes_correlation_result_dict:
                        two_nanoholes_correlation_result_dict['2 nanoholes'].append(value)
                        two_nanoholes_correlation_result_dict['se'].append(get_sensitivity(SR, GT))
                        two_nanoholes_correlation_result_dict['pc'].append(get_precision(SR, GT))
                        two_nanoholes_correlation_result_dict['f1'].append(get_F1(SR, GT))
                        two_nanoholes_correlation_result_dict['auroc'].append(roc_auc_score(y_true=GT.flatten().cpu().numpy(),
                                              y_score=SR.flatten().cpu().numpy()))
                        two_nanoholes_correlation_result_dict['CR'].append(pearson_correlation(GT_flat, SR_flat))
                #
                        points = center_info['2 nanoholes']
                        a, b = points[0], points[1]
                        distance = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
                        two_nanoholes_correlation_result_dict['distance'].append(distance)'
                '''
        '''
        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        epoch_loss = epoch_loss / length
        auc_roc = auc_roc / length
        CR = CR / length
        # unet_score = JS + DC
        unet_score = acc

        print(
            '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, AUROC: %.4f, Pearson: %.4f' % (
                epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR, ))

        # idx order of accs in descending order
        idx_order_acc = np.argsort(accs)[::-1]
        # idx order of SEs in descending order
        idx_order_se = np.argsort(SEs)[::-1]
        # idx order of PCs in descending order
        idx_order_PC = np.argsort(PCs)[::-1]
        # idx order of F1s in descending order
        idx_order_f1 = np.argsort(F1s)[::-1]
        # idx order of auc_rocs in descending order
        idx_order_auc_roc = np.argsort(auc_rocs)[::-1]
        # idx order of cross
        idx_order_CR = np.argsort(CRs)[::-1]

        # str description of the idx order of accs in descending order
        str_idx_order = f'acc: {idx_order_acc} \n  se: {idx_order_se} \n  sp: {idx_order_PC} \n  f1: {idx_order_f1} \n  auc_roc: {idx_order_auc_roc} \n Pearson: {idx_order_CR}'
        print(str_idx_order)

        with open(os.path.join(self.test_result_path, 'metric_result_record.txt'), 'w') as f:
            f.write(
                '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, auroc: %.4f, Pearson: %.4f' % (
                    epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR))
            # write the best threshold distribution
            f.write('\n')
            f.write('best threshold distribution: ')
            f.write(str(best_threshold_distribution))
            f.write('\n')
            f.write('best threshold idx: ')
            f.write(str_idx_order)
            f.write('\n')
            f.write('all SEs: ')
            f.write(str(SEs))
            f.write('\n all Pearsons: ')
            f.write(str(CRs))
            f.write('\n all PCs: ')
            f.write(str(PCs))
            f.write('\n all F1s: ')
            f.write(str(F1s))
            # if 'distance' in two_nanoholes_correlation_result_dict and two_nanoholes_correlation_result_dict['distance']:
            #     f.write('\n all Distances: ')
            #     f.write(str(two_nanoholes_correlation_result_dict['distance']))

        np.save(os.path.join(self.test_result_path,
                             f'correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
                correlation_result_dict)
        if not os.path.exists(os.path.join(self.result_path, 'correlation_all_nanoholes_results')):
            os.makedirs(os.path.join(self.result_path, 'correlation_all_nanoholes_results'))
        np.save(os.path.join(self.result_path, 'correlation_all_nanoholes_results',
                             f'correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
                correlation_result_dict)

        # np.save(os.path.join(self.test_result_path,
        #                      f'two_nanoholes_correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
        #         two_nanoholes_correlation_result_dict)
        # if not os.path.exists(
        #         os.path.join(self.result_path, 'correlation_distance_2_nanoholes_results')):
        #     os.makedirs(os.path.join(self.result_path, 'correlation_distance_2_nanoholes_results'))
        # np.save(os.path.join(self.result_path, 'correlation_distance_2_nanoholes_results',
        #                      f'two_nanoholes_correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
        #         two_nanoholes_correlation_result_dict)'
        '''
        # ===================================== Testing ====================================#
        with torch.no_grad():
            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            epoch_loss = 0.
            auc_roc = 0.
            length = 0
            CR = 0. # Pearson correlation

            accs = []
            SEs = []
            PCs = []
            F1s = []
            auc_rocs = []
            CRs = []

            # create a dictionary to save the results of this subfolder
            correlation_result_dict = {
                '2 nanoholes': [],
                '3 nanoholes': [],
                '4 nanoholes': [],
                '5 nanoholes': [],
                '6 nanoholes': [],
                '7 nanoholes': [],
                '8 nanoholes': [],
                '9 nanoholes': [],
                '10 nanoholes': [],
            }



            for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.test_loader)):
                images = images.to(self.device)
                GT = GT.to(self.device)
                predict_image = self.unet(images)
                predict_image_flatten = predict_image.flatten()

                SR = torch.sigmoid(self.unet(images))

                SR_flat = SR.view(SR.size(0), -1)

                GT_flat = GT.view(GT.size(0), -1)
                loss = self.criterion(SR_flat, GT_flat)
                epoch_loss += loss.item()

                point_acc = get_accuracy(SR, GT)
                point_SE = get_sensitivity(SR, GT)
                point_SP = get_specificity(SR, GT)
                point_PC = get_precision(SR, GT)
                point_F1 = get_F1(SR, GT)
                point_JS = get_JS(SR, GT)
                point_DC = get_DC(SR, GT)
                # GT_binary = np.round(GT.flatten().cpu().numpy())
                # point_auc_roc = roc_auc_score(y_true=GT_binary.flatten(),
                #                               y_score=SR.flatten().cpu().numpy())
                point_CR = pearson_correlation(GT_flat, SR_flat)




                acc += point_acc
                SE += point_SE
                SP += point_SP
                PC += point_PC
                F1 += point_F1
                JS += point_JS
                DC += point_DC
                # auc_roc += point_auc_roc
                CR += point_CR

                accs.append(point_acc)
                SEs.append(point_SE)
                PCs.append(point_PC)
                F1s.append(point_F1)
                # auc_rocs.append(point_auc_roc)
                CRs.append(point_CR)

                # length += images.size(0)
                length += 1

                # use threshold to preprocess SR img
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)
                #
                #
                #
                # #print(SR.size(), GT.size())
                # # point_raw_correlation = cros_correlation_benquan.cross_correlation(predict=SR.cpu().numpy(), gt=GT.cpu().numpy())
                # # point_thres_correlation = cros_correlation_benquan.cross_correlation(predict=SR_050.cpu().numpy(), gt=GT.cpu().numpy())
                #
                # # print(f'raw corr: {point_raw_correlation}')
                # # print(f'thres corr: {point_thres_correlation}')
                #
                # # cor_raw += point_raw_correlation
                # # cor_thresh += point_thres_correlation
                # # cors_raw.append(point_raw_correlation)
                # # cors_thres.append(point_thres_correlation)
                #
                # output_list = (SR_075, SR_050, SR_030, SR_010)
                # output_SEs = []

                # new_file_path = os.path.join(self.test_result_path,
                #                              'test_%s_idx_%d_oringinal_image.tif' % (
                #                                  self.model_type, i))
                # concat_images(image_path, new_file_path)

                # torchvision.utils.save_image(images.data.cpu(),
                #                              os.path.join(self.test_result_path,
                #                                           'test_%s_idx_%d_image.png' % (
                #                                               self.model_type, i)))

                torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.test_result_path,
                                                          'test_%s_idx_%d_SR.png' % (
                                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_075.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_075.png' % (
                #                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_050.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_050.png' % (
                #                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_030.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_030.png' % (
                #                                          self.model_type, i)))
                #torchvision.utils.save_image(SR_010.data.cpu(),
                #                             os.path.join(self.test_result_path,
                #                                          'test_%s_idx_%d_SR_010.png' % (
                #                                          self.model_type, i)))
                torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.test_result_path,
                                                          'test_%s_idx_%d_GT.png' % (
                                                          self.model_type, i)))

                # # iterate over all the output images and compute the SE
        #         # for output in output_list:
        #         #     se = get_sensitivity_no_threshold(output, GT)
        #         #     output_SEs.append(se)
        #         # # update thresh distribution
        #         # # select the best threshold index
        #         # best_threshold_idx = np.argmax(output_SEs)
        #         # # update the best threshold distribution
        #         # best_threshold_distribution[best_threshold_idx] += 1
        #         #
        #         # # =========================== compute correlation ============================= #
        #         # # compute correlation in 512*512 size image (!!! 512 512 512 512 !!!)
        #         # # rayleigh_region, center_info = get_pixels_in_circles(i, x_all, y_all, group_info)
        #         # image = cv2.imread(os.path.join(self.test_result_path,
        #         #                                 'test_%s_idx_%d_SR_050.png' % (self.model_type, i)))
        #         # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #         # image_gray = cv2.resize(image_gray, (512, 512))
        #         # gt = cv2.imread(GT_path[0])
        #         # gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        #
        #         # correlation = region_correlation_single_image(rayleigh_region, image_gray, gt_gray)
        #         # for key, value in correlation.items():  # key is x nanoholes
        #         #     if key in correlation_result_dict:
        #         #         correlation_result_dict[key].append(value)
        #         #     if key in two_nanoholes_correlation_result_dict:
        #         #         two_nanoholes_correlation_result_dict['2 nanoholes'].append(value)
        #         #         two_nanoholes_correlation_result_dict['se'].append(get_sensitivity(SR, GT))
        #         #         two_nanoholes_correlation_result_dict['pc'].append(get_precision(SR, GT))
        #         #         two_nanoholes_correlation_result_dict['f1'].append(get_F1(SR, GT))
        #         #         two_nanoholes_correlation_result_dict['auroc'].append(roc_auc_score(y_true=GT.flatten().cpu().numpy(),
        #         #                               y_score=SR.flatten().cpu().numpy()))
        #         #         two_nanoholes_correlation_result_dict['CR'].append(pearson_correlation(GT_flat, SR_flat))
        #         #
        #         #         points = center_info['2 nanoholes']
        #         #         a, b = points[0], points[1]
        #         #         distance = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        #         #         two_nanoholes_correlation_result_dict['distance'].append(distance)
        #
        # acc = acc / length
        # SE = SE / length
        # SP = SP / length
        # PC = PC / length
        # F1 = F1 / length
        # JS = JS / length
        # DC = DC / length
        # epoch_loss = epoch_loss / length
        # auc_roc = auc_roc / length
        # CR = CR / length
        # # unet_score = JS + DC
        # unet_score = acc
        #
        # print(
        #     '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, AUROC: %.4f, Pearson: %.4f' % (
        #         epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR, ))
        #
        # # idx order of accs in descending order
        # idx_order_acc = np.argsort(accs)[::-1]
        # # idx order of SEs in descending order
        # idx_order_se = np.argsort(SEs)[::-1]
        # # idx order of PCs in descending order
        # idx_order_PC = np.argsort(PCs)[::-1]
        # # idx order of F1s in descending order
        # idx_order_f1 = np.argsort(F1s)[::-1]
        # # idx order of auc_rocs in descending order
        # idx_order_auc_roc = np.argsort(auc_rocs)[::-1]
        # # idx order of cross
        # idx_order_CR = np.argsort(CRs)[::-1]
        #
        # # str description of the idx order of accs in descending order
        # str_idx_order = f'acc: {idx_order_acc} \n  se: {idx_order_se} \n  sp: {idx_order_PC} \n  f1: {idx_order_f1} \n  auc_roc: {idx_order_auc_roc} \n Pearson: {idx_order_CR}'
        # print(str_idx_order)
        #
        # with open(os.path.join(self.test_result_path, 'metric_result_record.txt'), 'w') as f:
        #     f.write(
        #         '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, auroc: %.4f, Pearson: %.4f' % (
        #             epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR))
        #     # write the best threshold distribution
        #     f.write('\n')
        #     f.write('best threshold distribution: ')
        #     f.write(str(best_threshold_distribution))
        #     f.write('\n')
        #     f.write('best threshold idx: ')
        #     f.write(str_idx_order)
        #     f.write('\n')
        #     f.write('all SEs: ')
        #     f.write(str(SEs))
        #     f.write('\n all Pearsons: ')
        #     f.write(str(CRs))
        #     f.write('\n all PCs: ')
        #     f.write(str(PCs))
        #     f.write('\n all F1s: ')
        #     f.write(str(F1s))
        #     # if 'distance' in two_nanoholes_correlation_result_dict and two_nanoholes_correlation_result_dict['distance']:
        #     #     f.write('\n all Distances: ')
        #     #     f.write(str(two_nanoholes_correlation_result_dict['distance']))
        #
        # np.save(os.path.join(self.test_result_path,
        #                      f'correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
        #         correlation_result_dict)
        # if not os.path.exists(os.path.join(self.result_path, 'correlation_all_nanoholes_results')):
        #     os.makedirs(os.path.join(self.result_path, 'correlation_all_nanoholes_results'))
        # np.save(os.path.join(self.result_path, 'correlation_all_nanoholes_results',
        #                      f'correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
        #         correlation_result_dict)
        #
        # # np.save(os.path.join(self.test_result_path,
        # #                      f'two_nanoholes_correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
        # #         two_nanoholes_correlation_result_dict)
        # # if not os.path.exists(
        # #         os.path.join(self.result_path, 'correlation_distance_2_nanoholes_results')):
        # #     os.makedirs(os.path.join(self.result_path, 'correlation_distance_2_nanoholes_results'))
        # # np.save(os.path.join(self.result_path, 'correlation_distance_2_nanoholes_results',
        # #                      f'two_nanoholes_correlation_result_dict_{self.config.selected_test_fold[0]}.npy'),
        # #         two_nanoholes_correlation_result_dict)

        print(f'Pearson Correlation: {CRs}')
        print(f'Accuracy: {accs}')

