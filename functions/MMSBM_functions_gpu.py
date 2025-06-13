import torch
import numpy as np
from functools import lru_cache

# Cache for tensor conversions to avoid repeated conversions
@lru_cache(maxsize=128)
def to_tensor_cached(x, device='cuda'):
    """Cached version of tensor conversion for immutable inputs."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def to_tensor(x, device='cuda', cache=True):
    """
    Convert numpy array to torch tensor and move to specified device.
    Uses caching for immutable inputs to avoid repeated conversions.
    """
    if cache and isinstance(x, (np.ndarray, torch.Tensor)):
        return to_tensor_cached(x, device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def to_numpy(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

class TensorCache:
    """Class to manage tensor caching for mutable objects."""
    def __init__(self, device='cuda'):
        self.device = device
        self.cache = {}
    
    def get_tensor(self, key, data):
        """Get tensor from cache or create new one."""
        if key not in self.cache:
            self.cache[key] = to_tensor(data, self.device, cache=False)
        return self.cache[key]
    
    def update_tensor(self, key, data):
        """Update tensor in cache."""
        self.cache[key] = to_tensor(data, self.device, cache=False)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()

# Create a global cache instance
tensor_cache = TensorCache()

def omega_comp_arrays(Na, Nb, p_kl, theta, eta, K, L, links_array, links_ratings, device='cuda'):
    """
    GPU-accelerated version of omega_comp_arrays using PyTorch.
    """
    # Use cached tensors where possible
    p_kl = tensor_cache.get_tensor('p_kl', p_kl)
    theta = tensor_cache.get_tensor('theta', theta)
    eta = tensor_cache.get_tensor('eta', eta)
    links_array = tensor_cache.get_tensor('links_array', links_array)
    links_ratings = tensor_cache.get_tensor('links_ratings', links_ratings)
    
    # Initialize omega tensor
    omega = torch.empty((Na, Nb, K, L), device=device, dtype=p_kl.dtype)
    
    # Process links in batches for better GPU utilization
    batch_size = 1024
    n_links = len(links_ratings)
    
    for i in range(0, n_links, batch_size):
        batch_end = min(i + batch_size, n_links)
        batch_links = links_array[i:batch_end]
        batch_ratings = links_ratings[i:batch_end]
        
        # Get indices for the batch
        i_indices = batch_links[:, 0]
        j_indices = batch_links[:, 1]
        
        # Compute omega for the batch
        for idx, (i, j, rating) in enumerate(zip(i_indices, j_indices, batch_ratings)):
            omega[i, j] = p_kl[:, :, rating] * torch.outer(theta[i], eta[j])
            omega[i, j] /= (omega[i, j].sum() + 1e-16)
    
    return to_numpy(omega)

def omega_comp_arrays_exclusive(q_ka, N_att, theta, N_nodes, K, metas_links_arrays_nodes, device='cuda'):
    """
    GPU-accelerated version of omega_comp_arrays_exclusive using PyTorch.
    """
    # Use cached tensors where possible
    q_ka = tensor_cache.get_tensor('q_ka', q_ka)
    theta = tensor_cache.get_tensor('theta', theta)
    metas_links_arrays_nodes = tensor_cache.get_tensor('metas_links', metas_links_arrays_nodes)
    
    # Initialize omega tensor
    omega = torch.empty((N_nodes, N_att, K), device=device, dtype=q_ka.dtype)
    
    # Process links in batches
    batch_size = 1024
    n_links = len(metas_links_arrays_nodes)
    
    for i in range(0, n_links, batch_size):
        batch_end = min(i + batch_size, n_links)
        batch_links = metas_links_arrays_nodes[i:batch_end]
        
        # Get indices for the batch
        i_indices = batch_links[:, 0]
        a_indices = batch_links[:, 1]
        
        # Compute omega for the batch
        for idx, (i, a) in enumerate(zip(i_indices, a_indices)):
            omega[i, a] = theta[i] * q_ka[:, a]
            omega[i, a] /= (omega[i, a].sum() + 1e-16)
    
    return to_numpy(omega)

def theta_comp_arrays_multilayer(BiNet, layer="a", device='cuda'):
    """
    GPU-accelerated version of theta_comp_arrays_multilayer using PyTorch.
    """
    if layer == "a":
        na = BiNet.nodes_a
        observed = tensor_cache.get_tensor('observed_a', BiNet.observed_nodes_a)
        non_observed = tensor_cache.get_tensor('non_observed_a', BiNet.non_observed_nodes_a)
    elif layer == "b":
        na = BiNet.nodes_b
        observed = tensor_cache.get_tensor('observed_b', BiNet.observed_nodes_b)
        non_observed = tensor_cache.get_tensor('non_observed_b', BiNet.non_observed_nodes_b)
    else:
        raise TypeError("Layer must be 'a' or 'b'")
    
    # Convert omega to tensor
    omega = tensor_cache.get_tensor('omega', BiNet.omega)
    
    # Initialize new_theta
    new_theta = torch.empty((len(na), na.K), device=device, dtype=omega.dtype)
    
    # Compute theta for observed nodes
    if layer == "a":
        new_theta[observed] = torch.sum(omega[observed.unsqueeze(1), BiNet.observed_nodes_b], dim=(1, 2))
    else:
        new_theta[observed] = torch.sum(omega[BiNet.observed_nodes_a, observed.unsqueeze(1)], dim=(0, 2))
    
    # Add contributions from exclusive metadata
    for meta in na.meta_exclusives.values():
        meta_omega = tensor_cache.get_tensor(f'meta_omega_{meta.meta_name}', meta.omega)
        new_theta += torch.sum(meta_omega, dim=1) * meta.lambda_val
    
    # Add contributions from inclusive metadata
    for meta in na.meta_inclusives.values():
        meta_omega = tensor_cache.get_tensor(f'meta_omega_{meta.meta_name}', meta.omega)
        new_theta += torch.sum(meta_omega, dim=(1, 2)) * meta.lambda_val
    
    # Normalize
    new_theta /= tensor_cache.get_tensor('denominators', na.denominators)
    
    # Handle cold starts
    if not na._has_metas and len(non_observed) != 0:
        means = torch.mean(new_theta[observed], dim=0)
        new_theta[non_observed] = means
    
    return to_numpy(new_theta)

def p_kl_comp_arrays(Ka, Kb, N_labels, links, omega, mask_list, device='cuda'):
    """
    GPU-accelerated version of p_kl_comp_arrays using PyTorch.
    """
    # Convert inputs to tensors
    omega = to_tensor(omega, device)
    links = to_tensor(links, device)
    mask_list = [to_tensor(mask, device) for mask in mask_list]
    
    # Initialize p_kl tensor
    p_kl = torch.empty((Ka, Kb, N_labels), device=device)
    
    # Compute sum_list
    sum_list = omega[links[:, 0], links[:, 1]]
    
    # Compute p_kl for each label
    for l, mask in enumerate(mask_list):
        p_kl[:, :, l] = torch.sum(sum_list[mask], dim=0)
    
    # Normalize
    suma = torch.sum(p_kl, dim=2)
    p_kl /= (suma.unsqueeze(2) + 1e-16)
    
    return to_numpy(p_kl)

def q_ka_comp_arrays(K, N_att, omega, links, masks_att_list, device='cuda'):
    """
    GPU-accelerated version of q_ka_comp_arrays using PyTorch.
    """
    # Convert inputs to tensors
    omega = to_tensor(omega, device)
    links = to_tensor(links, device)
    masks_att_list = [to_tensor(mask, device) for mask in masks_att_list]
    
    # Initialize qka2 tensor
    qka2 = torch.zeros((K, N_att), device=device)
    
    # Compute unfolded_q
    unfolded_q = omega[links[:, 0], links[:, 1]]
    
    # Compute qka2 for each attribute
    for att, mask in enumerate(masks_att_list):
        qka2[:, att] = torch.sum(unfolded_q[mask], dim=0)
    
    # Normalize
    suma = torch.sum(qka2, dim=1)
    qka2 /= suma.unsqueeze(1)
    
    return to_numpy(qka2)

def log_like_comp(theta, eta, pkl, links, labels, device='cuda'):
    """
    GPU-accelerated version of log_like_comp using PyTorch.
    """
    # Convert inputs to tensors
    theta = to_tensor(theta, device)
    eta = to_tensor(eta, device)
    pkl = to_tensor(pkl, device)
    links = to_tensor(links, device)
    labels = to_tensor(labels, device)
    
    # Compute log likelihood
    log_like = 0
    for i, j, label in zip(links[:, 0], links[:, 1], labels):
        p = torch.sum(theta[i].unsqueeze(1) * eta[j].unsqueeze(0) * pkl[:, :, label])
        log_like += torch.log(p + 1e-16)
    
    return to_numpy(log_like)

def total_p_comp_test(theta, eta, pkl, links, device='cuda'):
    """
    GPU-accelerated version of total_p_comp_test using PyTorch.
    """
    # Convert inputs to tensors
    theta = to_tensor(theta, device)
    eta = to_tensor(eta, device)
    pkl = to_tensor(pkl, device)
    links = to_tensor(links, device)
    
    # Compute total probabilities
    total_p = torch.zeros(len(links), device=device)
    for idx, (i, j) in enumerate(zip(links[:, 0], links[:, 1])):
        total_p[idx] = torch.sum(theta[i].unsqueeze(1) * eta[j].unsqueeze(0) * pkl)
    
    return to_numpy(total_p) 