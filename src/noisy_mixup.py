import numpy as np
import torch
import src.p_corruption as p_corruption
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            var = torch.var(x)**0.5
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.empty(x.shape, dtype=torch.float16, device=device).normal_()
            #torch.clamp(add_noise, min=-(2*var), max=(2*var), out=add_noise) # clamp
            sparse = torch.empty(x.shape, dtype=torch.float16, device=device).uniform_()
            add_noise[sparse<sparse_level] = 0
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.empty(x.shape, dtype=torch.float16, device=device).uniform_()-1) + 1 
            sparse = torch.empty(x.shape, dtype=torch.float16, device=device).uniform_()
            mult_noise[sparse<sparse_level] = 1.0

            
    return mult_noise * x + add_noise      

def do_noisy_mixup(x, y, jsd=0, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0, p_norm=False):
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    
    if jsd==0:
        index = torch.randperm(x.size()[0]).to(device)
        x = lam * x + (1 - lam) * x[index]
        if p_norm == False:
            x = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
        else:
            x = p_corruption.apply_lp_corruption(x, 
                        minibatchsize=8, 
                        combine_train_corruptions=True, 
                        corruptions=p_corruption.train_corruptions, 
                        concurrent_combinations=1, 
                        noise_patch_scale=[list(p_corruption.noise_patch_scale.values())[0], list(p_corruption.noise_patch_scale.values())[1]],
                        random_noise_dist=p_corruption.random_noise_dist,
                        factor=1)
    else:
        kk = 0
        q = int(x.shape[0]/3)
        index = torch.randperm(q).to(device)
    
        for i in range(1,4):
            x[kk:kk+q] = lam * x[kk:kk+q] + (1 - lam) * x[kk:kk+q][index]
            if p_norm == False:
                x[kk:kk+q] = _noise(x[kk:kk+q], add_noise_level=add_noise_level*i, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
            else:
                x[kk:kk+q] = p_corruption.apply_lp_corruption(x[kk:kk+q], 
                        minibatchsize=8, 
                        combine_train_corruptions=True, 
                        corruptions=p_corruption.train_corruptions, 
                        concurrent_combinations=1, 
                        noise_patch_scale=[list(p_corruption.noise_patch_scale.values())[0], list(p_corruption.noise_patch_scale.values())[1]],
                        random_noise_dist=p_corruption.random_noise_dist,
                        factor = i)
            kk += q
     
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
