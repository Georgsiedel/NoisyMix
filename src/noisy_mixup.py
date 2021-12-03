import numpy as np
import torch

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparsity_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
            #torch.clamp(add_noise, min=-(2*var), max=(2*var), out=add_noise) # clamp
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.cuda.FloatTensor(x.shape).uniform_()-1) + 1 
    return mult_noise * x + add_noise      

def do_noisy_mixup(x, y, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    #index = torch.randperm(x.size()[0]).cuda()
    
    kk = 0
    q = np.int(x.shape[0]/3)
    index = torch.randperm(q).cuda()

    #print(x.shape)
    for i in range(3):
        x[kk:kk+q] = lam * x[kk:kk+q] + (1 - lam) * x[kk:kk+q][index]
        kk += q
    
    y_a, y_b = y, y[index]
    x = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
