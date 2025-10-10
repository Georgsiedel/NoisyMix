import os
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    for run in [0,1,2]:

        os.system(f"python cifar_tin.py --dataset=cifar10 --seed={run}")
        os.system(f"python cifar_tin.py --dataset=cifar100 --seed={run}")
        
    # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption

    print('Beginning metric evaluation')
    os.system(f"python evaluate_robustness_orig.py --dataset=cifar10 --dir=../data/NoisyMix/cifar10_models/")
    os.system(f"python evaluate_robustness_orig.py --dataset=cifar100 --dir=../data/NoisyMix/cifar100_models/")
    
            
