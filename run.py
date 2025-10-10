import os
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    for run in [0]:
        os.system(f"python cifar_tin.py --dataset=cifar100 --seed={run} --resume=1")
    
    for run in [1,2]:

        os.system(f"python cifar_tin.py --dataset=cifar10 --seed={run}")
        os.system(f"python cifar_tin.py --dataset=cifar100 --seed={run}")
    
    # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
    print('Beginning metric evaluation')
    os.system(f"python evaluate_robustness.py --dataset=cifar10 --dir=../trained_models/NoisyMix/cifar10_models/")
    os.system(f"python evaluate_robustness.py --dataset=cifar100 --dir=../trained_models/NoisyMix/cifar100_models/")
    
    for run in [0,1,2]:
        os.system(f"python cifar_tin.py --dataset=tin --epochs=300 --seed={run}")
        os.system(f"python cifar_tin.py --dataset=tin --seed={run}")
    
    os.system(f"python evaluate_robustness.py --dataset=tin --dir=../trained_models/NoisyMix/tin_models/")
