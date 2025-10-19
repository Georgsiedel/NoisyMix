import os
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    os.system(f"python cifar_tin.py --dataset=tin --epochs=300 --seed={0}")

    # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
    print('Beginning metric evaluation')
    os.system(f"python evaluate_robustness.py --dataset=cifar10 --dir=../trained_models/NoisyMix/cifar10_models/")
    os.system(f"python evaluate_robustness.py --dataset=cifar100 --dir=../trained_models/NoisyMix/cifar100_models/")

    os.system(f"python cifar_tin.py --dataset=tin --epochs=300 --seed={2}")
    os.system(f"python cifar_tin.py --dataset=tin --seed={1}")
    os.system(f"python cifar_tin.py --dataset=tin --seed={2}")
    
    os.system(f"python evaluate_robustness.py --dataset=tin --dir=../trained_models/NoisyMix/tin_models/")

    os.system(f"python cifar_tin.py --dataset=tin --epochs=300 --seed={1}")
