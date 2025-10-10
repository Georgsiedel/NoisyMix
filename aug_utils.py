import augmentations
import numpy as np

import torch


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, jsd=False, image_size=32, args=None):
    self.dataset = dataset
    self.preprocess = preprocess
    self.jsd = jsd
    self.args = args

    augmenter = augmentations.Augment(image_size=image_size)
    
    self.aug_list = augmenter.augmentations
    if args.all_ops:
      self.aug_list = augmenter.augmentations_all

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.jsd==0:
      return self.aug(x, self.preprocess, self.args), y
    else:
      im_tuple = (self.preprocess(x), self.aug(x, self.preprocess, self.args), self.aug(x, self.preprocess, self.args))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)
  
  def aug(self, image, preprocess, args):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
      image_aug = image.copy()
      depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
          1, 4)
      for _ in range(depth):
        op = np.random.choice(self.aug_list)
        image_aug = op(image_aug, args.aug_severity)
      # Preprocessing commutes since all coefficients are convex
      mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed



