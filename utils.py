from monai.transforms import Transform
import torch
import numpy as np


class CustomConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ["torch", "numpy"]

    def __call__(self, img):
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 2) | (img == 3), (img == 2) | (img == 3) | (img == 1), img == 3]
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        # label 4 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)
    
class AverageMeter(object):
    def __init__(self):        
        self.reset()
    def reset(self):        
        self.val = 0        
        self.avg = 0        
        self.sum = 0        
        self.count = 0
    def update(self, val, n=1):        
        self.val = val       
        self.sum += val * n        
        self.count += n        
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)