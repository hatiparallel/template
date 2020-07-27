import numpy as np
import torch
import torch.nn as nn

# an example with simple cce
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cce = nn.CrossEntropyLoss()
        return

    def forward(self, pred : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        Args
            pred : prediction
            target : target with the indexes of class
        Returns
            loss
        """
        loss = self.cce(pred, target)
        
        return loss