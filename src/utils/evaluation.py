import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Evaluation
def compute_f1(preds, y, dataset):
    
    rounded_preds = F.softmax(preds, dim=1)
    _, indices = torch.max(rounded_preds, dim=1)
    
    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    
    if dataset != 'argmin':
        result = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0.0)
        f1_average = result[2]
    else:
        result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0.0)
        f1_average = (result[2][0]+result[2][1])/2 # macro-averaged f1
        
    return f1_average