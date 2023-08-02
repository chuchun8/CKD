import torch
import torch.nn.functional as F
import numpy as np
from utils import evaluation


def get_bucket_scores(y_score):
    """
    Organizes real-valued posterior probabilities into buckets.
    For example, if we have 10 buckets, the probabilities 0.0, 0.1,
    0.2 are placed into buckets 0 (0.0 <= p < 0.1), 1 (0.1 <= p < 0.2),
    and 2 (0.2 <= p < 0.3), respectively.
    """
    bucket_values = [[] for _ in range(10)]
    bucket_indices = [[] for _ in range(10)]
    for i, score in enumerate(y_score):
        for j in range(10):
            if score < float((j + 1) / 10):
                break
        bucket_values[j].append(score)
        bucket_indices[j].append(i)
    return bucket_values, bucket_indices


def get_bucket_confidence(bucket_values):
    """
    Computes average confidence for each bucket. If a bucket does
    not have predictions, returns -1.
    """
    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_values
    ]


def get_bucket_f1(bucket_indices, y_true, y_pred, dataset):
    """
    Computes f1 for each bucket. If a bucket does
    not have predictions, returns -1.
    """  
    bucket_f1 = [[] for _ in range(10)]
    for i in range(len(bucket_indices)):
        if len(bucket_indices[i]) > 0:
            f1 = evaluation.compute_f1(y_pred[bucket_indices[i]],y_true[bucket_indices[i]],dataset)
            bucket_f1[i].append(f1)
    
    return [
        bucket[0]
        if len(bucket) > 0 else -1.
        for bucket in bucket_f1
    ]


def calculate_error(n_samples, bucket_values, bucket_confidence, bucket_f1):
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k (b_k / n) |acc(k) - conf(k)|
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Total Calibration Error (TCE): \sum_k |acc(k) - conf(k)|
    """
    assert len(bucket_values) == len(bucket_confidence) == len(bucket_f1)
    assert sum(map(len, bucket_values)) == n_samples

    expected_error = 0.
    for (bucket, f1, confidence) in zip(bucket_values, bucket_f1, bucket_confidence):
        if len(bucket) > 0:
            delta = abs(f1 - confidence)
            expected_error += (len(bucket) / n_samples) * delta
#     for (f1, confidence) in zip(bucket_f1, bucket_confidence):
#         print("f1,conf: ", f1, confidence)
            
    return expected_error * 100.


def get_expected_error(y_logits, y_true, dataset, temp=1):
    
    probs = [F.softmax(elem / temp, dim=0) for elem in y_logits]
#     print("teacher logits: ",teacher_val_logits[0], probs[0])
    confs = [prob.max().item() for prob in probs]
    bucket_values, bucket_indices = get_bucket_scores(confs) # split confs in 10 intervals
    bucket_confidence = get_bucket_confidence(bucket_values)
    bucket_f1 = get_bucket_f1(bucket_indices, y_true, y_logits, dataset)
    expected_error = calculate_error(len(y_logits), bucket_values, bucket_confidence, bucket_f1)
            
    return expected_error


def get_best_temp(teacher_val_outputs, y_val, dataset):
    
    best_temp = 1
    best_error = float('inf')
    best_bucket_conf, best_bucket_f1 = [], []
    for temp_ind in range(1, 501):
        temp = temp_ind*0.01
        expected_error = get_expected_error(teacher_val_outputs, y_val, dataset, temp)
        if expected_error < best_error:
            best_error = expected_error
            best_temp = temp
    if len(y_val) > 100:
        print("Best temp: ", best_temp)
        print("Best calibration error: ", best_error)

    return best_temp
