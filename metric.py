import numpy as np
from numba import njit
from sklearn.covariance import LedoitWolf
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import torch
import pandas as pd
import itertools

def random_projection( method = 'gaussian', n_components = 'auto'):
    if method == 'gaussian': 
        rp = GaussianRandomProjection(n_components)
    elif method =='sparse':
        rp = SparseRandomProjection(n_components)
    return rp


def log_maximum_evidence(features: np.ndarray, targets: np.ndarray, regression=False, return_weights=False):
    r"""
    Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models
    for Transfer Learning (ICML 2021) <https://arxiv.org/pdf/2102.11005.pdf>`_.
    
    Args:
        features (np.ndarray): feature matrix from pre-trained model.
        targets (np.ndarray): targets labels/values.
        regression (bool, optional): whether to apply in regression setting. (Default: False)
        return_weights (bool, optional): whether to return bayesian weight. (Default: False)
    Shape:
        - features: (N, F) with element in [0, :math:`C_t`) and feature dimension F, where :math:`C_t` denotes the number of target class
        - targets: (N, ) or (N, C), with C regression-labels.
        - weights: (F, :math:`C_t`).
        - score: scalar.
    """
    f = features.astype(np.float64)
    y = targets
    if regression:
        y = targets.astype(np.float64)

    fh = f
    f = f.transpose()
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

    evidences = []
    weights = []
    if regression:
        C = y.shape[1]
        for i in range(C):
            y_ = y[:, i]
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)
    else:
        C = int(y.max() + 1)
        for i in range(C):
            y_ = (y == i).astype(np.float64)
            evidence, weight = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            weights.append(weight)

    score = np.mean(evidences)
    weights = np.vstack(weights)

    if return_weights:
        return score, weights
    else:
        return score


@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ y_))

    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / alpha_de
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / beta_de
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam

    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * beta_de \
               - alpha / 2.0 * alpha_de \
               - N / 2.0 * np.log(2 * np.pi)

    return evidence / N, m



def h_score(features: np.ndarray, labels: np.ndarray):
    r"""
    H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019) 
    <http://yangli-feasibility.com/home/media/icip-19.pdf>`_.
    
    The H-Score :math:`\mathcal{H}` can be described as:
    .. math::
        \mathcal{H}=\operatorname{tr}\left(\operatorname{cov}(f)^{-1} \operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector

    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.
    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features
    y = labels

    covf = np.cov(f, rowvar = False)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar = False)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score

def regularized_h_score(features: np.ndarray, labels: np.ndarray):
    f = features.astype('float64')
    f = f - np.mean(f, axis= 0, keepdims= True) #Center the features for correct Ledoit-Wolf Estimation
    y = labels

    C = int(y.max() + 1)
    g = np.zeros_like(f)

    cov = LedoitWolf(assume_centered= False).fit(f)
    alpha = cov.shrinkage_
    covf_alpha = cov.covariance_

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar = False)
    score = np.trace(np.dot(np.linalg.pinv(covf_alpha, rcond=1e-15), (1-alpha) * covg))

    return score


def coding_rate(features : np.ndarray, eps = 1e-4):
    f = features
    n, d = f.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * f.transpose() @ f))
    return 0.5 * rate

def transrate(features : np.ndarray, labels : np.ndarray, eps = 1e-4):
    f = features
    y = labels
    f = f - np.mean(f, axis = 0, keepdims = True)
    Rf = coding_rate(f, eps)
    Rfy = 0.0
    C = int(y.max() + 1)
    for i in range(C):
        Rfy += coding_rate(f[(y==i).flatten()], eps)
    return Rf - Rfy / C


def compute_metric_per_layer(dataset = 'BCCD', metric = 'label_LogME', rp = None):
    feats_metric, bbox_metric = [], []
    annots= torch.load(f"/data.nfs/AUTO_TL_OD/extracted_feats/{dataset}/labels/all_labels.pt")

    labels = annots[:,-1].cpu().int().numpy() - 1 # Extract only labels, convert to int32 and remove background class (0 class)
    if dataset == 'Open_Images':
        labels[labels==12] = 10
        labels[labels==11] = 2  #Label 10 and 2 are  not there so we replace label 10 by label 12 #dirty patch up to modify !
    for i in range(1,6):
        feats = torch.load(f"/data.nfs/AUTO_TL_OD/extracted_feats/{dataset}/layer_{i}/all_features.pt").cpu().numpy()
        feats_bbox = torch.load(f"/data.nfs/AUTO_TL_OD/extracted_feats/{dataset}/layer_{i}/all_features_bbox.pt").cpu().numpy()

        #Random projection
        if rp is not None : 
            feats = rp.fit_transform(feats)
            feats_bbox = rp.fit_transform(feats)

        if metric == 'label_LogME':
            logme = log_maximum_evidence(feats, labels)
            feats_metric.append(logme)
            logme = log_maximum_evidence(feats_bbox, labels)
            bbox_metric.append(logme)

        if metric == 'xy_LogME':
            logme_xy = log_maximum_evidence(feats,annots[:,0:4].int().numpy(), regression = True)
            feats_metric.append(logme_xy)
            logme = log_maximum_evidence(feats_bbox,annots[:,0:4].int().numpy(), regression = True)
            bbox_metric.append(logme_xy)

        elif metric == 'hscore':
            hscore =  h_score(feats, labels)
            feats_metric.append(hscore)
            hscore =  h_score(feats_bbox, labels)
            bbox_metric.append(hscore)

        elif metric == 'regularized_hscore':
            hscore =  regularized_h_score(feats, labels)
            feats_metric.append(hscore)
            hscore =  regularized_h_score(feats_bbox, labels)
            bbox_metric.append(hscore)
        
        elif metric == 'transrate':
            tr =  transrate(feats, labels)
            feats_metric.append(tr)
            tr =  transrate(feats_bbox, labels)
            bbox_metric.append(tr)
        

    return feats_metric, bbox_metric



def compute_metric_per_dataset(metric = 'LogME', rp = None):
    metric_bccd, bbox_metric_bccd = compute_metric_per_layer(dataset = 'BCCD', metric = metric, rp = rp)
    metric_chess, bbox_metric_chess = compute_metric_per_layer(dataset = 'CHESS', metric = metric, rp = rp)
    metric_gw, bbox_metric_gw = compute_metric_per_layer(dataset = 'Global_Wheat', metric = metric, rp = rp)
    metric_voc, bbox_metric_voc = compute_metric_per_layer(dataset = 'VOC', metric = metric, rp = rp)
    metric_oi, bbox_metric_oi = compute_metric_per_layer(dataset = 'Open_Images', metric = metric, rp = rp)

    metric_df = np.vstack((metric_bccd, metric_chess, metric_gw, metric_voc, metric_oi))
    metric_df = pd.DataFrame(metric_df, index = ['BCCD', 'CHESS', 'Global_Wheat', 'VOC', 'Open_Images'])

    bbox_metric_df = np.vstack((bbox_metric_bccd, bbox_metric_chess, bbox_metric_gw, bbox_metric_voc, bbox_metric_oi))
    bbox_metric_df = pd.DataFrame(bbox_metric_df, index = ['BCCD', 'CHESS', 'Global_Wheat', 'VOC', 'Open_Images'])
    
    return metric_df, bbox_metric_df


def compute_metric_synth(metric = 'label_LogME', rp = None, layer = 'layer_5', device = 'cpu'): 

    datasets = ['MNIST', 'KMNIST', 'EMNIST', 'FASHION_MNIST', 'USPS']
    df_metric = pd.DataFrame(np.zeros((5,5)), index = datasets, columns = datasets)
    bbox_metric_df = pd.DataFrame(np.zeros((5,5)), index = datasets, columns = datasets)

    for dataset_source, dataset_target in itertools.permutations(datasets, 2):
        annots= torch.load(f"/data.nfs/AUTO_TL_OD/extracted_feats/{dataset_target}/from_{dataset_source}/labels/all_labels.pt").to(device)

        labels = annots[:,-1].to(device).int().numpy() - 1

        feats = torch.load(f"/data.nfs/AUTO_TL_OD/extracted_feats/{dataset_target}/from_{dataset_source}/{layer}/all_features.pt").to(device).numpy()
        feats_bbox = torch.load(f"/data.nfs/AUTO_TL_OD/extracted_feats/{dataset_target}/from_{dataset_source}/{layer}/all_features_bbox.pt").to(device).numpy()

        if rp is not None:
            feats = rp.fit_transform(feats)
            feats_bbox = rp.fit_transform(feats)

        elif metric == 'label_LogME':
            logme = log_maximum_evidence(feats, labels)
            df_metric.loc[dataset_source, dataset_target] = logme
            logme = log_maximum_evidence(feats_bbox, labels)
            bbox_metric_df.loc[dataset_source, dataset_target] = logme

        elif metric == 'xy_LogME':
            logme = log_maximum_evidence(feats,annots[:,0:4].int().numpy(), regression = True)
            df_metric.loc[dataset_source, dataset_target] = logme
            logme = log_maximum_evidence(feats_bbox,annots[:,0:4].int().numpy(), regression = True)
            bbox_metric_df.loc[dataset_source, dataset_target] = logme

        elif metric == 'hscore':
            hscore =  h_score(feats, labels)
            df_metric.loc[dataset_source, dataset_target] = hscore
            hscore =  h_score(feats_bbox, labels)
            bbox_metric_df.loc[dataset_source, dataset_target] = hscore

        elif metric == 'regularized_hscore':
            hscore =  regularized_h_score(feats, labels)
            df_metric.loc[dataset_source, dataset_target] = hscore
            hscore =  regularized_h_score(feats_bbox, labels)
            bbox_metric_df.loc[dataset_source, dataset_target] = hscore

        elif metric == 'transrate':
            tr =  transrate(feats, labels)
            df_metric.loc[dataset_source, dataset_target] = tr
            tr =  transrate(feats_bbox, labels)
            bbox_metric_df.loc[dataset_source, dataset_target] = tr

    return df_metric, bbox_metric_df


def compute_metric_real(metric = 'label_LogME', rp = None, dataset_paths = None,  data_dir = "/data.nfs/AUTO_TL_OD/extracted_feats/", layer  = 'layer_5'): 

    metrics = []
    bbox_metrics = []

    for dataset_path in dataset_paths:

        annots= torch.load(data_dir + f"{dataset_path}/labels/all_labels.pt").cpu()
        labels = annots[:,-1].cpu().int().numpy() - 1

        feats = torch.load(data_dir + f"{dataset_path}/{layer}/all_features.pt").cpu().numpy()
        feats_bbox = torch.load(data_dir + f"{dataset_path}/{layer}/all_features_bbox.pt").cpu().numpy()

        if dataset_path == 'Open_Images':
            labels[labels==12] = 10
            labels[labels==11] = 2  #Label 10 and 2 are  not there so we replace label 10 by label 12 #dirty patch up to modify !

        if rp is not None:
            feats = rp.fit_transform(feats)
            feats_bbox = rp.fit_transform(feats)

        if metric == 'label_LogME':
            logme = log_maximum_evidence(feats, labels)
            metrics.append(logme)
            logme = log_maximum_evidence(feats_bbox, labels)
            bbox_metrics.append(logme)

        elif metric == 'xy_LogME':
            logme = log_maximum_evidence(feats,annots[:,0:4].int().numpy(), regression = True)
            metrics.append(logme)
            logme = log_maximum_evidence(feats_bbox,annots[:,0:4].int().numpy(), regression = True)
            bbox_metrics.append(logme)

        elif metric == 'hscore':
            hscore =  h_score(feats, labels)
            metrics.append(hscore)
            hscore =  h_score(feats_bbox, labels)
            bbox_metrics.append(hscore)

        elif metric == 'regularized_hscore':
            hscore =  regularized_h_score(feats, labels)
            metrics.append(hscore)
            hscore =  regularized_h_score(feats_bbox, labels)
            bbox_metrics.append(hscore)

        elif metric == 'transrate':
            tr =  transrate(feats, labels)
            metrics.append(tr)
            tr =  transrate(feats_bbox, labels)
            bbox_metrics.append(tr)


    return metrics, bbox_metrics