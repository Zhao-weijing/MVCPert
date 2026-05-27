from collections import OrderedDict
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sklearn
import random
import h5py
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _normalize_smiles_values(smiles_values):
    return np.asarray([str(s).strip() for s in np.asarray(smiles_values).astype(str)])


def _unique_preserve_order(smiles_values):
    normalized = _normalize_smiles_values(smiles_values)
    seen = set()
    ordered = []
    for smiles in normalized:
        if smiles not in seen:
            seen.add(smiles)
            ordered.append(smiles)
    return ordered


def build_split_lock_payload(
    dataset_name,
    split_data_type,
    seed,
    train_smiles,
    valid_smiles,
    test_smiles,
    source="runtime",
    extra_meta=None,
):
    train_unique = _unique_preserve_order(train_smiles)
    valid_unique = _unique_preserve_order(valid_smiles)
    test_unique = _unique_preserve_order(test_smiles)
    payload = {
        "dataset_name": str(dataset_name),
        "split_data_type": str(split_data_type),
        "seed": int(seed),
        "source": str(source),
        "train_smiles": train_unique,
        "valid_smiles": valid_unique,
        "test_smiles": test_unique,
        "counts": {
            "train_unique_smiles": int(len(train_unique)),
            "valid_unique_smiles": int(len(valid_unique)),
            "test_unique_smiles": int(len(test_unique)),
        },
    }
    if extra_meta:
        payload["created_with"] = extra_meta
    return payload


def save_split_lock(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_split_lock(path, expected_dataset_name=None, expected_split_data_type=None):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "splits" in payload and isinstance(payload["splits"], dict):
        split_section = payload["splits"]
        payload.setdefault("train_smiles", split_section.get("train_smiles", []))
        payload.setdefault("valid_smiles", split_section.get("valid_smiles", []))
        payload.setdefault("test_smiles", split_section.get("test_smiles", []))

    required_keys = {
        "dataset_name",
        "split_data_type",
        "seed",
        "train_smiles",
        "valid_smiles",
        "test_smiles",
    }
    missing = required_keys.difference(payload.keys())
    if missing:
        raise ValueError(f"split lock 缺少字段: {sorted(missing)}")

    if expected_dataset_name is not None and payload["dataset_name"] != expected_dataset_name:
        raise ValueError(
            f"split lock dataset_name 不匹配: expected={expected_dataset_name}, got={payload['dataset_name']}"
        )
    if expected_split_data_type is not None and payload["split_data_type"] != expected_split_data_type:
        raise ValueError(
            f"split lock split_data_type 不匹配: expected={expected_split_data_type}, got={payload['split_data_type']}"
        )

    for key in ("train_smiles", "valid_smiles", "test_smiles"):
        payload[key] = _unique_preserve_order(payload[key])

    return payload


def split_data_with_lock(data, split_lock_payload):
    if "canonical_smiles" not in data:
        raise ValueError("data 中缺少 canonical_smiles，无法应用 split lock")

    all_smiles = _normalize_smiles_values(data["canonical_smiles"])
    train_set = set(_normalize_smiles_values(split_lock_payload["train_smiles"]))
    valid_set = set(_normalize_smiles_values(split_lock_payload["valid_smiles"]))
    test_set = set(_normalize_smiles_values(split_lock_payload["test_smiles"]))

    train_mask = np.isin(all_smiles, list(train_set))
    valid_mask = np.isin(all_smiles, list(valid_set))
    test_mask = np.isin(all_smiles, list(test_set))

    pair = subsetDict(data, np.where(train_mask)[0])
    pairv = subsetDict(data, np.where(valid_mask)[0])
    pairt = subsetDict(data, np.where(test_mask)[0])

    split_lock_meta = {
        "dataset_name": split_lock_payload["dataset_name"],
        "split_data_type": split_lock_payload["split_data_type"],
        "seed": int(split_lock_payload["seed"]),
        "source": split_lock_payload.get("source", "unknown"),
        "train_unique_smiles": int(len(train_set)),
        "valid_unique_smiles": int(len(valid_set)),
        "test_unique_smiles": int(len(test_set)),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }

    if split_lock_payload.get("counts"):
        split_lock_meta["declared_counts"] = dict(split_lock_payload["counts"])

    return pair, pairv, pairt, split_lock_meta


def setup_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def load_from_HDF(fname):
    """Load data from a HDF5 file to a dictionary."""
    data = dict()
    with h5py.File(fname, 'r') as f:
        for key in f:
            data[key] = np.asarray(f[key])
            if data[key].size == 0:
                continue
            first_item = data[key][0]
            if isinstance(first_item, (np.bytes_, bytes, bytearray)):
                data[key] = np.asarray(
                    [
                        item.decode("utf-8") if isinstance(item, (np.bytes_, bytes, bytearray)) else str(item)
                        for item in data[key]
                    ],
                    dtype=object,
                )
    return data


def compute_same_compound_dose_ordinal_loss(
    pred_delta,
    target_delta,
    mol_ids,
    dose_feature,
    temperature=0.2,
    margin_scale=0.25,
):
    """Batch-local same-compound dose ordinal retrieval loss.

    For each compound appearing with multiple doses in the same batch, we build a
    query-vs-target similarity matrix using cosine similarity between predicted GE
    deltas and target GE deltas. The diagonal is the correct dose match. We then
    combine:
    1. an identity CE loss over same-compound candidate doses;
    2. a rank margin loss that requires farther dose ranks to be less similar.

    Parameters
    ----------
    pred_delta, target_delta : torch.Tensor
        Shape [B, D].
    mol_ids : sequence[str]
        Batch identifiers. Same compound / same canonical SMILES share the same id.
    dose_feature : torch.Tensor
        Shape [B]. Can be normalized log-dose; only relative order is used.
    """
    zero = pred_delta.new_zeros(())
    if pred_delta.ndim != 2 or target_delta.ndim != 2:
        raise ValueError("pred_delta and target_delta must be rank-2 tensors")
    if pred_delta.shape != target_delta.shape:
        raise ValueError("pred_delta and target_delta must have the same shape")
    if pred_delta.shape[0] < 2:
        return {
            "loss": zero,
            "ce_loss": zero,
            "margin_loss": zero,
            "group_count": 0,
            "query_count": 0,
            "batch_top1": 0.0,
        }

    mol_id_list = [str(x) for x in mol_ids]
    group_map = {}
    for idx, mol_id in enumerate(mol_id_list):
        group_map.setdefault(mol_id, []).append(idx)

    ce_terms = []
    margin_terms = []
    top1_hits = 0.0
    query_count = 0
    group_count = 0
    temperature = float(max(temperature, 1e-6))
    margin_scale = float(max(margin_scale, 0.0))

    pred_delta = F.normalize(pred_delta, dim=1, eps=1e-8)
    target_delta = F.normalize(target_delta, dim=1, eps=1e-8)

    for indices in group_map.values():
        if len(indices) < 2:
            continue

        group_count += 1
        query_count += len(indices)
        index_tensor = torch.as_tensor(indices, device=pred_delta.device, dtype=torch.long)
        group_pred = pred_delta.index_select(0, index_tensor)
        group_target = target_delta.index_select(0, index_tensor)
        group_dose = dose_feature.index_select(0, index_tensor)

        similarity = group_pred @ group_target.T
        labels = torch.arange(len(indices), device=pred_delta.device, dtype=torch.long)
        ce_terms.append(F.cross_entropy(similarity / temperature, labels))
        top1_hits += float((similarity.argmax(dim=1) == labels).float().sum().item())

        order = torch.argsort(group_dose, stable=True)
        ranks = torch.empty_like(order, dtype=group_pred.dtype)
        ranks[order] = torch.arange(len(indices), device=pred_delta.device, dtype=group_pred.dtype)
        rank_gap = (ranks[:, None] - ranks[None, :]).abs()
        if len(indices) > 1 and margin_scale > 0.0:
            margin = margin_scale * (rank_gap / float(len(indices) - 1))
            positive = similarity.diagonal().unsqueeze(1)
            violations = F.relu(similarity + margin - positive)
            negative_mask = ~torch.eye(len(indices), device=pred_delta.device, dtype=torch.bool)
            if negative_mask.any():
                margin_terms.append(violations.masked_select(negative_mask).mean())

    if not ce_terms:
        return {
            "loss": zero,
            "ce_loss": zero,
            "margin_loss": zero,
            "group_count": 0,
            "query_count": 0,
            "batch_top1": 0.0,
        }

    ce_loss = torch.stack(ce_terms).mean()
    margin_loss = torch.stack(margin_terms).mean() if margin_terms else zero
    total_loss = ce_loss + margin_loss
    batch_top1 = float(top1_hits / max(query_count, 1))
    return {
        "loss": total_loss,
        "ce_loss": ce_loss.detach(),
        "margin_loss": margin_loss.detach(),
        "group_count": int(group_count),
        "query_count": int(query_count),
        "batch_top1": batch_top1,
    }



def calculate_pcc_per_sample(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    计算每个样本的预测值与真实值之间的Pearson相关系数(PCC)，并返回所有样本的平均PCC。
    
    参数:
        true_labels: 真实值数组，形状为 [样本数, 基因数]
        predicted_labels: 预测结果数组，形状为 [样本数, 基因数]
        
    返回:
        所有样本的平均PCC值
    """
    # 确保输入数组形状一致
    assert true_labels.shape == predicted_labels.shape, "预测结果和真实值的形状必须一致"
    
    pcc_sum = 0.0
    num_samples = true_labels.shape[0]  # 样本数量（行数）
    
    # 逐样本计算PCC
    for sample in range(num_samples):
        # 获取第sample个样本的所有基因（一行数据）
        y_true = true_labels[sample, :]
        y_pred = predicted_labels[sample, :]
        
        # 计算均值（每个样本内所有基因的均值）
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        
        # 计算协方差和方差（样本内基因维度）
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        variance_true = np.mean((y_true - mean_true) **2)
        variance_pred = np.mean((y_pred - mean_pred)** 2)
        
        # 计算PCC（添加小常数避免除零）
        eps = 1e-8
        pcc = covariance / np.sqrt(variance_true * variance_pred + eps)
        pcc_sum += pcc
    
    # 返回所有样本的平均PCC
    return pcc_sum / num_samples

def calculate_rmse_per_sample(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    计算每个样本的预测值与真实值之间的均方根误差(RMSE)，并返回所有样本的平均RMSE。
    
    参数:
        true_labels: 真实值数组，形状为 [样本数, 基因数]
        predicted_labels: 预测结果数组，形状为 [样本数, 基因数]
        
    返回:
        所有样本的平均RMSE值
    """
    # 确保输入数组形状一致
    assert true_labels.shape == predicted_labels.shape, "预测结果和真实值的形状必须一致"
    
    rmse_sum = 0.0
    num_samples = true_labels.shape[0]  # 样本数量（行数）
    
    # 逐样本计算RMSE
    for sample in range(num_samples):
        # 获取第sample个样本的所有基因（一行数据）
        y_true = true_labels[sample, :]
        y_pred = predicted_labels[sample, :]
        
        # 计算均方误差(MSE)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # 计算RMSE并累加
        rmse = np.sqrt(mse)
        rmse_sum += rmse
    
    # 返回所有样本的平均RMSE
    return rmse_sum / num_samples



def save_to_HDF(fname, data):
    """Save data (a dictionary) to a HDF5 file."""
    with h5py.File(fname, 'w') as f:
        for key, item in data.items():
            if isinstance(item[0], str):
                item = item.astype(np.bytes_)
            f[key] = item

def selectFromDict(d, keys):
    '''select the chosen @keys in dict @d'''
    res = dict()
    for k in keys:
        if k not in d.keys():
            raise ValueError("Key " + str(k) + " not found")
        else:
            res[k] = d[k]
    return res

def subsetDict(d, ind):
    '''subset each numpy array in dict @d to indices @ind'''
    res = dict()
    if not isinstance(ind, np.ndarray):
        ind = np.asarray(ind)
    for k in d.keys():
        if isinstance(d[k], np.ndarray):
            res[k] = d[k][ind]
        # res[k] = d[k][ind]
    return res

def concatDictElemetwise(a, b):
    '''concatenate dict @a and @b elementwise'''
    res = dict()
    if not sorted(a.keys()) == sorted(b.keys()):
        raise ValueError("Mismatching key sets")
    for k in a.keys():
        res[k] = np.concatenate((a[k], b[k]), axis=0)
    return res

def concat_dicts(a, b):
    # works for both dict and OrderedDict classes
    assert (len(set(a.keys()) & set(b.keys())) == 0), "Can't unambiguously concatenate."
    return type(a)(list(a.items()) + list(b.items()))

def getSplitsByGroupKFold(groups, n_splits, shuffle, random_state):
    assert (n_splits >= 3)
    kf = GroupKFold(n_splits=n_splits)

    # Convert groups to a simple hashable type if they're not already
    if isinstance(groups[0], (list, dict, np.ndarray)) or hasattr(groups[0], '__iter__') and not isinstance(groups[0], str):
        # If groups are complex objects, convert to strings for hashing
        groups_simple = np.array([str(g) for g in groups])
    else:
        groups_simple = np.array(groups)
    
    if shuffle:
        # Use the simplified groups for unique identification
        unique_groups = np.unique(groups_simple)
        rnd_renames = sklearn.utils.shuffle(np.arange(len(unique_groups)), random_state=random_state)
        
        # Create a mapping from original values to renamed values
        group_to_rename = {g: rnd_renames[i] for i, g in enumerate(unique_groups)}
        
        # Use the mapping directly instead of argwhere
        groups_renamed = np.array([group_to_rename[g] for g in groups_simple])
        kfsplit = kf.split(X=np.zeros(len(groups)), groups=groups_renamed)
    else:
        kfsplit = kf.split(X=np.zeros(len(groups)), groups=groups_simple)

    folds = [list(x[1]) for x in kfsplit]
    folds_nums = list(range(len(folds)))
    
    tr_fold_nums = folds_nums[:-2]
    ind_tr = sum([folds[i] for i in tr_fold_nums], [])
    ind_va = folds[folds_nums[-2]]
    ind_te = folds[folds_nums[-1]]

    return ind_tr, ind_va, ind_te

    if shuffle:
        # randomly rename groups so that the GroupKFold (which sorts by group ids first) splits can be randomized
        unique_groups = np.unique(groups)
        rnd_renames = sklearn.utils.shuffle(np.arange(len(unique_groups)), random_state=random_state)
        groups_renamed = np.array([rnd_renames[np.argwhere(unique_groups == g)[0]] for g in groups])
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups_renamed)
    else:
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups)

    folds = [list(x[1]) for x in kfsplit]
    folds_nums = list(range(len(folds)))

    tr_fold_nums = folds_nums[:-2]
    ind_tr = sum([folds[i] for i in tr_fold_nums], [])
    ind_va = folds[folds_nums[-2]]
    ind_te = folds[folds_nums[-1]]

    return ind_tr, ind_va, ind_te


def split_data(data, n_folds=5, split_type='random_split', rnds=None):
    if split_type == 'random_split':
        ind_tr, ind_va, ind_te = getSplitsByGroupKFold(data['sig'], n_folds, shuffle=True, random_state=rnds)
    elif split_type == 'smiles_split':
        # print(data['smiles'])
        data['canonical_smiles'] = data['canonical_smiles'].astype(str)
        # ​​GroupKFold仍会严格保证同一组的数据不会同时出现在训练集和验证集/测试集中​​。
        ind_tr, ind_va, ind_te = getSplitsByGroupKFold(data['canonical_smiles'], n_folds, shuffle=True, random_state=rnds)
        # ind_tr, ind_va, ind_te = getSplitsByGroupKFold(data['canonical_smiles'], n_folds, shuffle=True, random_state=rnds)
    ppair = subsetDict(data, ind_tr)
    ppairv = subsetDict(data, ind_va)
    ppairt = subsetDict(data, ind_te)

    return ppair, ppairv, ppairt


def split_data_cid(data, train_cell_count='all'):
    cell_ls_pretrain = ['HT29', 'NPC', 'A375', 'PC3', 'ASC', 'MCF7', 'HCC515', 'VCAP', 'HA1E', 'A549']

    df_data = pd.DataFrame(data['cid'], columns=['cid'])
    df_data_cid = df_data.value_counts('cid').reset_index()
    df_data_cid.columns = ['cid', 'count']

    test_cell_ls = df_data_cid[~df_data_cid['cid'].isin(cell_ls_pretrain)].sample(n=7, replace=False, random_state=1234)['cid'].tolist()
    valid_cell_ls = df_data_cid[~df_data_cid['cid'].isin(cell_ls_pretrain + test_cell_ls)].sample(n=7, replace=False, random_state=1234)['cid'].tolist()

    if train_cell_count == 'all':
        train_cell_ls = df_data_cid[~(df_data_cid['cid'].isin(valid_cell_ls + test_cell_ls))]['cid'].tolist()
    else:
        train_cell_ls = df_data_cid[~df_data_cid['cid'].isin(valid_cell_ls + test_cell_ls)].sample(n=int(train_cell_count), replace=False, random_state=1234)['cid'].tolist()

    index_ls = df_data[df_data['cid'].isin(train_cell_ls)].index.tolist()
    ppair = subsetDict(data, index_ls)
    index_ls = df_data[df_data['cid'].isin(valid_cell_ls)].index.tolist()
    ppairv = subsetDict(data, index_ls)
    index_ls = df_data[df_data['cid'].isin(test_cell_ls)].index.tolist()
    ppairt = subsetDict(data, index_ls)
    return ppair, ppairv, ppairt


def precision_10(label_test, label_predict):
    k = 10
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k

def precision_20(label_test, label_predict):
    k = 20
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k

def precision_50(label_test, label_predict):
    k = 50
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k

def precision_100(label_test, label_predict):
    k = 100
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test)
    label_predict = np.argsort(label_predict)
    neg_test_set = label_test[:num_neg]
    pos_test_set = label_test[-num_pos:]
    neg_predict_set = label_predict[:k]
    pos_predict_set = label_predict[-k:]

    neg_test = set(neg_test_set)
    pos_test = set(pos_test_set)
    neg_predict = set(neg_predict_set)
    pos_predict = set(pos_predict_set)

    return len(neg_test.intersection(neg_predict)) / k, len(pos_test.intersection(pos_predict)) / k



def rmse(targets, preds):

    return np.sqrt(mean_squared_error(targets, preds))


def pearson(targets, preds):
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return pearsonr(targets, preds)[0]
    except ValueError:
        print(targets, preds)
        print(np.isnan(targets), np.isnan(preds))
        return float('nan')


def spearman(targets, preds):
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return spearmanr(targets, preds)[0]
    except ValueError:
        return float('nan')


# This file consists of useful functions that are related to cmap
def computecs(qup, qdown, expression):
    '''
    This function takes qup & qdown, which are lists of gene
    names, and  expression, a panda data frame of the expressions
    of genes as input, and output the connectivity score vector
    '''
    r1 = ranklist(expression)
    if qup and qdown:
        esup = computees(qup, r1)
        esdown = computees(qdown, r1)
        w = []
        for i in range(len(esup)):
            if esup[i]*esdown[i] <= 0:
                w.append(esup[i]-esdown[i])
            else:
                w.append(0)
        return pd.DataFrame(w, expression.columns)
    elif qup and qdown==None:
        print('None down')
        esup = computees(qup, r1)
        return pd.DataFrame(esup, expression.columns)
    elif qup == None and qdown:
        print('None up')
        esdown = computees(qdown, r1)
        return pd.DataFrame(esdown, expression.columns)
    else:
        return None

def computees(q, r1):
    '''
    This function takes q, a list of gene names, and r1, a panda data
    frame as the input, and output the enrichment score vector
    '''
    if len(q) == 0:
        ks = 0
    elif len(q) == 1:
        ks = r1.loc[q, :]
        ks.index = [0]
        ks = ks.T

    else:
        n = r1.shape[0]

        sub = r1.loc[q, :]
        J = sub.rank()

        a_vect = J/len(q)-sub/n
        b_vect = sub / n - (J - 1) / len(q)
        a = a_vect.max()
        b = b_vect.max()

        ks = []
        for i in range(len(a)):
            if a[i] > b[i]:
                ks.append(a[i])
            else:
                ks.append(-b[i])

    return ks

def ranklist(DT):
    # This function takes a panda data frame of gene names and expressions
    # as an input, and output a data frame of gene names and ranks
    ranks = DT.rank(ascending=False, method="first")
    return ranks

def get_metric_func(metric):
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'pearson':
        return pearson

    if metric == 'spearman':
        return spearman

    if metric == 'precision10':
        return precision_10

    if metric == 'precision20':
        return precision_20

    if metric == 'precision50':
        return precision_50

    if metric == 'precision100':
        return precision_100

    raise ValueError(f'Metric "{metric}" not supported.')


def calc_MODZ(array):

    array = array.data.cpu().numpy().astype(float)

    if len(array) == 2:
        return torch.Tensor([np.mean(array, 0)])
    else:
        CM = spearmanr(array.T)[0]

        weights = np.sum(CM, 1) - 1
        weights = weights / np.sum(weights)
        weights = weights.reshape((-1, 1))
        return torch.Tensor([np.dot(array.T, weights).reshape((-1, 1)[0])])
