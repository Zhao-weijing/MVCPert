from dataset import MVC_Dataset
from model_MVC import (
    MVCModel,
    MVCModel_GatedPCC,
    MVCModel_GatedPCC_MaskCPInput,
    MVCModel_GatedPCC_MaskGEInput,
    MVCModel_MaskCPInput,
    MVCModel_MaskGEInput,
    MVCModel_NoCP,
    MVCModel_NoGE,
)
from MVCModel_HyperGate import (
    MVCModel_HyperGate,
    MVCModel_HyperGate_MaskCPInput,
    MVCModel_HyperGate_MaskGEInput,
)
from model_MVC_residual_vae import (
    MVCModel_HyperGateResidualVAE,
    MVCModel_HyperGateResidualVAE_MaskCPInput,
    MVCModel_HyperGateResidualVAE_MaskGEInput,
    MVCModel_HyperGateResidualVAE_NoCP,
    MVCModel_HyperGateResidualVAE_NoGE,
    MVCModel_HyperGateResidualVAE_RandomCPInput,
    MVCModel_HyperGateResidualVAE_RandomGEInput,
)
from memory_utils import log_cuda_memory, release_cuda_resources
from utils import *
import numpy as np
import pandas as pd
import argparse
import contextlib
import json
import warnings
import sys
import os
import h5py
import importlib.util
import torch
from datetime import datetime
from torch.utils.data import Dataset, Sampler
warnings.filterwarnings('ignore')

SUPPORTED_MODEL_TYPES = {
    "MVC",
    "MVC_GatedPCC",
    "MVC_HyperGate",
    "MVC_HyperGateResidualVAE",
    "MVC_HyperGateResidualVAE_NoCP",
    "MVC_HyperGateResidualVAE_NoGE",
}

# 导入 DynamicMultiModalDataset（从指定路径）
# 需要先添加 src 目录到路径，以便 dataset.py 可以导入 utils
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

paired_data_module_path = os.environ.get(
    "AIDD_PAIRED_DATASET_PY",
    "<DATA_ROOT>/Paired_Data/dataset.py",
)
DynamicMultiModalDataset = None
if os.path.exists(paired_data_module_path):
    spec = importlib.util.spec_from_file_location("paired_data_dataset", paired_data_module_path)
    paired_data_dataset_module = importlib.util.module_from_spec(spec)
    # 在执行模块之前，确保模块可以访问 utils
    spec.loader.exec_module(paired_data_dataset_module)
    DynamicMultiModalDataset = paired_data_dataset_module.DynamicMultiModalDataset
else:
    warnings.warn(
        "Optional paired-data loader was not found at "
        f"{paired_data_module_path}. Runs with --use_paired_data True require "
        "AIDD_PAIRED_DATASET_PY or an external paired-data path.",
        RuntimeWarning,
    )


OFFICIAL_SPLIT_LOCK_RELATIVE_PATHS = {
    ("BBBC047", "smiles_split"): "../artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json",
}


GENE_EMBED_DIM_MAP = {
    'Default': 977,
    'CellPLM': 512,
    'geneformer': 256,
    'Openbiomed': 512,
    'scBERT': 200,
    'scFoundation': 3072,
    'scGPT': 512,
    'tGPT': 1024,
    'UCE': 1280,
}


def resolve_official_split_lock_path(args):
    explicit_path = str(getattr(args, "split_lock_path", "") or "").strip()
    if explicit_path:
        return explicit_path

    # paired-data 分支依赖 <DATA_ROOT>/Paired_Data/*.h5 中历史 split_id，
    # 不应再自动套用 official_v1 的 split lock，否则会把两套数据定义混在一起。
    if bool(getattr(args, "use_paired_data", False)):
        return ""

    if not bool(getattr(args, "eval_metric", False)):
        return ""

    key = (
        str(getattr(args, "dataset_name", "") or "").strip(),
        str(getattr(args, "split_data_type", "") or "").strip(),
    )
    relative_path = OFFICIAL_SPLIT_LOCK_RELATIVE_PATHS.get(key)
    if not relative_path:
        return ""

    return os.path.abspath(os.path.join(src_dir, relative_path))


class DynamicMultiModalDatasetWrapper(Dataset):
    """包装器类，将 DynamicMultiModalDataset 的字典输出转换为元组格式"""
    def __init__(self, dynamic_dataset):
        self.dataset = dynamic_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # 将字典格式转换为元组格式: (control_CP, control_GE, target_CP, target_GE, mol_feature, mol_id)
        # 直接返回 tensor，避免不必要的 numpy 转换（DataLoader 会自动处理）
        return (
            sample['control_cp'],  # 已经是 torch.Tensor
            sample['control_ge'],  # 已经是 torch.Tensor
            sample['target_cp'],   # 已经是 torch.Tensor
            sample['target_ge'],    # 已经是 torch.Tensor
            sample['drug'],         # 已经是 torch.Tensor
            sample['smile']         # 字符串
        )
    
    def get_perturbed_centroid(self):
        """返回扰动中心（占位符，如果需要可以从数据中计算）"""
        return self.dataset.perturbed_centroid_CP, self.dataset.perturbed_centroid_GE
    
    def close(self):
        """关闭底层数据集的文件句柄"""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()


def get_split_indices_from_smiles(h5_path, smiles_list, split_type='smiles_split', n_folds=5, random_seed=3407):
    """
    从 HDF5 文件中根据 SMILES 列表获取对应的 split_id 列表
    
    Args:
        h5_path: HDF5 文件路径（可以是 GE 或 CP 文件）
        smiles_list: SMILES 列表
        split_type: 分割类型
        n_folds: 交叉验证折数
        random_seed: 随机种子
    
    Returns:
        split_indices: split_id 列表
    """
    with h5py.File(h5_path, 'r') as f:
        all_smiles = f['combined']['smiles'][:].astype(str)
        all_split_ids = f['combined']['split_id'][:]
    
    # 将 SMILES 列表转换为字符串并去除可能的空格
    smiles_list_clean = [str(s).strip() for s in smiles_list]
    all_smiles_clean = np.array([str(s).strip() for s in all_smiles])
    
    # 将 SMILES 列表转换为集合以便快速查找
    smiles_set = set(smiles_list_clean)
    
    # 找到匹配的 split_id
    matching_indices = np.where(np.isin(all_smiles_clean, smiles_set))[0]
    
    if len(matching_indices) == 0:
        # 如果没找到匹配，尝试更宽松的匹配（检查是否有部分匹配）
        print(f"警告: 在 HDF5 文件中未找到匹配的 SMILES")
        print(f"  输入的 SMILES 数量: {len(smiles_list_clean)}")
        print(f"  输入的 SMILES 示例: {smiles_list_clean[:3] if len(smiles_list_clean) > 0 else 'None'}")
        print(f"  HDF5 中的 SMILES 数量: {len(all_smiles_clean)}")
        print(f"  HDF5 中的 SMILES 示例: {all_smiles_clean[:3] if len(all_smiles_clean) > 0 else 'None'}")
        # 尝试查找是否有交集
        input_set = set(smiles_list_clean)
        h5_set = set(all_smiles_clean)
        intersection = input_set & h5_set
        print(f"  交集数量: {len(intersection)}")
        if len(intersection) > 0:
            print(f"  交集示例: {list(intersection)[:3]}")
            # 使用交集重新匹配
            matching_indices = np.where(np.isin(all_smiles_clean, list(intersection)))[0]
    
    split_indices = np.unique(all_split_ids[matching_indices]).tolist()

    return split_indices


def build_split_from_historical_split_ids(current_data, h5_path, train_split_indices, valid_split_indices, test_split_indices):
    """
    使用旧 paired-data H5 中的 split_id 定义，把当前 data_path 里的样本划分为
    train/valid/test，确保 use_paired_data=True 时训练和评估口径一致。
    """
    with h5py.File(h5_path, 'r') as f:
        all_smiles = np.asarray(f['combined']['smiles'][:]).astype(str)
        all_split_ids = np.asarray(f['combined']['split_id'][:])

    smiles_to_split = {}
    for smile, split_id in zip(all_smiles, all_split_ids):
        smile = str(smile).strip()
        split_id = int(split_id)
        prev = smiles_to_split.get(smile)
        if prev is None:
            smiles_to_split[smile] = split_id
        elif prev != split_id:
            raise ValueError(
                f"SMILES {smile} appears in multiple historical split_ids: {prev} vs {split_id}"
            )

    def _filter_by_split_indices(indices):
        index_set = set(int(i) for i in indices)
        mask = np.array(
            [smiles_to_split.get(str(s).strip(), -1) in index_set for s in current_data['canonical_smiles']],
            dtype=bool,
        )
        return {k: v[mask] for k, v in current_data.items()}

    pair = _filter_by_split_indices(train_split_indices)
    pairv = _filter_by_split_indices(valid_split_indices)
    pairt = _filter_by_split_indices(test_split_indices)
    return pair, pairv, pairt


def parse_args():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = argparse.ArgumentParser(description="Arguments for training gene+image=MVC")
    parser.add_argument("--dataset_name", type=str, default="BBBC047") # 'BBBC047', 'BBBC036'
    
    parser.add_argument("--gene_encoder_type", type=str, default='Default', help='gene_encoder_feature(Default_978, CellPLM_512,\
        geneformer_256, Openbiomed_512, scBERT_200, scFoundation_3072, scGPT_512, tGPT_1024, UCE_1280)')
    parser.add_argument("--dev", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--molecule_feature", type=str, default='ECFP4', help='molecule_feature(KPGT_emb2304, ECFP4_emb2048, InfoAlign_emb300,\
        ChemBERTa2_emb384, MolT5_emb768, Chemprop_emb300, MolCLR_emb512, Mole_BERT_emb300, GeminiMol_emb2048, Ouroboros_emb2048, UniMol_512, UniMolV2_1024)')
    parser.add_argument("--initialization_model", type=str, default='random', help='molecule_feature(pretrain_shRNA, random)')
    parser.add_argument("--split_data_type", type=str, default='smiles_split', help='split_data_type(random_split, smiles_split, cell_split)')
    parser.add_argument("--train_cell_count", type=str, default='None', help='if cell_split, train_cell_count=10,50,all, else None')

    parser.add_argument("--batch_size", type=int, default=1000)  # 500 1000
    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--n_latent", type=int, default=1536)
    parser.add_argument("--molecule_feature_embed_dim", nargs='+', type=int, default=[512])
    parser.add_argument("--learning_rate", type=float, default=1e-3) ## 1e-3 default
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4) ## 1e-5 default

    # parser.add_argument("--train_flag", type=bool, default=True)
    parser.add_argument("--train_flag", type=str2bool, default=True)  # 测试跨数据集的性能
    parser.add_argument("--eval_metric", type=str2bool, default=True)
    parser.add_argument("--predict_profile", type=str2bool, default=False) # profile比较大，如果不需要预测profile，可以节省内存和计算，尤其是在使用动态数据集时
    parser.add_argument("--run_ablation", type=str2bool, default=False, help='是否运行消融实验')
    ##############################################################################################
    parser.add_argument("--model_type", type=str, default='MVC_HyperGateResidualVAE', choices=sorted(SUPPORTED_MODEL_TYPES), help='Model type included in the lightweight MVCPert release.')
    parser.add_argument("--n_attn_heads", type=int, default=8, help='Number of attention heads used by HyperGate variants')
    parser.add_argument("--n_fusion_layers", type=int, default=2, help='Number of fusion layers used by HyperGate variants')
    parser.add_argument("--use_paired_data", type=str2bool, default=True, help='是否使用 <DATA_ROOT>/Paired_Data/ 路径下的数据集')
    parser.add_argument("--use_augmentation", type=str2bool, default=True, help='是否启用数据增强（仅对训练集）')
    parser.add_argument("--molecule_balanced_sampling", type=str2bool, default=False, help="训练集是否按 canonical_smiles 做分子均衡采样（仅替换 train loader sampler）")
    parser.add_argument("--molecule_balanced_samples_per_molecule", type=int, default=1, help="分子均衡采样时，每个 unique molecule 在每个 epoch 中采样的记录数")
    parser.add_argument("--molecule_balanced_temp_mean", type=str2bool, default=False, help="是否将同一分子采样得到的 k 条记录先均值成一个临时训练样本")
    parser.add_argument("--aug_ge_drop_prob", type=float, default=0.18, help="MVC_Dataset 训练集 GE feature dropout 概率")
    parser.add_argument("--aug_cp_drop_prob", type=float, default=0.05, help="MVC_Dataset 训练集 CP feature dropout 概率")
    parser.add_argument("--aug_mol_drop_prob", type=float, default=0.25, help="MVC_Dataset 训练集分子 feature dropout 概率")
    parser.add_argument("--aug_mol_noise_std", type=float, default=0.01, help="MVC_Dataset 训练集分子高斯噪声强度")
    parser.add_argument("--aug_noise_std", type=float, default=0.005, help="MVC_Dataset 训练集 CP/GE student-t 噪声强度")
    parser.add_argument("--aug_winsor_q_low", type=float, default=0.001, help="MVC_Dataset 训练集 winsor 下分位数")
    parser.add_argument("--aug_winsor_q_high", type=float, default=0.999, help="MVC_Dataset 训练集 winsor 上分位数")
    parser.add_argument("--aug_winsor_after_norm", type=str2bool, default=True, help="MVC_Dataset 训练集是否在归一化后做 winsorize")
    parser.add_argument("--output_tag", type=str, default="", help="输出目录附加后缀，避免覆盖已有实验")
    parser.add_argument("--run_date_override", type=str, default="", help="可选：覆盖输出目录中的日期，例如 2026-04-08")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="<ARTIFACT_ROOT>",
        help="输出目录根路径（默认 <ARTIFACT_ROOT>）",
    )
    parser.add_argument("--loss_amplification", type=float, default=None, help="已弃用；HyperGate 不再使用 loss amplification")
    parser.add_argument("--grad_clip_norm", type=float, default=5.0, help="HyperGate 的梯度裁剪阈值，<=0 表示关闭")
    parser.add_argument("--hypergate_small_data_mode", action="store_true", help="为 BBBC036 这类小数据集启用更稳的 HyperGate 训练配置")
    parser.add_argument("--hypergate_disable_hyper_refinement", type=str2bool, default=False, help="是否禁用 HyperGate 中的 hypergraph refinement，仅保留 tokenizer + transformer + gates")
    parser.add_argument("--pcc_weight", type=float, default=0.75, help="HyperGate/GatedPCC 中原始 profile PCC 辅助损失权重")
    parser.add_argument("--delta_pcc_weight", type=float, default=0.25, help="HyperGate/GatedPCC 中 delta profile PCC 辅助损失权重")
    parser.add_argument("--cp_pcc_weight", type=float, default=None, help="HyperGate 中 CP profile PCC 权重；默认回退到 pcc_weight")
    parser.add_argument("--ge_pcc_weight", type=float, default=None, help="HyperGate 中 GE profile PCC 权重；默认回退到 pcc_weight")
    parser.add_argument("--cp_delta_pcc_weight", type=float, default=None, help="HyperGate 中 CP delta PCC 权重；默认回退到 delta_pcc_weight")
    parser.add_argument("--ge_delta_pcc_weight", type=float, default=None, help="HyperGate 中 GE delta PCC 权重；默认回退到 delta_pcc_weight")
    parser.add_argument("--residual_vae_latent_dim", type=int, default=128, help="Residual refiner latent dimension")
    parser.add_argument("--residual_vae_hidden_dim", type=int, default=1536, help="Residual refiner hidden dimension")
    parser.add_argument("--residual_vae_kl_weight", type=float, default=1e-4, help="Residual refiner KL regularization weight")
    parser.add_argument("--residual_vae_kl_warmup_epochs", type=int, default=10, help="Residual refiner KL warmup epochs; 0 disables warmup")
    parser.add_argument("--residual_vae_disentangle_weight", type=float, default=1e-3, help="Shared/private residual disentanglement weight")
    parser.add_argument("--residual_vae_disentangle_mode", type=str, default="corr", choices=["none", "corr", "hsic"], help="Shared/private residual disentanglement mode")
    parser.add_argument("--residual_vae_hsic_sigma", type=float, default=1.0, help="RBF kernel sigma for HSIC disentanglement")
    parser.add_argument("--residual_vae_prior_recon_weight", type=float, default=0.5, help="Prior-mean reconstruction auxiliary weight")
    parser.add_argument("--residual_vae_base_recon_weight", type=float, default=1.0, help="MVC_HyperGateResidualVAE 中 deterministic base 路径的重建锚定权重")
    parser.add_argument("--residual_vae_correction_scale", type=float, default=0.25, help="MVC_HyperGateResidualVAE 中 VAE correction 的全局缩放系数")
    parser.add_argument("--residual_vae_correction_l2_weight", type=float, default=0.05, help="MVC_HyperGateResidualVAE 中 correction 幅度的 L2 正则权重")
    parser.add_argument("--residual_vae_private_orth_weight", type=float, default=0.0, help="MVC_HyperGateResidualVAE 中 private correction 的跨模态去相似损失权重")
    parser.add_argument("--residual_vae_private_remainder_weight", type=float, default=0.0, help="MVC_HyperGateResidualVAE 中 private correction 的 remainder 监督权重，拟合 target_correction - shared_correction")
    parser.add_argument("--residual_vae_shared_infonce_weight", type=float, default=0.0, help="MVC_HyperGateResidualVAE 中 shared correction 的 InfoNCE 权重，用 shared 分支贴近跨模态公共部分并远离 private remainder")
    parser.add_argument("--residual_vae_shared_infonce_temperature", type=float, default=0.1, help="MVC_HyperGateResidualVAE 中 shared InfoNCE 的温度系数")
    parser.add_argument("--residual_vae_modal_mask_prob", type=float, default=0.15, help="Probability of masking one baseline modality during residual-refiner training")
    parser.add_argument("--residual_vae_sample_train", type=str2bool, default=True, help="Whether to sample from the posterior during training; False uses posterior means")
    parser.add_argument("--residual_vae_training_stage", type=str, default="full_only", choices=["full_only", "three_view"], help="MVC_HyperGateResidualVAE 训练阶段：只训练 full-view 或 full-view+随机单辅助视角")
    parser.add_argument("--residual_vae_geview_weight", type=float, default=0.2, help="MVC_HyperGateResidualVAE GE-view 辅助损失权重")
    parser.add_argument("--residual_vae_cpview_weight", type=float, default=0.2, help="MVC_HyperGateResidualVAE CP-view 辅助损失权重")
    parser.add_argument("--residual_vae_aux_view_mode", type=str, default="random", choices=["random", "ge", "cp", "alternate"], help="MVC_HyperGateResidualVAE 每 batch 辅助视角采样方式")
    parser.add_argument("--mse_cp_weight", type=float, default=1.0, help="MVC_GatedPCC 中 CP 重建损失权重")
    parser.add_argument("--mse_ge_weight", type=float, default=1.0, help="MVC_GatedPCC 中 GE 重建损失权重")
    parser.add_argument("--split_lock_path", type=str, default="", help="可选：split lock JSON 路径，提供后优先按 lock 划分 train/valid/test")
    parser.add_argument("--write_split_lock_path", type=str, default="", help="可选：将本次实际使用的 split 写出为 lock JSON")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count for release models (default 4)",
    )
    parser.add_argument(
        "--data_path_override",
        type=str,
        default="",
        help="可选：覆盖默认数据路径（例如 <DATA_ROOT>/MVC_BBBC047/Paired_CP_GE_Data.h5）",
    )
    parser.add_argument(
        "--use_dose_feature",
        type=str2bool,
        default=False,
        help="是否将 paired H5 中的 pert_dose 作为额外输入特征拼接到分子特征后",
    )
    parser.add_argument(
        "--dose_ordinal_weight",
        type=float,
        default=0.0,
        help="same-compound 多剂量 GE-delta ordinal retrieval 辅助损失权重（默认关闭）",
    )
    parser.add_argument(
        "--dose_ordinal_temperature",
        type=float,
        default=0.2,
        help="dose ordinal retrieval CE 的温度参数",
    )
    parser.add_argument(
        "--dose_ordinal_margin_scale",
        type=float,
        default=0.25,
        help="dose ordinal retrieval margin 的最大尺度（按组内 dose-rank gap 归一化）",
    )
    parser.add_argument(
        "--pretrained_checkpoint_path",
        type=str,
        default="",
        help="可选：加载预训练 checkpoint 做 partial initialization；仅加载参数名与 shape 同时匹配的权重",
    )
    parser.add_argument(
        "--freeze_module_prefixes",
        nargs='*',
        default=[],
        help="可选：训练前冻结这些模块前缀下的参数，例如 cp_encoder ge_encoder mol_encoder cp_pool ge_pool",
    )

    args = parser.parse_args()
    return args


def build_mvc_dataset_kwargs(
    args,
    mol_feature_type,
    gene_encoder_type,
    mol_id,
    paired_data,
    perturbed_centroid_cp,
    perturbed_centroid_ge,
    normalization_stats=None,
    is_train=False,
):
    kwargs = {
        "dataset_name": args.dataset_name,
        "mol_feature_type": mol_feature_type,
        "gene_encoder_type": gene_encoder_type,
        "mol_id": mol_id,
        "paired_data": paired_data,
        "perturbed_centroid_CP": perturbed_centroid_cp,
        "perturbed_centroid_GE": perturbed_centroid_ge,
        "augment": bool(args.use_augmentation) if is_train else False,
        "use_dose_feature": bool(getattr(args, "use_dose_feature", False)),
    }
    if normalization_stats is not None:
        kwargs["normalization_stats"] = normalization_stats
    if is_train:
        kwargs.update(
            {
                "ge_drop_prob": float(args.aug_ge_drop_prob),
                "cp_drop_prob": float(args.aug_cp_drop_prob),
                "mol_drop_prob": float(args.aug_mol_drop_prob),
                "mol_noise_std": float(args.aug_mol_noise_std),
                "noise_std": float(args.aug_noise_std),
                "winsor_q_low": float(args.aug_winsor_q_low),
                "winsor_q_high": float(args.aug_winsor_q_high),
                "winsor_after_norm": bool(args.aug_winsor_after_norm),
            }
        )
    return kwargs


class StreamTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_output_dir(args, run_date=None, base_dir="<ARTIFACT_ROOT>"):
    if run_date is None:
        run_date = datetime.now().date().isoformat()
    base_dir = getattr(args, "output_base_dir", base_dir)
    model_out_name = args.model_type if not getattr(args, "output_tag", "") else f"{args.model_type}_{args.output_tag}"
    return os.path.join(
        base_dir,
        str(run_date),
        f"{model_out_name}_{args.dataset_name}_{args.split_data_type}",
        f"{args.molecule_feature}_{args.gene_encoder_type}",
        "",
    )


def compute_numeric_column_means(df):
    numeric_df = df.select_dtypes(include=[np.number])
    mean_dict = {}
    for col in numeric_df.columns:
        mean_value = numeric_df[col].mean()
        if pd.notna(mean_value):
            mean_dict[f"{col}_mean"] = float(mean_value)
    return mean_dict


METRICS_MEAN_LOG_ORDER = [
    ("CP_PCC", "cp_pred_pearson_mean"),
    ("GE_PCC", "ge_pred_pearson_mean"),
    ("systema_CP_PCC", "systema_pearson_CP_mean"),
    ("systema_GE_PCC", "systema_pearson_GE_mean"),
    ("DEG_CP_PCC", "DEG_cp_pred_pearson_mean"),
    ("DEG_GE_PCC", "DEG_ge_pred_pearson_mean"),
    ("CP_RMSE", "cp_pred_rmse_mean"),
    ("GE_RMSE", "ge_pred_rmse_mean"),
    ("systema_CP_RMSE", "systema_rmse_CP_mean"),
    ("systema_GE_RMSE", "systema_rmse_GE_mean"),
    ("DEG_CP_RMSE", "DEG_cp_pred_rmse_mean"),
    ("DEG_GE_RMSE", "DEG_ge_pred_rmse_mean"),
    ("CP_Neg_Precision100", "cp_pred_neg_precision100_mean"),
    ("GE_Neg_Precision100", "ge_pred_neg_precision100_mean"),
    ("CP_Pos_Precision100", "cp_pred_pos_precision100_mean"),
    ("GE_Pos_Precision100", "ge_pred_pos_precision100_mean"),
]


def order_metrics_mean_for_log(metrics_mean):
    remaining_metrics = dict(_to_jsonable(metrics_mean or {}))
    ordered_metrics = []
    for display_name, metric_key in METRICS_MEAN_LOG_ORDER:
        if metric_key in remaining_metrics:
            ordered_metrics.append((display_name, remaining_metrics.pop(metric_key)))
    for metric_key in sorted(remaining_metrics.keys()):
        ordered_metrics.append((metric_key, remaining_metrics[metric_key]))
    return ordered_metrics


def log_metrics_mean(metrics_mean, label="Mean test metrics"):
    ordered_metrics = order_metrics_mean_for_log(metrics_mean)
    if not ordered_metrics:
        print(f"{label}: {{}}")
        return

    print(f"{label}:")
    for display_name, metric_value in ordered_metrics:
        print(f"{display_name}: {metric_value}")


def save_metrics_mean(df_rec, metrics_mean_path, log_label="Mean test metrics"):
    metrics_mean = compute_numeric_column_means(df_rec)
    pd.DataFrame([metrics_mean]).to_csv(metrics_mean_path, index=False)
    print('Saved mean test metrics to', metrics_mean_path)
    log_metrics_mean(metrics_mean, label=log_label)
    return metrics_mean


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def extract_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()) and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            return checkpoint

    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def load_partial_checkpoint_into_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = extract_state_dict_from_checkpoint(checkpoint)
    model_state = model.state_dict()

    matched_state = {}
    missing_in_model = 0
    shape_mismatches = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            missing_in_model += 1
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatches.append(
                {
                    "key": str(key),
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
            continue
        matched_state[key] = value.detach().cpu().clone()

    if not matched_state:
        raise ValueError(
            f"No compatible tensors found in checkpoint {checkpoint_path} for partial initialization"
        )

    updated_state = dict(model_state)
    updated_state.update(matched_state)
    model.load_state_dict(updated_state)

    total_tensor_count = int(len(model_state))
    total_parameter_count = int(sum(t.numel() for t in model_state.values()))
    loaded_parameter_count = int(sum(t.numel() for t in matched_state.values()))
    summary = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "loaded_tensor_count": int(len(matched_state)),
        "total_tensor_count": total_tensor_count,
        "loaded_tensor_fraction": float(len(matched_state) / total_tensor_count) if total_tensor_count else float("nan"),
        "loaded_parameter_count": loaded_parameter_count,
        "total_parameter_count": total_parameter_count,
        "loaded_parameter_fraction": float(loaded_parameter_count / total_parameter_count)
        if total_parameter_count
        else float("nan"),
        "missing_in_model_count": int(missing_in_model),
        "shape_mismatch_count": int(len(shape_mismatches)),
        "loaded_tensor_examples": sorted(matched_state.keys())[:10],
        "shape_mismatch_examples": shape_mismatches[:10],
    }
    return summary


def freeze_model_modules_by_prefix(model, module_prefixes):
    prefixes = [str(prefix).strip() for prefix in (module_prefixes or []) if str(prefix).strip()]
    if not prefixes:
        return {
            "requested_prefixes": [],
            "matched_parameter_count": 0,
            "matched_parameter_examples": [],
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        }

    matched_names = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            parameter.requires_grad = False
            matched_names.append(name)

    trainable_parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_parameter_count = int(sum(p.numel() for p in model.parameters()))
    return {
        "requested_prefixes": prefixes,
        "matched_parameter_count": int(len(matched_names)),
        "matched_parameter_examples": matched_names[:20],
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_fraction": float(trainable_parameter_count / total_parameter_count)
        if total_parameter_count
        else float("nan"),
    }


def compute_smiles_overlap_summary(train_smiles, valid_smiles, test_smiles):
    train_set = set(np.asarray(train_smiles).astype(str))
    valid_set = set(np.asarray(valid_smiles).astype(str))
    test_set = set(np.asarray(test_smiles).astype(str))
    return {
        "train_valid": int(len(train_set & valid_set)),
        "train_test": int(len(train_set & test_set)),
        "valid_test": int(len(valid_set & test_set)),
        "train_unique": int(len(train_set)),
        "valid_unique": int(len(valid_set)),
        "test_unique": int(len(test_set)),
    }


def log_smiles_overlap(prefix, train_smiles, valid_smiles, test_smiles, enforce_zero=False):
    overlap_summary = compute_smiles_overlap_summary(train_smiles, valid_smiles, test_smiles)
    print(
        f"{prefix} SMILES overlap -> "
        f"train-valid: {overlap_summary['train_valid']}, "
        f"train-test: {overlap_summary['train_test']}, "
        f"valid-test: {overlap_summary['valid_test']}"
    )
    if enforce_zero and any(
        overlap_summary[key] != 0 for key in ["train_valid", "train_test", "valid_test"]
    ):
        raise ValueError(f"{prefix} SMILES overlap is not zero: {overlap_summary}")
    return overlap_summary


def resolve_validation_loss_column(epoch_result, model_type):
    candidate_columns = []
    candidate_columns.extend(['valid_loss', 'valid_total_loss', 'loss'])
    for column in candidate_columns:
        if column in epoch_result.columns:
            return column
    raise KeyError(f"No validation loss column found for {model_type}; available columns: {list(epoch_result.columns)}")


def resolve_validation_selection(epoch_result, args):
    return resolve_validation_loss_column(epoch_result, model_type=args.model_type), "min"


def unpack_prediction_profile_output(profile_output):
    if isinstance(profile_output, dict):
        return (
            profile_output["control_cp"],
            profile_output["control_ge"],
            profile_output["target_cp"],
            profile_output["target_ge"],
            profile_output["cp_pred"],
            profile_output["ge_pred"],
            profile_output["smiles"],
        )
    return profile_output


def get_paired_data_paths(dataset_name):
    if dataset_name == 'BBBC047':
        return (
            "<DATA_ROOT>/Paired_Data/BBBC047_GE.h5",
            "<DATA_ROOT>/Paired_Data/BBBC047_CP.h5",
        )
    if dataset_name == 'BBBC036':
        return (
            "<DATA_ROOT>/Paired_Data/BBBC036_GE.h5",
            "<DATA_ROOT>/Paired_Data/BBBC036_CP.h5",
        )
    raise ValueError(f"不支持的数据集名称: {dataset_name}")


def infer_input_dims_from_data(data):
    required_keys = ("control_CP", "control_GE", "target_CP", "target_GE")
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"paired H5 missing keys: {missing}")
    return int(data["control_CP"].shape[1]), int(data["control_GE"].shape[1])


def assert_finite_validation_history(epoch_result, valid_loss_col, output_dir):
    valid_values = pd.to_numeric(epoch_result[valid_loss_col], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(valid_values).any():
        return
    raise RuntimeError(
        f"No finite values were produced for {valid_loss_col}. "
        f"Training diverged before saving a checkpoint under {output_dir}."
    )


def validate_model_configuration(args):
    if args.model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"{args.model_type} is not included in this lightweight release. "
            f"Supported model types: {supported}"
        )


def _extract_dataset_smiles_for_sampling(dataset):
    if hasattr(dataset, "Image_Gene_data") and isinstance(getattr(dataset, "Image_Gene_data"), dict):
        image_gene_data = getattr(dataset, "Image_Gene_data")
        if "canonical_smiles" in image_gene_data:
            return np.asarray(image_gene_data["canonical_smiles"]).astype(str)
    if hasattr(dataset, "mol_id"):
        return np.asarray(getattr(dataset, "mol_id")).astype(str)
    raise ValueError("Unable to extract canonical_smiles from dataset for molecule-balanced sampling")


def _build_molecule_groups(smiles):
    unique_smiles, inverse, counts = np.unique(smiles, return_inverse=True, return_counts=True)
    grouped_indices = [np.flatnonzero(inverse == idx) for idx in range(len(unique_smiles))]
    return unique_smiles.astype(str), grouped_indices, counts.astype(np.int64)


def _summarize_molecule_groups(smiles, samples_per_molecule):
    unique_smiles, grouped_indices, counts = _build_molecule_groups(smiles)
    summary = {
        "train_rows": int(len(smiles)),
        "unique_molecules": int(len(unique_smiles)),
        "count_min": int(np.min(counts)),
        "count_median": float(np.median(counts)),
        "count_p95": float(np.quantile(counts, 0.95)),
        "count_max": int(np.max(counts)),
        "samples_per_molecule": int(samples_per_molecule),
        "effective_samples_per_epoch": int(len(unique_smiles) * samples_per_molecule),
        "epoch_coverage_ratio": float((len(unique_smiles) * samples_per_molecule) / max(len(smiles), 1)),
        "molecules_requiring_replacement": int(np.sum(counts < samples_per_molecule)),
    }
    return unique_smiles, grouped_indices, counts, summary


class MoleculeBalancedSampler(Sampler):
    """每个 epoch 对每个分子精确采样固定条数，避免高重复分子主导训练。"""

    def __init__(self, grouped_indices, samples_per_molecule, random_seed, shuffle=True):
        self.grouped_indices = [np.asarray(indices, dtype=np.int64) for indices in grouped_indices]
        self.samples_per_molecule = int(samples_per_molecule)
        self.random_seed = int(random_seed)
        self.shuffle = bool(shuffle)
        self.num_samples = int(len(self.grouped_indices) * self.samples_per_molecule)
        self._epoch = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.random_seed + self._epoch)
        self._epoch += 1
        sampled_indices = np.empty(self.num_samples, dtype=np.int64)
        cursor = 0
        for indices in self.grouped_indices:
            replace = len(indices) < self.samples_per_molecule
            chosen = rng.choice(indices, size=self.samples_per_molecule, replace=replace)
            sampled_indices[cursor: cursor + self.samples_per_molecule] = chosen
            cursor += self.samples_per_molecule
        if self.shuffle:
            rng.shuffle(sampled_indices)
        return iter(sampled_indices.tolist())


def build_molecule_balanced_sampler(dataset, random_seed, samples_per_molecule=1):
    smiles = _extract_dataset_smiles_for_sampling(dataset)
    samples_per_molecule = int(samples_per_molecule)
    if samples_per_molecule <= 0:
        raise ValueError(
            f"molecule_balanced_samples_per_molecule must be a positive integer, got {samples_per_molecule}"
        )
    unique_smiles, grouped_indices, counts, summary = _summarize_molecule_groups(
        smiles,
        samples_per_molecule=samples_per_molecule,
    )
    sampler = MoleculeBalancedSampler(
        grouped_indices=grouped_indices,
        samples_per_molecule=samples_per_molecule,
        random_seed=random_seed,
        shuffle=True,
    )
    return sampler, summary


class MoleculeBalancedTempMeanDataset(Dataset):
    """每个临时样本由同一分子下采样的 k 条记录均值构成，模型与损失保持不变。"""

    def __init__(self, base_dataset, samples_per_molecule, random_seed):
        self.base_dataset = base_dataset
        self.samples_per_molecule = int(samples_per_molecule)
        self.random_seed = int(random_seed)
        if self.samples_per_molecule <= 0:
            raise ValueError(
                f"molecule_balanced_samples_per_molecule must be a positive integer, got {self.samples_per_molecule}"
            )

        smiles = _extract_dataset_smiles_for_sampling(base_dataset)
        self.unique_smiles, self.grouped_indices, self.group_counts, self.summary = _summarize_molecule_groups(
            smiles,
            samples_per_molecule=self.samples_per_molecule,
        )
        self.mol_id = self.unique_smiles
        self.Image_Gene_data = {"canonical_smiles": self.unique_smiles}
        self.normalization_stats = getattr(base_dataset, "normalization_stats", None)
        self.perturbed_centroid_CP = getattr(base_dataset, "perturbed_centroid_CP", None)
        self.perturbed_centroid_GE = getattr(base_dataset, "perturbed_centroid_GE", None)

    def __len__(self):
        return int(len(self.unique_smiles))

    def get_perturbed_centroid(self):
        return self.perturbed_centroid_CP, self.perturbed_centroid_GE

    def get_normalization_stats(self):
        return self.normalization_stats

    def _sample_row_indices(self, molecule_index):
        indices = self.grouped_indices[molecule_index]
        replace = len(indices) < self.samples_per_molecule
        chosen = np.random.choice(indices, size=self.samples_per_molecule, replace=replace)
        return np.asarray(chosen, dtype=np.int64)

    def _apply_molecule_augment(self, mol_feature):
        if self.base_dataset.use_dose_feature and mol_feature.shape[0] >= 1:
            mol_base = mol_feature[:-1]
            mol_dose = mol_feature[-1:]
            mol_base = apply_noise(mol_base, noise_std=self.base_dataset._mol_noise_std, dist='gaussian')
            mol_base = apply_feature_dropout(
                mol_base,
                drop_prob=self.base_dataset._mol_drop_prob,
                keep_zeros=True,
            )
            return np.concatenate([mol_base, mol_dose], axis=0)
        mol_feature = apply_noise(mol_feature, noise_std=self.base_dataset._mol_noise_std, dist='gaussian')
        mol_feature = apply_feature_dropout(
            mol_feature,
            drop_prob=self.base_dataset._mol_drop_prob,
            keep_zeros=True,
        )
        return mol_feature

    def __getitem__(self, molecule_index):
        row_indices = self._sample_row_indices(int(molecule_index))
        samples = [self.base_dataset._get_raw_sample(int(row_index)) for row_index in row_indices]

        control_cp = np.stack([sample[0] for sample in samples], axis=0).mean(axis=0).astype(np.float32)
        control_ge = np.stack([sample[1] for sample in samples], axis=0).mean(axis=0).astype(np.float32)
        target_cp = np.stack([sample[2] for sample in samples], axis=0).mean(axis=0).astype(np.float32)
        target_ge = np.stack([sample[3] for sample in samples], axis=0).mean(axis=0).astype(np.float32)
        mol_feature = np.stack([sample[4] for sample in samples], axis=0).mean(axis=0).astype(np.float32)

        if self.base_dataset.augment:
            control_cp, control_ge = self.base_dataset._apply_intra_sample_augment(control_cp, control_ge)
            mol_feature = self._apply_molecule_augment(mol_feature.astype(np.float32))

        return (
            control_cp.astype(np.float32),
            control_ge.astype(np.float32),
            target_cp.astype(np.float32),
            target_ge.astype(np.float32),
            mol_feature.astype(np.float32),
            self.unique_smiles[molecule_index],
        )


def build_run_summary(
    args,
    output_dir,
    predict_dir,
    train_log_path,
    best_epoch,
    test_metrics_mean,
    run_timestamp,
    output_files=None,
    resolved_config=None,
    dataset_sizes=None,
):
    summary = {
        "run_timestamp": run_timestamp,
        "run_date": str(run_timestamp).split("T")[0],
        "model_type": args.model_type,
        "dataset_name": args.dataset_name,
        "split_data_type": args.split_data_type,
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "hyperparameters": _to_jsonable(vars(args)),
        "metrics_mean": _to_jsonable(test_metrics_mean or {}),
        "paths": {
            "output_dir": output_dir,
            "predict_dir": predict_dir,
            "train_log": train_log_path,
        },
    }
    if output_files:
        summary["output_files"] = _to_jsonable(output_files)
    if resolved_config:
        summary["resolved_config"] = _to_jsonable(resolved_config)
    if dataset_sizes:
        summary["dataset_sizes"] = _to_jsonable(dataset_sizes)
    return summary


def train_MVCModel(args):
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    run_date = str(getattr(args, "run_date_override", "") or "").strip() or run_timestamp.split("T")[0]
    local_out = build_output_dir(args, run_date=run_date)
    output_dir_exists = os.path.exists(local_out)
    os.makedirs(local_out, exist_ok=True)
    train_log_path = os.path.join(local_out, "train.log")

    with open(train_log_path, "a", encoding="utf-8") as log_file:
        stdout_tee = StreamTee(sys.stdout, log_file)
        stderr_tee = StreamTee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Run timestamp: {run_timestamp}")
            print(f"Experiment output directory: {local_out}")
            print("Directory already exists" if output_dir_exists else "Directory created successfully")
            return _train_MVCModel_impl(
                args=args,
                local_out=local_out,
                train_log_path=train_log_path,
                run_timestamp=run_timestamp,
            )


def _train_MVCModel_impl(args, local_out, train_log_path, run_timestamp):
    print(args)
    validate_model_configuration(args)
    random_seed = args.seed
    setup_seed(random_seed)
    dev = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    best_epoch = None
    test_metrics_mean = {}
    output_files = {}
    split_lock_meta = None
    split_lock_source = "runtime"
    split_lock_path = resolve_official_split_lock_path(args)
    write_split_lock_path = str(getattr(args, "write_split_lock_path", "") or "").strip()
    split_lock_written_path = None

    # # 判断是否使用 Paired_Data 路径
    use_paired_data = args.use_paired_data
    
    ge_h5_path = None
    cp_h5_path = None
    if use_paired_data:
        ge_h5_path, cp_h5_path = get_paired_data_paths(args.dataset_name)

    # 使用默认路径（可被 --data_path_override 覆盖）
    if args.dataset_name == 'BBBC047':
        data_path = "<DATA_ROOT>/MVC_BBBC047/Paired_CP_GE_Data_one_sample_per_molecule.h5"
    elif args.dataset_name == 'BBBC036':
        data_path = "<DATA_ROOT>/MVC_BBBC036/Paired_CP_GE_Data.h5"
    elif args.data_path_override:
        data_path = args.data_path_override
    else:
        raise ValueError(
            f"Unsupported dataset_name without --data_path_override: {args.dataset_name}"
        )

    if args.data_path_override:
        data_path = args.data_path_override
    print(f"Using data_path: {data_path}")
    
    data = load_from_HDF(data_path)
    # print(data)
    
    print('all data: total smiles / unique smiles', len(data['canonical_smiles']), len(set(data['canonical_smiles'])))

    train_flag = args.train_flag
    eval_metric = args.eval_metric
    predict_profile = args.predict_profile

    feat_type = args.molecule_feature
    feature_dim_map = {
        'KPGT': 2304,
        'ECFP4': 2048,
        'InfoAlign': 300,
        'ChemBERTa2': 384,
        'MolT5': 768,
        'Chemprop': 300,
        'MolCLR': 512,
        'Mole_BERT': 300,
        'GeminiMol': 2048,
        'Ouroboros': 2048,
        'UniMol': 512,
        'UniMolV2': 1024
    }
    molecule_feature_dim = feature_dim_map.get(feat_type)
    if molecule_feature_dim is None:
        raise ValueError(f"Unknown feature type: {feat_type}")
    if bool(args.use_dose_feature):
        if 'pert_dose' not in data:
            raise ValueError("use_dose_feature=True but loaded paired H5 does not contain pert_dose")
        dose_values = np.asarray(data['pert_dose'], dtype=np.float32)
        if not np.isfinite(dose_values).all():
            raise ValueError("use_dose_feature=True but pert_dose contains non-finite values")
        if np.any(dose_values <= 0):
            raise ValueError("use_dose_feature=True but pert_dose contains non-positive values")
        molecule_feature_dim += 1
    
    Gene_encoder_type = args.gene_encoder_type
    # Default_978, CellPLM_512, geneformer_256, Openbiomed_512, scBERT_200, scFoundation_3072, scGPT_512, tGPT_1024, UCE_1280
    
    input_images, input_genes = infer_input_dims_from_data(data)
    gene_emb_dim = GENE_EMBED_DIM_MAP.get(Gene_encoder_type)
    if gene_emb_dim is None:
        raise ValueError(f"Unknown gene encoder type: {Gene_encoder_type}")
    if Gene_encoder_type == 'Default':
        gene_emb_dim = input_genes
        
    split_type = args.split_data_type
    train_cell_count = args.train_cell_count
    n_folds = 5

    n_epochs = args.n_epochs
    n_latent = args.n_latent
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    beta = args.beta
    dropout = args.dropout
    weight_decay = args.weight_decay

    print('Model_type:', args.model_type, 'Molecule_encoder:', feat_type, 'Gene_encoder:', Gene_encoder_type, 'split_type:', split_type, 'learning_rate:', learning_rate,\
        'dropout:', dropout, 'beta:', beta, 'weight decay:', weight_decay, 'use_augmentation:', args.use_augmentation)

    if split_type != 'cells_split':
        print('train_cell_count used for cell_split')
        assert (train_cell_count == 'None')

    print(local_out)

    paired_historical_split = bool(use_paired_data and not split_lock_path)
    historical_train_split_indices = None
    historical_valid_split_indices = None
    historical_test_split_indices = None
    if paired_historical_split:
        if split_type in ['random_split', 'smiles_split']:
            historical_train_split_indices = [0, 1, 2]
            historical_valid_split_indices = [3]
            historical_test_split_indices = [4]
        elif split_type == 'cells_split':
            historical_train_split_indices = [0, 1, 2]
            historical_valid_split_indices = [3]
            historical_test_split_indices = [4]
        else:
            raise ValueError(f"不支持的 split_type: {split_type}")

        pair, pairv, pairt = build_split_from_historical_split_ids(
            current_data=data,
            h5_path=ge_h5_path,
            train_split_indices=historical_train_split_indices,
            valid_split_indices=historical_valid_split_indices,
            test_split_indices=historical_test_split_indices,
        )
        split_lock_source = "historical_paired_split_id"
        print('Using historical paired-data split_id instead of official_v1 split lock')
    if split_lock_path:
        if not str(getattr(args, "split_lock_path", "") or "").strip():
            print(f'Auto-selected official split lock: {os.path.abspath(split_lock_path)}')
        split_lock_payload = load_split_lock(
            split_lock_path,
            expected_dataset_name=args.dataset_name,
            expected_split_data_type=split_type,
        )
        pair, pairv, pairt, split_lock_meta = split_data_with_lock(data, split_lock_payload)
        split_lock_source = "lock_file"
        print(f'Using split lock: {os.path.abspath(split_lock_path)}')
    elif paired_historical_split:
        pass
    elif split_type in ['random_split', 'smiles_split']:
        pair, pairv, pairt = split_data(data, n_folds=n_folds, split_type=split_type, rnds=random_seed)
    elif split_type == 'cells_split':
        pair, pairv, pairt = split_data_cid(data, train_cell_count=train_cell_count)
    else:
        raise ValueError(f"不支持的 split_type: {split_type}")

    if write_split_lock_path:
        split_lock_payload_to_write = build_split_lock_payload(
            dataset_name=args.dataset_name,
            split_data_type=split_type,
            seed=random_seed,
            train_smiles=pair['canonical_smiles'],
            valid_smiles=pairv['canonical_smiles'],
            test_smiles=pairt['canonical_smiles'],
            source=split_lock_source,
            extra_meta={
                "numpy_version": np.__version__,
            },
        )
        save_split_lock(write_split_lock_path, split_lock_payload_to_write)
        split_lock_written_path = os.path.abspath(write_split_lock_path)
        output_files['split_lock_json'] = split_lock_written_path
        print(f'Wrote split lock to {split_lock_written_path}')
    # print(pair['target'].shape)
    print('===============', split_type, '================')
    print('train', len(pair['canonical_smiles']), len(set(pair['canonical_smiles'])), )
    print('valid', len(pairv['canonical_smiles']), len(set(pairv['canonical_smiles'])), )
    print('test', len(pairt['canonical_smiles']), len(set(pairt['canonical_smiles'])), )
    intended_overlap_summary = log_smiles_overlap(
        prefix='Intended split',
        train_smiles=pair['canonical_smiles'],
        valid_smiles=pairv['canonical_smiles'],
        test_smiles=pairt['canonical_smiles'],
        enforce_zero=(split_type == 'smiles_split'),
    )
    dataset_sizes = {
        'train': int(len(pair['canonical_smiles'])),
        'valid': int(len(pairv['canonical_smiles'])),
        'test': int(len(pairt['canonical_smiles'])),
        'train_unique_smiles': int(len(set(pair['canonical_smiles']))),
        'valid_unique_smiles': int(len(set(pairv['canonical_smiles']))),
        'test_unique_smiles': int(len(set(pairt['canonical_smiles']))),
        'intended_smiles_overlap': intended_overlap_summary,
        'split_source': split_lock_source,
    }
    if split_lock_path:
        dataset_sizes['split_lock_path'] = os.path.abspath(split_lock_path)
    if split_lock_meta is not None:
        dataset_sizes['split_lock_meta'] = split_lock_meta
    if split_lock_written_path is not None:
        dataset_sizes['written_split_lock_path'] = split_lock_written_path
    for name, pair_data in zip(['train', 'valid', 'test'], [pair, pairv, pairt]):
        df = pd.DataFrame(pair_data['canonical_smiles'], columns=['canonical_smiles'])
        df.drop_duplicates(inplace=True)
        # df.to_csv(local_out + '{}_{}_data_canonical_smiles.csv'.format(split_type, name), index=False)

    if use_paired_data:
        if DynamicMultiModalDataset is None:
            raise FileNotFoundError(
                "DynamicMultiModalDataset is unavailable. Set AIDD_PAIRED_DATASET_PY "
                "to the paired-data dataset.py path, or run with --use_paired_data False."
            )
        # 使用 DynamicMultiModalDataset
        # 直接从 HDF5 文件中的 split_id 进行分割，而不是基于 SMILES
        # 根据 split_type 和 n_folds 来确定 split_indices
        # 对于 5 折交叉验证：训练集 [0,1,2]，验证集 [3]，测试集 [4]
        if paired_historical_split:
            train_split_indices = historical_train_split_indices
            valid_split_indices = historical_valid_split_indices
            test_split_indices = historical_test_split_indices
        elif split_type in ['random_split', 'smiles_split']:
            # 5 折交叉验证：前3折训练，第4折验证，第5折测试
            train_split_indices = [0, 1, 2]
            valid_split_indices = [3]
            test_split_indices = [4]
        elif split_type == 'cells_split':
            # 对于 cells_split，需要根据实际的 split_id 分布来确定
            # 这里暂时使用相同的逻辑
            train_split_indices = [0, 1, 2]
            valid_split_indices = [3]
            test_split_indices = [4]
        else:
            raise ValueError(f"不支持的 split_type: {split_type}")
        
        print(f'Train split_indices: {train_split_indices} (对应前3折)')
        print(f'Valid split_indices: {valid_split_indices} (对应第4折)')
        print(f'Test split_indices: {test_split_indices} (对应第5折)')
        
        # 确保 SMILES 格式一致（用于计算 perturbed_centroid）
        train_smiles = np.array([str(s).strip() for s in pair['canonical_smiles']])
        valid_smiles = np.array([str(s).strip() for s in pairv['canonical_smiles']])
        test_smiles = np.array([str(s).strip() for s in pairt['canonical_smiles']])
        
        # 计算 perturbed_centroid（从训练集中计算）
        # 需要从 HDF5 文件中读取训练集数据来计算
        with h5py.File(cp_h5_path, 'r') as f:
            h5_smiles = np.array([str(s).strip() for s in f['combined']['smiles'][:].astype(str)])
            train_mask = np.isin(h5_smiles, train_smiles)
            if np.any(train_mask):
                perturbed_centroid_CP = f['combined']['Target'][train_mask].mean(axis=0).astype(np.float32)
            else:
                perturbed_centroid_CP = np.zeros(f['combined']['Target'].shape[1], dtype=np.float32)
        
        with h5py.File(ge_h5_path, 'r') as f:
            h5_smiles = np.array([str(s).strip() for s in f['combined']['smiles'][:].astype(str)])
            train_mask = np.isin(h5_smiles, train_smiles)
            if np.any(train_mask):
                perturbed_centroid_GE = f['combined']['Target'][train_mask].mean(axis=0).astype(np.float32)
            else:
                perturbed_centroid_GE = np.zeros(f['combined']['Target'].shape[1], dtype=np.float32)
        
        # 设置编码器路径
        encoder_path_root = '<ARTIFACT_ROOT>/data/MVC/Molecule_encoder/'
        if not os.path.exists(encoder_path_root):
            # 尝试其他可能的路径
            encoder_path_root = '<PROJECT_ROOT>/baseline/data/MVC/Molecule_encoder/'
        
        # 创建 DynamicMultiModalDataset 并包装
        # 使用 preload_data=True 可以加速数据加载，但会占用更多内存
        # 如果内存不足，可以设置为 False
        train_dynamic = DynamicMultiModalDataset(
            ge_h5_path=ge_h5_path,
            cp_h5_path=cp_h5_path,
            split_indices=train_split_indices,
            mode='train',
            mol_feature_type=feat_type,
            encoder_path_root=encoder_path_root,
            preload_data=True  # 预加载数据到内存以加速
        )
        train_dynamic.perturbed_centroid_CP = perturbed_centroid_CP
        train_dynamic.perturbed_centroid_GE = perturbed_centroid_GE
        train = DynamicMultiModalDatasetWrapper(train_dynamic)
        
        # 验证/测试集共用训练集的归一化统计，保证尺度一致
        normalization_stats = getattr(train_dynamic, "normalization_stats", None)
        valid_dynamic = DynamicMultiModalDataset(
            ge_h5_path=ge_h5_path,
            cp_h5_path=cp_h5_path,
            split_indices=valid_split_indices,
            mode='val',
            mol_feature_type=feat_type,
            encoder_path_root=encoder_path_root,
            preload_data=True,  # 预加载数据到内存以加速
            normalization_stats=normalization_stats
        )
        valid_dynamic.perturbed_centroid_CP = perturbed_centroid_CP
        valid_dynamic.perturbed_centroid_GE = perturbed_centroid_GE
        valid = DynamicMultiModalDatasetWrapper(valid_dynamic)
        
        test_dynamic = DynamicMultiModalDataset(
            ge_h5_path=ge_h5_path,
            cp_h5_path=cp_h5_path,
            split_indices=test_split_indices,
            mode='test',
            mol_feature_type=feat_type,
            encoder_path_root=encoder_path_root,
            preload_data=True,  # 预加载数据到内存以加速
            normalization_stats=normalization_stats
        )
        test_dynamic.perturbed_centroid_CP = perturbed_centroid_CP
        test_dynamic.perturbed_centroid_GE = perturbed_centroid_GE
        test = DynamicMultiModalDatasetWrapper(test_dynamic)
        
        # 更新输入维度（从数据集中获取）
        input_images = train_dynamic.cp_dim
        input_genes = train_dynamic.gene_dim
        print(f'Updated input dimensions from dataset: images={input_images}, genes={input_genes}')
        
        # 更新 gene_emb_dim 以匹配实际数据维度（如果使用 Default 编码器）
        if Gene_encoder_type == 'Default':
            gene_emb_dim = input_genes
            print(f'Updated gene_emb_dim to match data: {gene_emb_dim}')
    else:
        # 使用原有的 MVC_Dataset
        # 新增
        train_mask = np.isin(data['canonical_smiles'], pair['canonical_smiles'])   # 按 id 找行号
        perturbed_centroid_CP = data['target_CP'][train_mask].mean(axis=0).astype(np.float32)
        perturbed_centroid_GE = data['target_GE'][train_mask].mean(axis=0).astype(np.float32)

        # print(len(pair['canonical_smiles'])) # 12205
        train = MVC_Dataset(
            **build_mvc_dataset_kwargs(
                args=args,
                mol_feature_type=feat_type,
                gene_encoder_type=Gene_encoder_type,
                mol_id=pair['canonical_smiles'],
                paired_data=pair,
                perturbed_centroid_cp=perturbed_centroid_CP,
                perturbed_centroid_ge=perturbed_centroid_GE,
                normalization_stats=None,
                is_train=True,
            )
        )
        normalization_stats = train.get_normalization_stats()

        valid = MVC_Dataset(
            **build_mvc_dataset_kwargs(
                args=args,
                mol_feature_type=feat_type,
                gene_encoder_type=Gene_encoder_type,
                mol_id=pairv['canonical_smiles'],
                paired_data=pairv,
                perturbed_centroid_cp=perturbed_centroid_CP,
                perturbed_centroid_ge=perturbed_centroid_GE,
                normalization_stats=normalization_stats,
                is_train=False,
            )
        )

        test = MVC_Dataset(
            **build_mvc_dataset_kwargs(
                args=args,
                mol_feature_type=feat_type,
                gene_encoder_type=Gene_encoder_type,
                mol_id=pairt['canonical_smiles'],
                paired_data=pairt,
                perturbed_centroid_cp=perturbed_centroid_CP,
                perturbed_centroid_ge=perturbed_centroid_GE,
                normalization_stats=normalization_stats,
                is_train=False,
            )
        )
        actual_overlap_summary = log_smiles_overlap(
            prefix='Actual dataset',
            train_smiles=train.Image_Gene_data['canonical_smiles'],
            valid_smiles=valid.Image_Gene_data['canonical_smiles'],
            test_smiles=test.Image_Gene_data['canonical_smiles'],
            enforce_zero=(split_type == 'smiles_split'),
        )
        dataset_sizes['actual_dataset_smiles_overlap'] = actual_overlap_summary
        if bool(getattr(args, "molecule_balanced_temp_mean", False)):
            train = MoleculeBalancedTempMeanDataset(
                base_dataset=train,
                samples_per_molecule=getattr(args, "molecule_balanced_samples_per_molecule", 1),
                random_seed=random_seed,
            )
            dataset_sizes['molecule_balanced_temp_mean'] = _to_jsonable(train.summary)
            dataset_sizes['train_effective'] = int(len(train))
            dataset_sizes['train_effective_unique_smiles'] = int(len(set(train.Image_Gene_data['canonical_smiles'])))
            print(
                "Using molecule-balanced temporary-mean dataset: "
                + json.dumps(_to_jsonable(train.summary), ensure_ascii=False)
            )

    # 对于使用 DynamicMultiModalDataset 的情况，HDF5 文件句柄在多进程环境下可能有问题
    # 如果使用 use_paired_data，建议使用 num_workers=0 或较小的值
    if use_paired_data:
        num_workers = 0  # HDF5 文件在多进程环境下可能阻塞，使用单进程
        pin_memory = True  # 使用 pin_memory 加速数据传输到 GPU
        prefetch_factor = 2  # 预取因子
    else:
        num_workers = int(getattr(args, "num_workers", 4))
        pin_memory = False
        prefetch_factor = 2
    train_sampler = None
    train_shuffle = True
    molecule_balanced_summary = {}
    if bool(getattr(args, "molecule_balanced_temp_mean", False)):
        train_sampler = None
        train_shuffle = True
    elif bool(getattr(args, "molecule_balanced_sampling", False)):
        train_sampler, molecule_balanced_summary = build_molecule_balanced_sampler(
            train,
            random_seed,
            samples_per_molecule=getattr(args, "molecule_balanced_samples_per_molecule", 1),
        )
        train_shuffle = False
        dataset_sizes["molecule_balanced_sampling"] = _to_jsonable(molecule_balanced_summary)
        print(
            "Using molecule-balanced sampler: "
            + json.dumps(_to_jsonable(molecule_balanced_summary), ensure_ascii=False)
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train, 
        batch_size=batch_size, 
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=False, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    # print('train/valid/test samples:', len(train), len(valid), len(test))
    resolved_config = {
        'device': str(dev),
        'molecule_feature_dim': int(molecule_feature_dim),
        'gene_embedding_dim': int(gene_emb_dim),
        'input_images': int(input_images),
        'input_genes': int(input_genes),
        'num_workers': int(num_workers),
        'pin_memory': bool(pin_memory),
        'prefetch_factor': int(prefetch_factor),
        'effective_learning_rate': float(learning_rate),
        'effective_weight_decay': float(weight_decay),
        'effective_n_latent': int(n_latent),
        'effective_dropout': float(dropout),
        'use_paired_data': bool(use_paired_data),
        'use_augmentation': bool(args.use_augmentation),
        'molecule_balanced_sampling': bool(args.molecule_balanced_sampling),
        'molecule_balanced_samples_per_molecule': int(args.molecule_balanced_samples_per_molecule),
        'molecule_balanced_temp_mean': bool(args.molecule_balanced_temp_mean),
        'use_dose_feature': bool(args.use_dose_feature),
        'dose_ordinal_weight': float(args.dose_ordinal_weight),
        'dose_ordinal_temperature': float(args.dose_ordinal_temperature),
        'dose_ordinal_margin_scale': float(args.dose_ordinal_margin_scale),
        'aug_ge_drop_prob': float(args.aug_ge_drop_prob),
        'aug_cp_drop_prob': float(args.aug_cp_drop_prob),
        'aug_mol_drop_prob': float(args.aug_mol_drop_prob),
        'aug_mol_noise_std': float(args.aug_mol_noise_std),
        'aug_noise_std': float(args.aug_noise_std),
        'aug_winsor_q_low': float(args.aug_winsor_q_low),
        'aug_winsor_q_high': float(args.aug_winsor_q_high),
        'aug_winsor_after_norm': bool(args.aug_winsor_after_norm),
        'train_sampler': (
            f'molecule_temp_mean_x{int(args.molecule_balanced_samples_per_molecule)}'
            if bool(getattr(args, "molecule_balanced_temp_mean", False)) else
            f'molecule_balanced_x{int(args.molecule_balanced_samples_per_molecule)}'
            if train_sampler is not None else 'shuffle'
        ),
        'split_source': split_lock_source,
        'split_lock_path': os.path.abspath(split_lock_path) if split_lock_path else None,
        'written_split_lock_path': split_lock_written_path,
    }
    
    ablation_arch_kwargs = {
        'n_genes': input_genes,
        'n_images': input_images,
        'n_emd': gene_emb_dim,
        'n_latent': n_latent,
        'n_en_hidden': [512],
        'n_de_hidden': [768],
        'molecule_feature_dim': molecule_feature_dim,
        'molecule_hidden': 512,
        'init_w': True,
        'beta': beta,
        'device': dev,
        'dropout': dropout,
        'random_seed': random_seed,
    }

    # 根据 model_type 选择 release 包中保留的模型。
    if args.model_type == 'MVC_HyperGate':
        hypergate_n_en_hidden = [512]
        hypergate_n_de_hidden = [768]
        hypergate_n_latent = n_latent
        hypergate_dropout = dropout
        hypergate_heads = args.n_attn_heads
        hypergate_layers = args.n_fusion_layers

        if args.dataset_name == 'BBBC036' and args.hypergate_small_data_mode:
            hypergate_n_en_hidden = [256]
            hypergate_n_de_hidden = [512]
            hypergate_n_latent = min(n_latent, 768)
            hypergate_dropout = max(dropout, 0.2)
            hypergate_heads = min(args.n_attn_heads, 4)
            hypergate_layers = 1
            if learning_rate == 1e-3:
                learning_rate = 3e-4
            print(
                "Enable hypergate_small_data_mode for BBBC036 -> "
                f"latent={hypergate_n_latent}, embed={hypergate_n_en_hidden[0]}, "
                f"dropout={hypergate_dropout}, heads={hypergate_heads}, layers={hypergate_layers}, "
                f"lr={learning_rate}, pcc_weight={args.pcc_weight}, delta_pcc_weight={args.delta_pcc_weight}"
            )

        model = MVCModel_HyperGate(
            n_genes=input_genes,
            n_images=input_images,
            n_emd=gene_emb_dim,
            n_latent=hypergate_n_latent,
            n_en_hidden=hypergate_n_en_hidden,
            n_de_hidden=hypergate_n_de_hidden,
            molecule_feature_dim=molecule_feature_dim,
            molecule_hidden=512,
            init_w=True,
            beta=beta,
            device=dev,
            dropout=hypergate_dropout,
            path_model=local_out,
            random_seed=random_seed,
            n_attn_heads=hypergate_heads,
            n_fusion_layers=hypergate_layers,
            disable_hyper_refinement=args.hypergate_disable_hyper_refinement,
            pcc_weight=args.pcc_weight,
            delta_pcc_weight=args.delta_pcc_weight,
            cp_pcc_weight=args.cp_pcc_weight,
            ge_pcc_weight=args.ge_pcc_weight,
            cp_delta_pcc_weight=args.cp_delta_pcc_weight,
            ge_delta_pcc_weight=args.ge_delta_pcc_weight,
            grad_clip_norm=args.grad_clip_norm,
            use_dose_feature=args.use_dose_feature,
            dose_ordinal_weight=args.dose_ordinal_weight,
            dose_ordinal_temperature=args.dose_ordinal_temperature,
            dose_ordinal_margin_scale=args.dose_ordinal_margin_scale,
        )
        resolved_config.update({
            'effective_learning_rate': float(learning_rate),
            'effective_dropout': float(hypergate_dropout),
            'effective_n_latent': int(hypergate_n_latent),
            'effective_n_attn_heads': int(hypergate_heads),
            'effective_n_fusion_layers': int(hypergate_layers),
            'disable_hyper_refinement': bool(args.hypergate_disable_hyper_refinement),
            'pcc_weight': float(args.pcc_weight),
            'delta_pcc_weight': float(args.delta_pcc_weight),
            'cp_pcc_weight': None if args.cp_pcc_weight is None else float(args.cp_pcc_weight),
            'ge_pcc_weight': None if args.ge_pcc_weight is None else float(args.ge_pcc_weight),
            'cp_delta_pcc_weight': None if args.cp_delta_pcc_weight is None else float(args.cp_delta_pcc_weight),
            'ge_delta_pcc_weight': None if args.ge_delta_pcc_weight is None else float(args.ge_delta_pcc_weight),
            'grad_clip_norm': float(args.grad_clip_norm),
            'dose_ordinal_weight': float(args.dose_ordinal_weight),
            'dose_ordinal_temperature': float(args.dose_ordinal_temperature),
            'dose_ordinal_margin_scale': float(args.dose_ordinal_margin_scale),
            'hypergate_encoder_hidden': list(hypergate_n_en_hidden),
            'hypergate_decoder_hidden': list(hypergate_n_de_hidden),
        })
        ablation_arch_kwargs.update({
            'n_latent': hypergate_n_latent,
            'n_en_hidden': list(hypergate_n_en_hidden),
            'n_de_hidden': list(hypergate_n_de_hidden),
            'dropout': hypergate_dropout,
        })
    elif args.model_type in {'MVC_HyperGateResidualVAE', 'MVC_HyperGateResidualVAE_NoCP', 'MVC_HyperGateResidualVAE_NoGE'}:
        hypergate_n_en_hidden = [512]
        hypergate_n_de_hidden = [768]
        hypergate_n_latent = n_latent
        hypergate_dropout = dropout
        hypergate_heads = args.n_attn_heads
        hypergate_layers = args.n_fusion_layers

        if args.dataset_name == 'BBBC036' and args.hypergate_small_data_mode:
            hypergate_n_en_hidden = [256]
            hypergate_n_de_hidden = [512]
            hypergate_n_latent = min(n_latent, 768)
            hypergate_dropout = max(dropout, 0.2)
            hypergate_heads = min(args.n_attn_heads, 4)
            hypergate_layers = 1
            if learning_rate == 1e-3:
                learning_rate = 3e-4

        residual_vae_model_cls = MVCModel_HyperGateResidualVAE
        if args.model_type == 'MVC_HyperGateResidualVAE_NoCP':
            residual_vae_model_cls = MVCModel_HyperGateResidualVAE_NoCP
        elif args.model_type == 'MVC_HyperGateResidualVAE_NoGE':
            residual_vae_model_cls = MVCModel_HyperGateResidualVAE_NoGE

        model = residual_vae_model_cls(
            n_genes=input_genes,
            n_images=input_images,
            n_emd=gene_emb_dim,
            n_latent=hypergate_n_latent,
            n_en_hidden=hypergate_n_en_hidden,
            n_de_hidden=hypergate_n_de_hidden,
            molecule_feature_dim=molecule_feature_dim,
            molecule_hidden=512,
            init_w=True,
            beta=beta,
            device=dev,
            dropout=hypergate_dropout,
            path_model=local_out,
            random_seed=random_seed,
            n_attn_heads=hypergate_heads,
            n_fusion_layers=hypergate_layers,
            disable_hyper_refinement=args.hypergate_disable_hyper_refinement,
            pcc_weight=args.pcc_weight,
            delta_pcc_weight=args.delta_pcc_weight,
            cp_pcc_weight=args.cp_pcc_weight,
            ge_pcc_weight=args.ge_pcc_weight,
            cp_delta_pcc_weight=args.cp_delta_pcc_weight,
            ge_delta_pcc_weight=args.ge_delta_pcc_weight,
            grad_clip_norm=args.grad_clip_norm,
            use_dose_feature=args.use_dose_feature,
            dose_ordinal_weight=args.dose_ordinal_weight,
            dose_ordinal_temperature=args.dose_ordinal_temperature,
            dose_ordinal_margin_scale=args.dose_ordinal_margin_scale,
            residual_vae_latent_dim=args.residual_vae_latent_dim,
            residual_vae_hidden_dim=args.residual_vae_hidden_dim,
            residual_vae_kl_weight=args.residual_vae_kl_weight,
            residual_vae_kl_warmup_epochs=args.residual_vae_kl_warmup_epochs,
            residual_vae_disentangle_weight=args.residual_vae_disentangle_weight,
            residual_vae_disentangle_mode=args.residual_vae_disentangle_mode,
            residual_vae_hsic_sigma=args.residual_vae_hsic_sigma,
            residual_vae_prior_recon_weight=args.residual_vae_prior_recon_weight,
            residual_vae_base_recon_weight=args.residual_vae_base_recon_weight,
            residual_vae_correction_scale=args.residual_vae_correction_scale,
            residual_vae_correction_l2_weight=args.residual_vae_correction_l2_weight,
            residual_vae_private_orth_weight=args.residual_vae_private_orth_weight,
            residual_vae_private_remainder_weight=args.residual_vae_private_remainder_weight,
            residual_vae_shared_infonce_weight=args.residual_vae_shared_infonce_weight,
            residual_vae_shared_infonce_temperature=args.residual_vae_shared_infonce_temperature,
            residual_vae_modal_mask_prob=args.residual_vae_modal_mask_prob,
            residual_vae_sample_train=args.residual_vae_sample_train,
            residual_vae_training_stage=args.residual_vae_training_stage,
            residual_vae_geview_weight=args.residual_vae_geview_weight,
            residual_vae_cpview_weight=args.residual_vae_cpview_weight,
            residual_vae_aux_view_mode=args.residual_vae_aux_view_mode,
        )
        resolved_config.update({
            'effective_learning_rate': float(learning_rate),
            'effective_dropout': float(hypergate_dropout),
            'effective_n_latent': int(hypergate_n_latent),
            'effective_n_attn_heads': int(hypergate_heads),
            'effective_n_fusion_layers': int(hypergate_layers),
            'disable_hyper_refinement': bool(args.hypergate_disable_hyper_refinement),
            'pcc_weight': float(args.pcc_weight),
            'delta_pcc_weight': float(args.delta_pcc_weight),
            'cp_pcc_weight': None if args.cp_pcc_weight is None else float(args.cp_pcc_weight),
            'ge_pcc_weight': None if args.ge_pcc_weight is None else float(args.ge_pcc_weight),
            'cp_delta_pcc_weight': None if args.cp_delta_pcc_weight is None else float(args.cp_delta_pcc_weight),
            'ge_delta_pcc_weight': None if args.ge_delta_pcc_weight is None else float(args.ge_delta_pcc_weight),
            'grad_clip_norm': float(args.grad_clip_norm),
            'dose_ordinal_weight': float(args.dose_ordinal_weight),
            'dose_ordinal_temperature': float(args.dose_ordinal_temperature),
            'dose_ordinal_margin_scale': float(args.dose_ordinal_margin_scale),
            'residual_vae_latent_dim': int(args.residual_vae_latent_dim),
            'residual_vae_hidden_dim': int(args.residual_vae_hidden_dim),
            'residual_vae_kl_weight': float(args.residual_vae_kl_weight),
            'residual_vae_kl_warmup_epochs': int(args.residual_vae_kl_warmup_epochs),
            'residual_vae_disentangle_weight': float(args.residual_vae_disentangle_weight),
            'residual_vae_disentangle_mode': str(args.residual_vae_disentangle_mode),
            'residual_vae_hsic_sigma': float(args.residual_vae_hsic_sigma),
            'residual_vae_prior_recon_weight': float(args.residual_vae_prior_recon_weight),
            'residual_vae_base_recon_weight': float(args.residual_vae_base_recon_weight),
            'residual_vae_correction_scale': float(args.residual_vae_correction_scale),
            'residual_vae_correction_l2_weight': float(args.residual_vae_correction_l2_weight),
            'residual_vae_private_orth_weight': float(args.residual_vae_private_orth_weight),
            'residual_vae_private_remainder_weight': float(args.residual_vae_private_remainder_weight),
            'residual_vae_shared_infonce_weight': float(args.residual_vae_shared_infonce_weight),
            'residual_vae_shared_infonce_temperature': float(args.residual_vae_shared_infonce_temperature),
            'residual_vae_modal_mask_prob': float(args.residual_vae_modal_mask_prob),
            'residual_vae_sample_train': bool(args.residual_vae_sample_train),
            'residual_vae_training_stage': str(args.residual_vae_training_stage),
            'residual_vae_geview_weight': float(args.residual_vae_geview_weight),
            'residual_vae_cpview_weight': float(args.residual_vae_cpview_weight),
            'residual_vae_aux_view_mode': str(args.residual_vae_aux_view_mode),
            'hypergate_encoder_hidden': list(hypergate_n_en_hidden),
            'hypergate_decoder_hidden': list(hypergate_n_de_hidden),
        })
        if args.model_type == 'MVC_HyperGateResidualVAE_NoCP':
            resolved_config['single_target_branch'] = 'GE-only target'
        elif args.model_type == 'MVC_HyperGateResidualVAE_NoGE':
            resolved_config['single_target_branch'] = 'CP-only target'
        ablation_arch_kwargs.update({
            'n_latent': hypergate_n_latent,
            'n_en_hidden': list(hypergate_n_en_hidden),
            'n_de_hidden': list(hypergate_n_de_hidden),
            'dropout': hypergate_dropout,
        })
    elif args.model_type == 'MVC_GatedPCC':
        model = MVCModel_GatedPCC(
            n_genes=input_genes,
            n_images=input_images,
            n_emd=gene_emb_dim,
            n_latent=n_latent,
            n_en_hidden=[512],
            n_de_hidden=[768],
            molecule_feature_dim=molecule_feature_dim,
            molecule_hidden=512,
            init_w=True,
            beta=beta,
            device=dev,
            dropout=dropout,
            path_model=local_out,
            random_seed=random_seed,
            pcc_weight=args.pcc_weight,
            delta_pcc_weight=args.delta_pcc_weight,
            mse_cp_weight=args.mse_cp_weight,
            mse_ge_weight=args.mse_ge_weight,
            use_dose_feature=args.use_dose_feature,
            dose_ordinal_weight=args.dose_ordinal_weight,
            dose_ordinal_temperature=args.dose_ordinal_temperature,
            dose_ordinal_margin_scale=args.dose_ordinal_margin_scale,
        )
        resolved_config.update({
            'pcc_weight': float(args.pcc_weight),
            'delta_pcc_weight': float(args.delta_pcc_weight),
            'mse_cp_weight': float(args.mse_cp_weight),
            'mse_ge_weight': float(args.mse_ge_weight),
            'dose_ordinal_weight': float(args.dose_ordinal_weight),
            'dose_ordinal_temperature': float(args.dose_ordinal_temperature),
            'dose_ordinal_margin_scale': float(args.dose_ordinal_margin_scale),
        })
    elif args.model_type == 'MVC':
        model = MVCModel(
            n_genes=input_genes,
            n_images=input_images,
            n_emd=gene_emb_dim,
            n_latent=n_latent,
            n_en_hidden=[512],
            n_de_hidden=[768],
            molecule_feature_dim=molecule_feature_dim,
            molecule_hidden=512,
            init_w=True,
            beta=beta,
            device=dev,
            dropout=dropout,
            path_model=local_out,
            random_seed=random_seed,
            use_dose_feature=args.use_dose_feature,
            dose_ordinal_weight=args.dose_ordinal_weight,
            dose_ordinal_temperature=args.dose_ordinal_temperature,
            dose_ordinal_margin_scale=args.dose_ordinal_margin_scale,
        )
    else:
        raise ValueError(f"Unsupported model_type in lightweight release: {args.model_type}")

    pretrained_checkpoint_path = str(getattr(args, "pretrained_checkpoint_path", "") or "").strip()
    if pretrained_checkpoint_path:
        pretrained_init_summary = load_partial_checkpoint_into_model(model, pretrained_checkpoint_path)
        resolved_config["pretrained_init"] = pretrained_init_summary
        print(
            "Loaded partial checkpoint init -> "
            f"tensors {pretrained_init_summary['loaded_tensor_count']}/{pretrained_init_summary['total_tensor_count']}, "
            f"parameters {pretrained_init_summary['loaded_parameter_count']}/{pretrained_init_summary['total_parameter_count']}, "
            f"shape_mismatch={pretrained_init_summary['shape_mismatch_count']}, "
            f"missing_in_model={pretrained_init_summary['missing_in_model_count']}"
        )
    freeze_summary = freeze_model_modules_by_prefix(model, getattr(args, "freeze_module_prefixes", []))
    if freeze_summary["requested_prefixes"]:
        resolved_config["freeze_modules"] = freeze_summary
        print(
            "Applied module freezing -> "
            f"prefixes={freeze_summary['requested_prefixes']}, "
            f"matched_tensors={freeze_summary['matched_parameter_count']}, "
            f"trainable_params={freeze_summary['trainable_parameter_count']}/"
            f"{freeze_summary['total_parameter_count']}"
        )
    if train_flag:
        model.to(dev)
        # load model
        # model.load_state_dict(torch.load(local_out + 'best_model.pt'))
        
        epoch_hist, best_epoch = model.train_model(train_loader=train_loader, test_loader=valid_loader,
                                                   n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay, save_model=True)

        epoch_result = pd.DataFrame.from_dict(epoch_hist)
        epoch_result['epoch'] = np.arange(n_epochs)
        epoch_result_path = os.path.join(local_out, f'epoch{n_epochs}_lr{learning_rate}.csv')
        epoch_result.to_csv(epoch_result_path, index=False)
        output_files['epoch_history_csv'] = epoch_result_path

        finite_check_col = resolve_validation_loss_column(epoch_result, model_type=args.model_type)
        assert_finite_validation_history(epoch_result, finite_check_col, local_out)

        selection_col, selection_mode = resolve_validation_selection(epoch_result, args)
        if selection_mode == "max":
            epoch = epoch_result[selection_col].idxmax()
        else:
            epoch = epoch_result[selection_col].idxmin()
        if epoch == best_epoch:
            print('best valid epoch:', epoch)
        else:
            print('warning: inconsistent best valid')

    # load the best model
    print(f'===============Load best model from {local_out}===============')
    filename = local_out + 'best_model.pt'
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing checkpoint: {filename}. Training likely diverged before best_model.pt was saved."
        )
    model = torch.load(filename, map_location='cpu')
    model.dev = torch.device(dev)
    model.to(dev)
    eval_output_suffix = ""

    save_dir = local_out + 'predict'
    isExists = os.path.exists(save_dir)
    print(save_dir)
    if not isExists:
        os.makedirs(save_dir)
        print('Directory created successfully')
    else:
        print('Directory already exists')

    if eval_metric:
        print('===============Evaluate model performance==============')
        setup_seed(random_seed)
        _, _, test_metrics_dict_ls = model.test_model(loader=test_loader, metrics_func=['pearson', 'rmse', 'precision100'],\
            perturbed_centroid_CP=perturbed_centroid_CP, perturbed_centroid_GE=perturbed_centroid_GE)

        for name, rec_dict_value in zip(['test'], [test_metrics_dict_ls]):
            df_rec = pd.DataFrame.from_dict(rec_dict_value)
            smi_ls = []
            # print(df_rec)
            for smi_id in df_rec['cp_id']:
                smi_ls.append(smi_id)
            df_rec['canonical_smiles'] = smi_ls
            per_sample_metrics_path = save_dir + f'/{name}_restruction_result_all_samples{eval_output_suffix}.csv'
            df_rec.to_csv(per_sample_metrics_path, index=False)
            output_files['per_sample_metrics_csv'] = per_sample_metrics_path

            metrics_mean_path = save_dir + f'/{name}_restruction_result_mean{eval_output_suffix}.csv'
            test_metrics_mean = save_metrics_mean(df_rec, metrics_mean_path)
            output_files['metrics_mean_csv'] = metrics_mean_path

    if predict_profile:
        print('===============Predict profile==============')
        for name, loader in zip(['test'], [test_loader]):
            (
                control_cp_array,
                control_ge_array,
                target_cp_array,
                target_ge_array,
                cp_pred_array,
                ge_pred_array,
                smiles_array,
            ) = unpack_prediction_profile_output(model.predict_profile(loader=loader))

            ddict_data = dict()
            ddict_data['control_cp'] = control_cp_array
            ddict_data['control_ge'] = control_ge_array
            ddict_data['target_cp'] = target_cp_array
            ddict_data['target_ge'] = target_ge_array
            ddict_data['cp_pred'] = cp_pred_array
            ddict_data['ge_pred'] = ge_pred_array
            ddict_data['smiles'] = np.array(smiles_array, dtype='S')
            
            for k in ddict_data.keys():
                print(k, type(ddict_data[k][0]), ddict_data[k].shape)
            prediction_profile_path = save_dir + f'/{name}_prediction_profile{eval_output_suffix}.h5'
            save_to_HDF(prediction_profile_path, ddict_data)
            output_files['prediction_profile_h5'] = prediction_profile_path

            if hasattr(model, "predict_all_views_and_latents"):
                ps_profiles, ps_latents = model.predict_all_views_and_latents(loader=loader)
                ps_profile_path = save_dir + f'/{name}_ps_all_views_prediction_profile{eval_output_suffix}.h5'
                save_to_HDF(ps_profile_path, ps_profiles)
                output_files['ps_all_views_prediction_profile_h5'] = ps_profile_path

                ps_latent_path = save_dir + f'/{name}_ps_latents{eval_output_suffix}.npz'
                np.savez_compressed(ps_latent_path, **ps_latents)
                output_files['ps_latents_npz'] = ps_latent_path

    def run_fair_masked_ablation(ablation_model_cls, label, description, extra_model_kwargs=None):
        extra_model_kwargs = extra_model_kwargs or {}
        print(f'训练公平消融模型：{ablation_model_cls.__name__} ({description})')
        model_prefix = local_out + f'{label}_'
        ablation_model = ablation_model_cls(
            n_genes=ablation_arch_kwargs['n_genes'],
            n_images=ablation_arch_kwargs['n_images'],
            n_emd=ablation_arch_kwargs['n_emd'],
            n_latent=ablation_arch_kwargs['n_latent'],
            n_en_hidden=ablation_arch_kwargs['n_en_hidden'],
            n_de_hidden=ablation_arch_kwargs['n_de_hidden'],
            molecule_feature_dim=ablation_arch_kwargs['molecule_feature_dim'],
            molecule_hidden=ablation_arch_kwargs['molecule_hidden'],
            init_w=ablation_arch_kwargs['init_w'],
            beta=ablation_arch_kwargs['beta'],
            device=ablation_arch_kwargs['device'],
            dropout=ablation_arch_kwargs['dropout'],
            path_model=model_prefix,
            random_seed=ablation_arch_kwargs['random_seed'],
            **extra_model_kwargs,
        )

        if train_flag:
            ablation_model.to(dev)
            epoch_hist, best_epoch = ablation_model.train_model(
                train_loader=train_loader,
                test_loader=valid_loader,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                save_model=True,
            )

            epoch_result = pd.DataFrame.from_dict(epoch_hist)
            epoch_result['epoch'] = np.arange(n_epochs)
            epoch_result.to_csv(local_out + f'epoch{n_epochs}_lr{learning_rate}_{label}.csv', index=False)

            print(f'{label}模型最佳验证epoch:', best_epoch)

        filename = model_prefix + 'best_model.pt'
        ablation_model = torch.load(filename, map_location='cpu')
        ablation_model.dev = torch.device(dev)
        ablation_model.to(dev)

        if eval_metric:
            print(f'评估{label}模型性能')
            setup_seed(random_seed)
            _, _, test_metrics_dict_ls = ablation_model.test_model(
                loader=test_loader,
                metrics_func=['pearson', 'rmse', 'precision100'],
                perturbed_centroid_CP=perturbed_centroid_CP,
                perturbed_centroid_GE=perturbed_centroid_GE,
            )

            df_rec = pd.DataFrame.from_dict(test_metrics_dict_ls)
            smi_ls = []
            for smi_id in df_rec['cp_id']:
                smi_ls.append(smi_id)
            df_rec['canonical_smiles'] = smi_ls
            df_rec.to_csv(save_dir + f'/test_restruction_result_all_samples_{label}.csv', index=False)
            save_metrics_mean(
                df_rec,
                save_dir + f'/test_restruction_result_mean_{label}.csv',
                log_label=f'{label} mean test metrics',
            )

        if predict_profile:
            print(f'预测{label}模型profile')
            (
                control_cp_array,
                control_ge_array,
                target_cp_array,
                target_ge_array,
                cp_pred_array,
                ge_pred_array,
                smiles_array,
            ) = unpack_prediction_profile_output(ablation_model.predict_profile(loader=test_loader))
            ddict_data = dict()
            ddict_data['control_cp'] = control_cp_array
            ddict_data['control_ge'] = control_ge_array
            ddict_data['target_cp'] = target_cp_array
            ddict_data['target_ge'] = target_ge_array
            ddict_data['cp_pred'] = cp_pred_array
            ddict_data['ge_pred'] = ge_pred_array
            ddict_data['smiles'] = np.array(smiles_array, dtype='S')

            for k in ddict_data.keys():
                print(label + '模型', k, 'shape:', ddict_data[k].shape)
            save_to_HDF(save_dir + f'/test_prediction_profile_{label}.h5', ddict_data)

        log_cuda_memory(f'Before {label} cleanup', device=dev)
        release_cuda_resources(ablation_model)
        ablation_model = None
        log_cuda_memory(f'After {label} cleanup', device=dev)

    # 消融实验
    if args.run_ablation and args.model_type in {'MVC', 'MVC_GatedPCC', 'MVC_HyperGate', 'MVC_HyperGateResidualVAE'}:
        print('===============开始消融实验==============')
        log_cuda_memory('Before ablation cleanup', device=dev)
        release_cuda_resources(model)
        model = None
        log_cuda_memory('After ablation cleanup', device=dev)

        ablation_extra_kwargs = {}
        if args.model_type == 'MVC':
            ablation_specs = [
                (MVCModel_MaskCPInput, 'MaskCPInput', 'CP输入置零，保留CP+GE双头预测'),
                (MVCModel_MaskGEInput, 'MaskGEInput', 'GE输入置零，保留CP+GE双头预测'),
            ]
        elif args.model_type == 'MVC_GatedPCC':
            ablation_specs = [
                (MVCModel_GatedPCC_MaskCPInput, 'MaskCPInput', 'CP输入置零，保留CP+GE双头预测'),
                (MVCModel_GatedPCC_MaskGEInput, 'MaskGEInput', 'GE输入置零，保留CP+GE双头预测'),
            ]
            ablation_extra_kwargs = {
                'pcc_weight': args.pcc_weight,
                'delta_pcc_weight': args.delta_pcc_weight,
                'mse_cp_weight': args.mse_cp_weight,
                'mse_ge_weight': args.mse_ge_weight,
            }
        elif args.model_type == 'MVC_HyperGate':
            ablation_specs = [
                (MVCModel_HyperGate_MaskCPInput, 'MaskCPInput', 'CP输入置零，保留HyperGate双头预测'),
                (MVCModel_HyperGate_MaskGEInput, 'MaskGEInput', 'GE输入置零，保留HyperGate双头预测'),
            ]
            ablation_extra_kwargs = {
                'n_attn_heads': resolved_config.get('effective_n_attn_heads', args.n_attn_heads),
                'n_fusion_layers': resolved_config.get('effective_n_fusion_layers', args.n_fusion_layers),
                'pcc_weight': args.pcc_weight,
                'delta_pcc_weight': args.delta_pcc_weight,
                'cp_pcc_weight': args.cp_pcc_weight,
                'ge_pcc_weight': args.ge_pcc_weight,
                'cp_delta_pcc_weight': args.cp_delta_pcc_weight,
                'ge_delta_pcc_weight': args.ge_delta_pcc_weight,
                'grad_clip_norm': args.grad_clip_norm,
                'disable_hyper_refinement': args.hypergate_disable_hyper_refinement,
            }
        else:
            ablation_specs = [
                (MVCModel_HyperGateResidualVAE_MaskCPInput, 'MaskCPInput', 'CP input zeroed, same dual-head MVCPert task'),
                (MVCModel_HyperGateResidualVAE_MaskGEInput, 'MaskGEInput', 'GE input zeroed, same dual-head MVCPert task'),
                (MVCModel_HyperGateResidualVAE_RandomCPInput, 'RandomCPInput', 'CP input batch-permuted, same dual-head MVCPert task'),
                (MVCModel_HyperGateResidualVAE_RandomGEInput, 'RandomGEInput', 'GE input batch-permuted, same dual-head MVCPert task'),
            ]
            ablation_extra_kwargs = {
                'n_attn_heads': resolved_config.get('effective_n_attn_heads', args.n_attn_heads),
                'n_fusion_layers': resolved_config.get('effective_n_fusion_layers', args.n_fusion_layers),
                'disable_hyper_refinement': args.hypergate_disable_hyper_refinement,
                'pcc_weight': args.pcc_weight,
                'delta_pcc_weight': args.delta_pcc_weight,
                'cp_pcc_weight': args.cp_pcc_weight,
                'ge_pcc_weight': args.ge_pcc_weight,
                'cp_delta_pcc_weight': args.cp_delta_pcc_weight,
                'ge_delta_pcc_weight': args.ge_delta_pcc_weight,
                'grad_clip_norm': args.grad_clip_norm,
                'use_dose_feature': args.use_dose_feature,
                'dose_ordinal_weight': args.dose_ordinal_weight,
                'dose_ordinal_temperature': args.dose_ordinal_temperature,
                'dose_ordinal_margin_scale': args.dose_ordinal_margin_scale,
                'residual_vae_latent_dim': args.residual_vae_latent_dim,
                'residual_vae_hidden_dim': args.residual_vae_hidden_dim,
                'residual_vae_kl_weight': args.residual_vae_kl_weight,
                'residual_vae_kl_warmup_epochs': args.residual_vae_kl_warmup_epochs,
                'residual_vae_disentangle_weight': args.residual_vae_disentangle_weight,
                'residual_vae_disentangle_mode': args.residual_vae_disentangle_mode,
                'residual_vae_hsic_sigma': args.residual_vae_hsic_sigma,
                'residual_vae_prior_recon_weight': args.residual_vae_prior_recon_weight,
                'residual_vae_base_recon_weight': args.residual_vae_base_recon_weight,
                'residual_vae_correction_scale': args.residual_vae_correction_scale,
                'residual_vae_correction_l2_weight': args.residual_vae_correction_l2_weight,
                'residual_vae_private_orth_weight': args.residual_vae_private_orth_weight,
                'residual_vae_private_remainder_weight': args.residual_vae_private_remainder_weight,
                'residual_vae_shared_infonce_weight': args.residual_vae_shared_infonce_weight,
                'residual_vae_shared_infonce_temperature': args.residual_vae_shared_infonce_temperature,
                'residual_vae_modal_mask_prob': args.residual_vae_modal_mask_prob,
                'residual_vae_sample_train': args.residual_vae_sample_train,
                'residual_vae_training_stage': args.residual_vae_training_stage,
                'residual_vae_geview_weight': args.residual_vae_geview_weight,
                'residual_vae_cpview_weight': args.residual_vae_cpview_weight,
                'residual_vae_aux_view_mode': args.residual_vae_aux_view_mode,
            }

        for ablation_model_cls, label, description in ablation_specs:
            run_fair_masked_ablation(
                ablation_model_cls=ablation_model_cls,
                label=label,
                description=description,
                extra_model_kwargs=ablation_extra_kwargs,
            )
        
        print('===============消融实验完成==============')
    elif args.run_ablation:
        print(f'No ablation recipe is defined for {args.model_type} in this release.')

    run_summary_path = os.path.join(local_out, 'run_summary.json')
    output_files['run_summary_json'] = run_summary_path
    run_summary = build_run_summary(
        args=args,
        output_dir=local_out,
        predict_dir=save_dir,
        train_log_path=train_log_path,
        best_epoch=best_epoch,
        test_metrics_mean=test_metrics_mean,
        run_timestamp=run_timestamp,
        output_files=output_files,
        resolved_config=resolved_config,
        dataset_sizes=dataset_sizes,
    )
    save_json(run_summary_path, run_summary)
    print('Saved run summary to', run_summary_path)
    return run_summary

if __name__ == "__main__":
    args = parse_args()
    train_MVCModel(args)
