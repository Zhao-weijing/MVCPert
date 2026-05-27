from torch.utils.data import Dataset
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from utils import *

DATA_ROOT = os.environ.get("MVCPERT_DATA_ROOT", "<DATA_ROOT>")
ARTIFACT_ROOT = os.environ.get("MVCPERT_ARTIFACT_ROOT", "<ARTIFACT_ROOT>")
PROJECT_ROOT = os.environ.get("MVCPERT_PROJECT_ROOT", "<PROJECT_ROOT>")

# 定义分子特征类型与对应嵌入文件的映射配置
EMBEDDING_CONFIG = {
    'ECFP4': 'ECFP4_emb2048.pickle',
    'KPGT': 'KPGT_emb2304.pickle',
    'InfoAlign': 'InfoAlign_emb300.pickle',
    'ChemBERTa2': 'ChemBERTa2_emb384.pickle',
    # 'MolT5': 'molt5_emb768.pickle',
    'MolT5': 'MolT5_emb768.pickle',
    'Chemprop': 'Chemprop_emb300.pickle',
    'MolCLR': 'MolCLR_emb512.pickle',
    'Mole_BERT': 'Mole_BERT_emb300.pickle',
    'GeminiMol': 'GeminiMol_emb2048.pkl',
    'Ouroboros': 'Ouroboros_emb2048.pkl',
    'UniMol': 'UniMol_emb512.pkl',
    'UniMolV2': 'UniMolV2_emb1024.pkl'
}

EMBEDDING_PATH_CANDIDATES = (
    f'{PROJECT_ROOT}/baseline/data/MVC/Molecule_encoder',
    f'{ARTIFACT_ROOT}/data/MVC/Molecule_encoder',
)

# 定义基因编码器类型与对应HDF文件的映射
GENE_ENCODER_MAP = {
    'Default': 'default.h5',
    'CellPLM': 'CellPLM.h5',
    'geneformer': 'geneformer.h5',
    'Openbiomed': 'Openbiomed.h5',
    'scBERT': 'scBERT.h5',
    'scFoundation': 'scFoundation.h5',
    'scGPT': 'scGPT.h5',
    'tGPT': 'tGPT.h5',
    'UCE': 'UCE.h5'
}

# ===== 数据增强辅助函数 (面向 BBBC047 的重尾/偏态分布做了更稳健的设计) =====
def apply_noise(data, noise_std=0.01, dist='student_t', df=3):
    """给输入加噪声（仅用于 input-side augmentation）。

    参数建议（BBBC047 归一化后）：
      - dist='student_t', df=3：更贴近重尾但不会像高斯那样过度“平滑”尾部；
      - noise_std=0.005~0.05：按特征 z-score 后的尺度取值。
    """
    if noise_std <= 0:
        return data
    if dist == 'gaussian':
        eps = np.random.normal(0.0, noise_std, size=data.shape)
    elif dist == 'laplace':
        eps = np.random.laplace(0.0, noise_std / np.sqrt(2), size=data.shape)
    elif dist == 'student_t':
        eps = np.random.standard_t(df, size=data.shape) * noise_std
    else:
        raise ValueError(f"Unknown noise dist: {dist}")
    return data + eps

def apply_feature_dropout(data, drop_prob=0.05, keep_zeros=True):
    """随机 Feature Dropout（Masking），模拟特征缺失或测量失败。
    
    对于高维生物数据（GE/CP），Masking 是比 Mixup 更稳健的正则化手段，
    因为它强制模型利用特征间的共线性（co-linearity）来恢复信息。
    """
    if drop_prob <= 0:
        return data
    if keep_zeros:
        nonzero = (data != 0)
        mask = np.ones_like(data, dtype=np.float32)
        rnd = np.random.rand(*data.shape)
        mask[(rnd < drop_prob) & nonzero] = 0.0
        return data * mask
    else:
        mask = np.random.binomial(1, 1 - drop_prob, size=data.shape).astype(np.float32)
        return data * mask

def winsorize(data, lower=None, upper=None):
    """稳健裁剪（Winsorization）：把极端值限制在 [lower, upper] 里。"""
    if lower is None or upper is None:
        return data
    return np.clip(data, lower, upper)

def _robust_bounds(X, q_low=0.001, q_high=0.999):
    """按 feature 计算分位数边界。X shape: (N, D)."""
    lo = np.quantile(X, q_low, axis=0)
    hi = np.quantile(X, q_high, axis=0)
    same = (hi - lo) < 1e-12
    if np.any(same):
        hi[same] = lo[same] + 1e-6
    return lo.astype(np.float32), hi.astype(np.float32)

# 兼容旧接口：保留原函数名（内部转到更稳健的实现）
def apply_gaussian_noise(data, noise_level=0.01):
    return apply_noise(data, noise_std=noise_level, dist='gaussian')

def apply_gene_masking(data, mask_prob=0.05):
    return apply_feature_dropout(data, drop_prob=mask_prob, keep_zeros=True)


def load_molecule_embedding_lookup(mol_feature_type):
    embedding_file = EMBEDDING_CONFIG.get(mol_feature_type)
    if embedding_file is None:
        raise ValueError(f"不支持的分子特征类型: {mol_feature_type}")

    candidate_paths = [f"{root}/{embedding_file}" for root in EMBEDDING_PATH_CANDIDATES]
    for embedding_path in candidate_paths:
        if os.path.exists(embedding_path):
            with open(embedding_path, 'rb') as f:
                return pickle.load(f)

    if mol_feature_type == 'ECFP4':
        return {}

    raise FileNotFoundError(f"未找到分子 embedding 文件: {candidate_paths}")


def compute_rdkit_ecfp4_embedding(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise KeyError(f"无法从 SMILES 生成 ECFP4 指纹: {smiles}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class Gene_Dataset(Dataset):
    def __init__(self, dataset_name, mol_feature_type, gene_encoder_type, mol_id, perturbed_centroid=None, augment=False):

        # self.LINCS_index = LINCS_indexooooooo
        self.mol_feature_type = mol_feature_type
        self.gene_encoder_type = gene_encoder_type
        self.mol_id = mol_id
        self.dataset_name = dataset_name
        self.dataset_name_part = self.dataset_name.split('_')[0]
        self.augment = augment # 是否启用数据增强
        
        # 加载smiles嵌入
        embedding_file = EMBEDDING_CONFIG.get(self.mol_feature_type)
        if embedding_file:
            embedding_path = f'{ARTIFACT_ROOT}/data/MVC/Molecule_encoder/{embedding_file}'
            # embedding_path = f'../Molecule_encoder/embeddings/{self.dataset_name_part}/{embedding_file}'
            with open(embedding_path, 'rb') as f:
                self.smi2emb = pickle.load(f)
        else:
            raise ValueError(f"不支持的分子特征类型: {self.mol_feature_type}")
        
        # 加载基因编码器数据
        encoder_file = GENE_ENCODER_MAP.get(self.gene_encoder_type)
        if encoder_file:
            if self.dataset_name.startswith('Tahoe_'):
                part = self.dataset_name.split('_')[1]  # 例如从'Tahoe_P2'中提取'P2'
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/Tahoe-100M/Tahoe_mini/{part}/{encoder_file}')
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/Tahoe-100M/Tahoe_mini/{part}/default.h5')
            
            elif self.dataset_name == 'BBBC036':
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/MVC/CDRPBIO-BBBC036-Bray/Processed_Paired_GE.h5')
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/MVC/CDRPBIO-BBBC036-Bray/Processed_Paired_GE.h5')
            elif self.dataset_name == 'BBBC047':
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/MVC/CDRP-BBBC047-Bray/Processed_Paired_GE.h5')
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/MVC/CDRP-BBBC047-Bray/Processed_Paired_GE.h5')
                
            elif self.dataset_name == 'LINCS':
                self.X1_data = load_from_HDF(f'../Gene_encoder/{self.dataset_name}/{encoder_file}')
                self.X2_data = load_from_HDF(f'../Gene_encoder/LINCS/default.h5')          
            elif self.dataset_name == 'LINCS965':
                self.X1_data = load_from_HDF(f'../Gene_encoder/{self.dataset_name}/{encoder_file}')
                self.X2_data = load_from_HDF(f'../Gene_encoder/LINCS965/default.h5')
            elif self.dataset_name == 'CIGS':
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/CIGS/Gene_Encoder/{encoder_file}')
                # print(self.X1_data)
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/CIGS/CIGS_2Cell_lines.h5')
        else:
            raise ValueError(f"不支持的基因编码器类型: {self.gene_encoder_type}")
        
        # ===== 2. 计算 Systema 用的 perturbed centroid =====
        # 只拿“扰动后”样本做中心（即 X2_data['target']）
        # 只用训练集里面的扰动中心
        self.perturbed_centroid = perturbed_centroid

    def __getitem__(self, index):
        # print(index, self.mol_id[index])
        mol_feature = self.smi2emb[str(self.mol_id[index])].astype(np.float32)
        
        control_gene = self.X1_data['control'][index]
        target_gene = self.X1_data['target'][index]
        
        # --- 数据增强 ---
        if self.augment:
            # 降低增强强度
            control_gene = apply_gaussian_noise(control_gene, noise_level=0.01)
            control_gene = apply_gene_masking(control_gene, mask_prob=0.05)
            # 移除分子特征增强，避免破坏Embedding语义
            # mol_feature = apply_gaussian_noise(mol_feature, noise_level=0.01)
            
        control_gene = control_gene.astype(np.float32)
        mol_feature = mol_feature.astype(np.float32)
        
        # x1表示基因编码器数据， x2表示原始的基因数据
        return control_gene, target_gene, self.X2_data['control'][index],\
            self.X2_data['target'][index], mol_feature, self.mol_id[index]

    # ===== 3. 给外部调用者暴露接口 =====
    def get_perturbed_centroid(self):
        return self.perturbed_centroid
    
    def __len__(self):
        return self.mol_id.shape[0]


class Gene_Cellline_Dataset(Dataset):
    def __init__(self, dataset_name, mol_feature_type, gene_encoder_type, mol_id, cid, perturbed_centroid=None, augment=False):

        # self.LINCS_index = LINCS_indexooooooo
        self.mol_feature_type = mol_feature_type
        self.gene_encoder_type = gene_encoder_type
        self.mol_id = mol_id
        self.dataset_name = dataset_name
        self.cid = cid
        self.dataset_name_part = self.dataset_name.split('_')[0]
        self.augment = augment
        
        # 加载smiles嵌入
        embedding_file = EMBEDDING_CONFIG.get(self.mol_feature_type)
        if embedding_file:
            embedding_path = f'{ARTIFACT_ROOT}/Molecule_encoder/embeddings/{self.dataset_name_part}/{embedding_file}'
            with open(embedding_path, 'rb') as f:
                self.smi2emb = pickle.load(f)
        else:
            raise ValueError(f"不支持的分子特征类型: {self.mol_feature_type}")
        
        # 加载基因编码器数据
        encoder_file = GENE_ENCODER_MAP.get(self.gene_encoder_type)
        if encoder_file:
            if self.dataset_name.startswith('Tahoe_'):
                part = self.dataset_name.split('_')[1]  # 例如从'Tahoe_P2'中提取'P2'
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/Tahoe-100M/Tahoe_mini/{part}/{encoder_file}')
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/Tahoe-100M/Tahoe_mini/{part}/default.h5')
            elif self.dataset_name == 'LINCS':
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/Gene_encoder/{self.dataset_name}/{encoder_file}')
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/Gene_encoder/LINCS/default.h5')          
            elif self.dataset_name == 'LINCS965':
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/Gene_encoder/{self.dataset_name}/{encoder_file}')
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/Gene_encoder/LINCS965/default.h5')
            elif self.dataset_name == 'CIGS':
                self.X1_data = load_from_HDF(f'{DATA_ROOT}/CIGS/Gene_Encoder/{encoder_file}')
                # print(self.X1_data)
                self.X2_data = load_from_HDF(f'{DATA_ROOT}/CIGS/CIGS_2Cell_lines.h5')
        else:
            raise ValueError(f"不支持的基因编码器类型: {self.gene_encoder_type}")
                
        # ===== 2. 计算 Systema 用的 perturbed centroid =====
        self.perturbed_centroid = perturbed_centroid

    def __getitem__(self, index):
        mol_feature = self.smi2emb[str(self.mol_id[index])].astype(np.float32)
        
        control_gene = self.X1_data['control'][index]
        target_gene = self.X1_data['target'][index]
        
        # --- 数据增强 ---
        if self.augment:
            control_gene = apply_gaussian_noise(control_gene, noise_level=0.01)
            control_gene = apply_gene_masking(control_gene, mask_prob=0.05)
        
        control_gene = control_gene.astype(np.float32)
        mol_feature = mol_feature.astype(np.float32)
        
        return control_gene, target_gene, self.X2_data['control'][index],\
            self.X2_data['target'][index], mol_feature, self.mol_id[index], \
                self.cid[index]

    # ===== 3. 给外部调用者暴露接口 =====
    def get_perturbed_centroid(self):
        return self.perturbed_centroid
    
    def __len__(self):
        return self.mol_id.shape[0]


class Gene_47_Dataset(Dataset):
    def __init__(self, dataset_name, mol_feature_type, gene_encoder_type, mol_id, perturbed_centroid=None, augment=False):

        self.mol_feature_type = mol_feature_type
        self.gene_encoder_type = gene_encoder_type
        self.mol_id = mol_id
        self.dataset_name = dataset_name
        self.augment = augment
        
        # 加载smiles嵌入
        embedding_file = EMBEDDING_CONFIG.get(self.mol_feature_type)
        if embedding_file:
            embedding_path = f'{ARTIFACT_ROOT}/Molecule_encoder/embeddings/Image/{embedding_file}'
            if self.dataset_name == 'BBBC047':
                embedding_path = f'{ARTIFACT_ROOT}/data/MVC/Molecule_encoder/{embedding_file}'
            with open(embedding_path, 'rb') as f:
                self.smi2emb = pickle.load(f)
        else:
            raise ValueError(f"不支持的分子特征类型: {self.mol_feature_type}")

        # 加载图像特征数据
        self.image_data = load_from_HDF(f'{ARTIFACT_ROOT}/data/Image/{self.dataset_name}_data.h5')
        
        # ===== 2. 计算 Systema 用的 perturbed centroid =====
        self.perturbed_centroid = perturbed_centroid
                
    def __getitem__(self, index):
        # print(index, self.mol_id[index])
        mol_feature = self.smi2emb[str(self.mol_id[index])].astype(np.float32)
        
        control_ge = self.image_data['control_GE'][index]
        target_ge = self.image_data['target_GE'][index]
        
        # --- 数据增强 ---
        if self.augment:
            # 降低噪声强度
            control_ge = apply_gaussian_noise(control_ge, noise_level=0.01)
            # 降低Mask比例
            control_ge = apply_gene_masking(control_ge, mask_prob=0.05) 

        control_ge = control_ge.astype(np.float32)
        mol_feature = mol_feature.astype(np.float32)

        return control_ge, target_ge, self.image_data['control_GE'][index],\
            self.image_data['target_GE'][index], mol_feature, self.mol_id[index]

    def __len__(self):
        return self.mol_id.shape[0]
    
    # ===== 3. 给外部调用者暴露接口 =====
    def get_perturbed_centroid(self):
        return self.perturbed_centroid

class Image_Dataset(Dataset):
    def __init__(self, dataset_name, mol_feature_type, image_encoder_type, mol_id, perturbed_centroid=None, augment=False):

        self.mol_feature_type = mol_feature_type
        self.image_encoder_type = image_encoder_type
        self.mol_id = mol_id
        self.dataset_name = dataset_name
        self.augment = augment
        
        # 加载smiles嵌入
        embedding_file = EMBEDDING_CONFIG.get(self.mol_feature_type)
        if embedding_file:
            embedding_path = f'{ARTIFACT_ROOT}/Molecule_encoder/embeddings/Image/{embedding_file}'
            if self.dataset_name == 'BBBC047':
                embedding_path = f'{ARTIFACT_ROOT}/data/MVC/Molecule_encoder/{embedding_file}'
            with open(embedding_path, 'rb') as f:
                self.smi2emb = pickle.load(f)
        else:
            raise ValueError(f"不支持的分子特征类型: {self.mol_feature_type}")

        # ===== 2. 计算 Systema 用的 perturbed centroid =====
        self.perturbed_centroid = perturbed_centroid        
        
        # 加载图像特征数据
        self.image_data = load_from_HDF(f'{ARTIFACT_ROOT}/data/MVC/MVC_{self.dataset_name}/Processed_Paired_CP.h5')
                
    def __getitem__(self, index):
        # print(index, self.mol_id[index])
        mol_feature = self.smi2emb[str(self.mol_id[index])].astype(np.float32)
        
        control_img = self.image_data['control'][index]
        target_img = self.image_data['target'][index]
        
        # --- 数据增强 ---
        if self.augment:
            # 图像特征增强：温和的噪声
            control_img = apply_gaussian_noise(control_img, noise_level=0.01)

        control_img = control_img.astype(np.float32)
        mol_feature = mol_feature.astype(np.float32)

        return control_img, target_img, self.image_data['control'][index],\
            self.image_data['target_img'][index], mol_feature, self.mol_id[index]

    def __len__(self):
        return self.mol_id.shape[0]
    
    # ===== 3. 给外部调用者暴露接口 =====
    def get_perturbed_centroid(self):
        return self.perturbed_centroid


class MVC_Dataset(Dataset):
    """
    核心任务数据集：条件化多模态状态预测
    输入: x_ge (control_GE), x_cp (control_CP), z_mol (mol_feature)
    输出: y_ge (target_GE), y_cp (target_CP)

    [数据增强 v12.0 - 非对称极限施压版 (Asymmetric Extreme Pressure)]
    问题诊断: v11.0 (Mol Dropout 20%) 取得了目前最佳结果 (Valid Loss ~1033)，
              证明"阻断捷径"策略有效。但训练集 (Loss ~880) 与验证集差距仍有 ~150，
              说明模型仍在一定程度上"偷懒"。
    改进策略: 在 v11.0 基础上，进一步压榨模型潜力，实行非对称施压。
    1. 极限 Mol Dropout (25%): 
       - 继续提升分子遮挡比例至 25%，进一步增加条件推断的难度。
    2. 强化 GE Dropout (18%): 
       - 基因特征冗余度高，从 15% 微调至 18%，在不破坏语义的前提下增加难度。
    3. 提升基础噪声 (0.005): 
       - 将基础数值噪声从 0.002 提升至 0.005，增加对微小测量误差的鲁棒性。
    4. 保持 CP 稳定 (5%): 形态特征最为敏感，保持 5% 不变。
    """
    def __init__(
        self,
        dataset_name,
        mol_feature_type,
        gene_encoder_type,
        mol_id,
        paired_data=None,
        normalization_stats=None,
        perturbed_centroid_CP=None,
        perturbed_centroid_GE=None,
        augment=False,
        # ===== 增强超参 (非对称施压) =====
        ge_drop_prob=0.18,    # 微调：15% -> 18%
        cp_drop_prob=0.05,    # 保持：5% (敏感特征不宜过大)
        mol_drop_prob=0.25,   # 极限：20% -> 25%
        mol_noise_std=0.01,   # 保持：分子噪声
        noise_std=0.005,      # 提升：0.002 -> 0.005
        target_noise_std=0.0, # 保持：0
        use_dose_feature=False,
        # 移除 Winsor 参数调整，保持默认
        winsor_q_low=0.001,
        winsor_q_high=0.999,
        winsor_after_norm=True,
    ):

        self.mol_feature_type = mol_feature_type
        self.gene_encoder_type = gene_encoder_type
        self.mol_id = mol_id
        self.dataset_name = dataset_name
        self.augment = augment  # 是否启用数据增强
        self.normalization_stats = normalization_stats
        self.use_dose_feature = bool(use_dose_feature)

        # 只用训练集里面的扰动中心
        self.perturbed_centroid_CP = perturbed_centroid_CP
        self.perturbed_centroid_GE = perturbed_centroid_GE

        # ===== 1) 读取分子 embedding =====
        self.smi2emb = load_molecule_embedding_lookup(self.mol_feature_type)

        # ===== 2) 读取 CP/GE 成对数据 =====
        if paired_data is not None:
            self.Image_Gene_data = {
                key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                for key, value in paired_data.items()
            }
        else:
            if self.dataset_name == 'BBBC047':
                self.Image_Gene_data = load_from_HDF(f'{DATA_ROOT}/MVC_BBBC047/Paired_CP_GE_Data.h5')
            elif self.dataset_name == 'BBBC036':
                self.Image_Gene_data = load_from_HDF(f'{DATA_ROOT}/MVC_BBBC036/Paired_CP_GE_Data.h5')
            else:
                raise ValueError(f"Unsupported dataset_name in MVC_Dataset: {self.dataset_name}")

        if 'canonical_smiles' in self.Image_Gene_data:
            paired_smiles = np.asarray(self.Image_Gene_data['canonical_smiles']).astype(str)
            mol_id_smiles = np.asarray(self.mol_id).astype(str)
            if len(paired_smiles) != len(mol_id_smiles):
                raise ValueError(
                    f"paired_data length ({len(paired_smiles)}) and mol_id length ({len(mol_id_smiles)}) do not match"
                )
            if not np.array_equal(paired_smiles, mol_id_smiles):
                raise ValueError("paired_data canonical_smiles and mol_id are not aligned row-by-row")
            self.Image_Gene_data['canonical_smiles'] = paired_smiles

        if self.use_dose_feature:
            if 'pert_dose' not in self.Image_Gene_data:
                raise ValueError("use_dose_feature=True but paired_data does not contain pert_dose")
            dose_values = np.asarray(self.Image_Gene_data['pert_dose'], dtype=np.float32)
            if dose_values.shape[0] != len(self.mol_id):
                raise ValueError(
                    f"pert_dose length ({dose_values.shape[0]}) and mol_id length ({len(self.mol_id)}) do not match"
                )
            if not np.isfinite(dose_values).all():
                raise ValueError("pert_dose contains non-finite values")
            if np.any(dose_values <= 0):
                raise ValueError("pert_dose must be positive when use_dose_feature=True")
            self.Image_Gene_data['pert_dose'] = dose_values

        # ===== 3) 用训练集统计量做 z-score 归一化（按 feature）=====
        self._assert_modalities_are_finite(self.Image_Gene_data, stage='pre-normalization')
        if self.normalization_stats is None:
            self.normalization_stats = self._build_normalization_stats(self.Image_Gene_data)
        self._apply_normalization(self.Image_Gene_data, self.normalization_stats)
        self._assert_modalities_are_finite(self.Image_Gene_data, stage='post-normalization')

        # ===== 4) 预计算 winsor 边界（只算一次，__getitem__ 直接 clip）=====
        self._winsor_after_norm = winsor_after_norm
        self._winsor_q_low = winsor_q_low
        self._winsor_q_high = winsor_q_high

        self._cp_lo = self.normalization_stats['control_cp_lo']
        self._cp_hi = self.normalization_stats['control_cp_hi']
        self._ge_lo = self.normalization_stats['control_ge_lo']
        self._ge_hi = self.normalization_stats['control_ge_hi']

        # ===== 5) 保存增强参数 =====
        self._ge_drop_prob = ge_drop_prob
        self._cp_drop_prob = cp_drop_prob
        self._mol_drop_prob = mol_drop_prob 
        self._mol_noise_std = mol_noise_std # 保存分子噪声参数
        self._noise_std = noise_std         
        self._target_noise_std = target_noise_std

    @staticmethod
    def _safe_std(values):
        std = values.std(axis=0)
        std = np.asarray(std, dtype=np.float32)
        std[std < 1e-8] = 1.0
        return std

    @classmethod
    def _build_normalization_stats(cls, image_gene_data):
        control_cp_mean = image_gene_data['control_CP'].mean(axis=0).astype(np.float32)
        control_cp_std = cls._safe_std(image_gene_data['control_CP'])
        target_cp_mean = image_gene_data['target_CP'].mean(axis=0).astype(np.float32)
        target_cp_std = cls._safe_std(image_gene_data['target_CP'])
        control_ge_mean = image_gene_data['control_GE'].mean(axis=0).astype(np.float32)
        control_ge_std = cls._safe_std(image_gene_data['control_GE'])
        target_ge_mean = image_gene_data['target_GE'].mean(axis=0).astype(np.float32)
        target_ge_std = cls._safe_std(image_gene_data['target_GE'])

        normalized_control_cp = ((image_gene_data['control_CP'] - control_cp_mean) / control_cp_std).astype(np.float32)
        normalized_control_ge = ((image_gene_data['control_GE'] - control_ge_mean) / control_ge_std).astype(np.float32)
        control_cp_lo, control_cp_hi = _robust_bounds(normalized_control_cp)
        control_ge_lo, control_ge_hi = _robust_bounds(normalized_control_ge)

        stats = {
            'control_cp_mean': control_cp_mean,
            'control_cp_std': control_cp_std,
            'target_cp_mean': target_cp_mean,
            'target_cp_std': target_cp_std,
            'control_ge_mean': control_ge_mean,
            'control_ge_std': control_ge_std,
            'target_ge_mean': target_ge_mean,
            'target_ge_std': target_ge_std,
            'control_cp_lo': control_cp_lo,
            'control_cp_hi': control_cp_hi,
            'control_ge_lo': control_ge_lo,
            'control_ge_hi': control_ge_hi,
        }
        if 'pert_dose' in image_gene_data:
            dose_log10 = np.log10(np.asarray(image_gene_data['pert_dose'], dtype=np.float32))
            dose_log10_mean = np.float32(dose_log10.mean())
            dose_log10_std = np.float32(dose_log10.std())
            if dose_log10_std < 1e-8:
                dose_log10_std = np.float32(1.0)
            stats['dose_log10_mean'] = dose_log10_mean
            stats['dose_log10_std'] = dose_log10_std
        return stats

    @staticmethod
    def _assert_modalities_are_finite(image_gene_data, stage):
        bad_modalities = []
        for key in ('control_CP', 'target_CP', 'control_GE', 'target_GE'):
            values = np.asarray(image_gene_data[key])
            non_finite_count = int(values.size - np.isfinite(values).sum())
            if non_finite_count:
                bad_modalities.append(f"{key}={non_finite_count}")
        if bad_modalities:
            raise ValueError(
                f"paired_data contains non-finite values at {stage}: {', '.join(bad_modalities)}"
            )

    @staticmethod
    def _apply_normalization(image_gene_data, normalization_stats):
        image_gene_data['control_CP'] = (
            (image_gene_data['control_CP'] - normalization_stats['control_cp_mean']) /
            normalization_stats['control_cp_std']
        ).astype(np.float32)
        image_gene_data['target_CP'] = (
            (image_gene_data['target_CP'] - normalization_stats['target_cp_mean']) /
            normalization_stats['target_cp_std']
        ).astype(np.float32)
        image_gene_data['control_GE'] = (
            (image_gene_data['control_GE'] - normalization_stats['control_ge_mean']) /
            normalization_stats['control_ge_std']
        ).astype(np.float32)
        image_gene_data['target_GE'] = (
            (image_gene_data['target_GE'] - normalization_stats['target_ge_mean']) /
            normalization_stats['target_ge_std']
        ).astype(np.float32)

    def _get_raw_sample(self, index):
        """内部辅助函数：读取原始样本并进行基础转换"""
        mol_key = str(self.mol_id[index])
        mol_feature = self.smi2emb.get(mol_key)
        if mol_feature is None:
            if self.mol_feature_type == 'ECFP4':
                mol_feature = compute_rdkit_ecfp4_embedding(mol_key)
                self.smi2emb[mol_key] = mol_feature
            else:
                raise KeyError(f"未找到分子 embedding: {mol_key}")
        mol_feature = np.asarray(mol_feature, dtype=np.float32)
        if self.use_dose_feature:
            dose_value = float(self.Image_Gene_data['pert_dose'][index])
            dose_log10 = np.log10(max(dose_value, 1e-8))
            dose_mean = float(self.normalization_stats.get('dose_log10_mean', 0.0))
            dose_std = float(self.normalization_stats.get('dose_log10_std', 1.0))
            dose_feature = np.asarray([(dose_log10 - dose_mean) / max(dose_std, 1e-8)], dtype=np.float32)
            mol_feature = np.concatenate([mol_feature, dose_feature], axis=0)
        control_cp = self.Image_Gene_data['control_CP'][index]
        control_ge = self.Image_Gene_data['control_GE'][index]
        target_cp  = self.Image_Gene_data['target_CP'][index]
        target_ge  = self.Image_Gene_data['target_GE'][index]
        return control_cp, control_ge, target_cp, target_ge, mol_feature

    def _apply_intra_sample_augment(self, control_cp, control_ge):
        """内部辅助函数：应用全模态微扰 (v12.0 逻辑)"""
        
        # (A) Winsor 裁剪
        if self._winsor_after_norm:
            control_cp = winsorize(control_cp, self._cp_lo, self._cp_hi)
            control_ge = winsorize(control_ge, self._ge_lo, self._ge_hi)

        # (B) 噪声：GE/CP 基础噪声提升 (0.005)
        control_cp = apply_noise(control_cp, noise_std=self._noise_std, dist='student_t', df=3)
        control_ge = apply_noise(control_ge, noise_std=self._noise_std, dist='student_t', df=3)

        # (C) Feature dropout (Masking)
        control_ge = apply_feature_dropout(control_ge, drop_prob=self._ge_drop_prob, keep_zeros=True)
        control_cp = apply_feature_dropout(control_cp, drop_prob=self._cp_drop_prob, keep_zeros=True)
            
        return control_cp, control_ge

    def __getitem__(self, index):
        # 1. 获取当前样本
        c_cp, c_ge, t_cp, t_ge, mol = self._get_raw_sample(index)

        # --- 数据增强核心逻辑 ---
        if self.augment:
            # 2. 应用细胞状态增强
            c_cp, c_ge = self._apply_intra_sample_augment(c_cp, c_ge)
            
            # 3. 分子特征增强 (Extreme)
            # 噪声 + 极限遮挡 (25%)
            # dose-aware 训练时，末尾追加的是 dose scalar；不要对它做噪声/遮挡。
            if self.use_dose_feature and mol.shape[0] >= 1:
                mol_base = mol[:-1]
                mol_dose = mol[-1:]
                mol_base = apply_noise(mol_base, noise_std=self._mol_noise_std, dist='gaussian')
                mol_base = apply_feature_dropout(mol_base, drop_prob=self._mol_drop_prob, keep_zeros=True)
                mol = np.concatenate([mol_base, mol_dose], axis=0)
            else:
                mol = apply_noise(mol, noise_std=self._mol_noise_std, dist='gaussian')
                mol = apply_feature_dropout(mol, drop_prob=self._mol_drop_prob, keep_zeros=True)
            
            # 4. 目标端：保持完全纯净
            # t_cp = ...

        # 确保数据类型正确
        control_cp = c_cp.astype(np.float32)
        control_ge = c_ge.astype(np.float32)
        target_cp  = t_cp.astype(np.float32)
        target_ge  = t_ge.astype(np.float32)
        mol_feature = mol.astype(np.float32)

        return control_cp, control_ge, target_cp, target_ge, mol_feature, self.mol_id[index]

    def __len__(self):
        return self.mol_id.shape[0]

    # ===== 3. 给外部调用者暴露接口 =====
    def get_perturbed_centroid(self):
        return self.perturbed_centroid_CP, self.perturbed_centroid_GE

    def get_normalization_stats(self):
        return self.normalization_stats
