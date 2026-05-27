#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
import numpy as np
from utils import *
# from geomloss import SamplesLoss


# class MMDLoss(nn.Module):
#     def __init__(self, kernel="energy", blur=0.05, scaling=0.5, downsample=1):
#         super().__init__()
#         self.mmd_loss = SamplesLoss(loss=kernel, blur=blur, scaling=scaling)
#         self.downsample = downsample

#     def forward(self, input, target):
#         input = input.reshape(-1, self.downsample, input.shape[-1])
#         target = target.reshape(-1, self.downsample, target.shape[-1])

#         return self.mmd_loss(input, target).mean()


class TaskWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_ge = nn.Parameter(torch.zeros(1))
        self.log_sigma_cp = nn.Parameter(torch.zeros(1))
    def loss(self, l_ge, l_cp):
        w_ge = torch.exp(-2*self.log_sigma_ge)
        w_cp = torch.exp(-2*self.log_sigma_cp)
        return w_ge*l_ge + w_cp*l_cp + (self.log_sigma_ge + self.log_sigma_cp)

class MVCModel(torch.nn.Module):
    def __init__(self, n_genes, n_images, n_emd, n_latent, n_en_hidden, n_de_hidden, molecule_feature_dim, molecule_hidden, **kwargs):
        """
        Initialize MVCModel model.
        :param n_genes: Number of genes in the dataset.
        :param n_emd: Embedding dimension for the input data.
        :param n_latent: Latent space dimension.
        :param n_en_hidden: List of hidden layer sizes for the encoder.
        :param n_de_hidden: List of hidden layer sizes for the decoder.
        :param smile_feature_dim: Dimension of the smiles features.
        :param kwargs: Additional keyword arguments.
        """
        super(MVCModel, self).__init__()
        self.n_genes = n_genes
        self.n_images = n_images
        self.out_genes = n_genes
        self.out_images = n_images
        self.n_emb = n_emd
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.molecule_feature_dim = molecule_feature_dim
        self.molecule_hidden = molecule_hidden
        self.init_w = kwargs.get('init_w', False)
        self.path_model = kwargs.get('path_model', 'trained_model')
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.2)
        self.random_seed = kwargs.get('random_seed', 1234)
        self.use_dose_feature = bool(kwargs.get('use_dose_feature', False))
        self.dose_ordinal_weight = float(kwargs.get('dose_ordinal_weight', 0.0))
        self.dose_ordinal_temperature = float(kwargs.get('dose_ordinal_temperature', 0.2))
        self.dose_ordinal_margin_scale = float(kwargs.get('dose_ordinal_margin_scale', 0.25))
        # self.MMDloss = MMDLoss()
        # self.uncertainty_loss = UncertaintyLoss() # 2个任务
        self.task_weight = TaskWeight()  # 2个任务
        
        # Encoder        
        encoder_cp = [
            nn.Linear(self.n_images, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
            nn.Dropout(self.dropout)
        ]
        encoder_ge = [
            nn.Linear(self.n_genes, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
            nn.Dropout(self.dropout)
        ]
        self.encoder_cp = nn.Sequential(*encoder_cp)
        self.encoder_ge = nn.Sequential(*encoder_ge)
        
        # latent space
        self.latent = nn.Sequential(
            nn.Linear(self.n_latent, self.n_latent),
            nn.BatchNorm1d(self.n_latent),
            nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
            nn.Dropout(self.dropout)
        )
                
        # late latent
        decoder_late_cp = [
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
                nn.Dropout(self.dropout)
                ]
        decoder_late_ge = [
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
                nn.Dropout(self.dropout)
                ]
        decoder_late_cp.append(nn.Linear(self.n_de_hidden[-1], self.n_latent))
        decoder_late_ge.append(nn.Linear(self.n_de_hidden[-1], self.n_latent))
        
        self.decoder_late_cp = nn.Sequential(*decoder_late_cp)
        self.decoder_late_ge = nn.Sequential(*decoder_late_ge)
        
        # decoder
        decoder_cp = [
                # nn.Linear(1024, self.n_de_hidden[0]),
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
                nn.Dropout(self.dropout)
                ]
        decoder_ge = [
                # nn.Linear(1024, self.n_de_hidden[0]),
                nn.Linear(self.n_latent, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(0.1),  # 添加负斜率参数(如0.1)
                nn.Dropout(self.dropout)
                ]
        decoder_cp.append(nn.Linear(self.n_de_hidden[-1], self.out_images))
        decoder_ge.append(nn.Linear(self.n_de_hidden[-1], self.out_genes))
        
        self.decoder_ge = nn.Sequential(*decoder_ge)
        self.decoder_cp = nn.Sequential(*decoder_cp)
        
        if self.molecule_hidden != None:  # 512
            feat_embeddings = [
                                nn.Linear(self.molecule_feature_dim, self.molecule_hidden),
                                nn.BatchNorm1d(self.molecule_hidden),
                                nn.ReLU(),
                                nn.Dropout(self.dropout)
                                ]
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

        if self.init_w:
            self.encoder_cp.apply(self._init_weights)
            self.encoder_ge.apply(self._init_weights)
            self.latent.apply(self._init_weights)
            self.decoder_ge.apply(self._init_weights)
            self.decoder_ge.apply(self._init_weights)

    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    def forward(self, x_cp, x_ge, features):
        """ Forward pass through full network"""
        if torch.isnan(x_cp).any() or torch.isnan(x_ge).any() or torch.isnan(features).any():
            print("⚠️ 输入里有 NaN")

        if self.molecule_hidden != None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features
        
        # 单模态
        # z_cp = self.encoder_cp(x_cp)
        # # z_ge = self.encoder_ge(x_ge)
        # z1_feat = torch.cat([z_cp, feat_embed], dim=1)
        # # print("z1_feat shape:", z1_feat.shape) # [1000, 1024]
        # cp_pred = self.decoder_cp(z1_feat)        
        # ge_pred = self.decoder_ge(z1_feat)
        
        z_cp = self.encoder_cp(x_cp)
        z_ge = self.encoder_ge(x_ge)
        z1_feat = torch.cat([z_ge, z_cp, feat_embed], dim=1)
        # print("z1_feat shape:", z1_feat.shape) # [1000, 1024]
        cp_pred = self.decoder_cp(z1_feat)        
        ge_pred = self.decoder_ge(z1_feat)
        
        # # 日常融合：
        # z_cp = self.encoder_cp(x_cp)
        # z_ge = self.encoder_ge(x_ge)
        # z1_feat = torch.cat([z_cp, z_ge, feat_embed], dim=1)
        # # print("z1_feat shape:", z1_feat.shape) # [1000, 1536]
        # cp_pred = self.decoder_cp(z1_feat)
        # ge_pred = self.decoder_ge(z1_feat)
        
        # 早期融合：将分子特征和图特征一起编码
        # print("x_cp shape:", x_cp.shape) # [1000, 745])
        # print("x_ge shape:", x_ge.shape) # [1000, 977])
        # z_cp_ge = torch.cat([x_cp, x_ge, feat_embed], dim=1)
        # # print("z_cp_ge shape:", z_cp_ge.shape) # ([1000, 2234])
        # z1_feat = self.latent(z_cp_ge) # [1000, 256])
        # ge_pred = self.decoder_ge(z1_feat)
        # cp_pred = self.decoder_cp(z1_feat)
        
        # 中期编码
        # z_cp = self.encoder_cp(x_cp)
        # z_ge = self.encoder_ge(x_ge)
        # # print("z_cp shape:", z_cp.shape) # [1000, 256])
        # z1_feat = torch.cat([z_cp, z_ge, feat_embed], dim=1)    
        # # print("z1_feat shape:", z1_feat.shape) # [1000, 1536]
        # cp_pred_from_cp = self.decoder_late_cp(z1_feat)
        # ge_pred_from_ge = self.decoder_late_ge(z1_feat)
        # # print("cp_pred_from_cp shape:", cp_pred_from_cp.shape) # [1000, 1]
        # ge_pred = self.decoder_ge(ge_pred_from_ge)
        # cp_pred = self.decoder_cp(cp_pred_from_cp)
        
        # 后期融合：
        # z_cp = self.encoder_cp(x_cp)
        # z_ge = self.encoder_ge(x_ge)        
        # # 独立解码
        # cp_pred_from_cp = self.decoder_late_cp(z_cp)
        # ge_pred_from_ge = self.decoder_late_ge(z_ge)                
        # # 1. 最简单：拼接后接一个小网络
        # joint = torch.cat([cp_pred_from_cp, ge_pred_from_ge, feat_embed], dim=1)
        # # print("joint shape:", joint.shape) # [1000, 2234])
        # cp_pred = self.decoder_cp(joint)                 # [B, 1]
        # ge_pred = self.decoder_ge(joint)                 # [B, 1]
        
        # 2. 交叉解码
        
        return cp_pred, ge_pred
        # return cp_pred, ge_pred, cp_pred_from_cp, ge_pred_from_ge, feat_embed

    def _compute_dose_ordinal_aux_loss(self, ge_pred, target_ge, control_ge=None, features=None, mol_ids=None):
        if (
            self.dose_ordinal_weight <= 0.0
            or not self.use_dose_feature
            or features is None
            or mol_ids is None
            or features.shape[0] < 2
        ):
            return ge_pred.new_zeros(())

        dose_feature = features[:, -1].reshape(-1)
        ge_pred_delta = ge_pred - control_ge if control_ge is not None else ge_pred
        target_ge_delta = target_ge - control_ge if control_ge is not None else target_ge
        aux = compute_same_compound_dose_ordinal_loss(
            pred_delta=ge_pred_delta,
            target_delta=target_ge_delta,
            mol_ids=mol_ids,
            dose_feature=dose_feature,
            temperature=self.dose_ordinal_temperature,
            margin_scale=self.dose_ordinal_margin_scale,
        )
        return self.dose_ordinal_weight * aux["loss"]

    def loss(
        self,
        target_cp,
        target_ge,
        cp_pred,
        ge_pred,
        control_ge=None,
        features=None,
        mol_ids=None,
        weight_cp=0.01,
        weight_ge=1.0,
    ):
        
        mse_ge = F.mse_loss(target_ge, ge_pred, reduction="sum")
        mse_cp = F.mse_loss(target_cp, cp_pred, reduction="sum")
        
        # with torch.no_grad():
        #     # 让难度大的任务权重小一些（避免主导训练）
        #     weight_ge = mse_cp / (mse_ge + mse_cp + 1e-8)
        #     weight_cp = mse_ge / (mse_ge + mse_cp + 1e-8)
        
        # balanced_loss = weight_ge * mse_ge + weight_cp * mse_cp
        
        # mse_cp = weight_cp * mse_cp        
        # mse_ge = weight_ge * mse_ge   # weight_cp=0.01, weight_ge=1.0 有进步，降低形态果然可以
        
        # print(mse_cp.item(), mse_ge.item())
        # balanced_loss = self.task_weight.loss(mse_ge, mse_cp)
        
        # loss = mse_ge
        
        total_loss = mse_ge + mse_cp
        total_loss = total_loss + self._compute_dose_ordinal_aux_loss(
            ge_pred=ge_pred,
            target_ge=target_ge,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        return total_loss
        # return balanced_loss
    
    def get_feat_embdding(self, features):
        feat_embed = self.feat_embeddings(features)
        return feat_embed

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        """ Train MVCModel """
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ['loss']

        # Train
        best_value = np.inf
        best_epoch = 0
        for epoch in range(n_epochs):
            train_size = 0
            loss_value = 0
            self.train()
            for control_cp, control_ge, target_cp, target_ge, features, mol_id in train_loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                features = features.to(self.dev)
                if control_cp.shape[0] == 1:
                    continue
                train_size += control_cp.shape[0]
                optimizer.zero_grad()

                cp_pred, ge_pred = self.forward(control_cp, control_ge, features)
                
                loss = self.loss(
                    target_cp,
                    target_ge,
                    cp_pred,
                    ge_pred,
                    control_ge=control_ge,
                    features=features,
                    mol_ids=mol_id,
                )

                loss_value += loss.item()
                loss.backward()
                optimizer.step()

            # Eval
            train_dict, train_metrics_dict, train_metrics_dict_ls= self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict['loss']

            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)

            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            test_dict, test_metrics_dict, test_metricsdict_ls= self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_loss = test_dict['loss']

            for k, v in test_dict.items():
                epoch_hist['valid_'+k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)
            if epoch % 10 == 0:
                print('[Epoch %d] | train loss: %.5f, valid_loss: %.5f' % (epoch, train_loss, test_loss), flush=True)

            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None, perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        """Test model on input loader."""

        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                mol_features = mol_features.to(self.dev)
                # cid = np.array(list(cid))
                # sig = np.array(list(sig))
                test_size += control_cp.shape[0]
                
                # cp_pred, ge_pred, _, _, _ = self.forward(control_cp, control_ge, mol_features)
                cp_pred, ge_pred = self.forward(control_cp, control_ge, mol_features)
                loss_ls = self.loss(
                    target_cp,
                    target_ge,
                    cp_pred,
                    ge_pred,
                    control_ge=control_ge,
                    features=mol_features,
                    mol_ids=mol_id,
                )
                
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls.item()

                if metrics_func != None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(control_cp, control_ge, target_cp, target_ge, cp_pred, ge_pred, \
                        metrics_func=metrics_func, perturbed_centroid_CP=perturbed_centroid_CP, perturbed_centroid_GE=perturbed_centroid_GE)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls['cp_id'] += list(mol_id)
                    # metrics_dict_all_ls['cid'] += list(cid)
                    # metrics_dict_all_ls['sig'] += list(sig)

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size

        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls
    

    def predict_profile(self, loader):
        """predict profiles."""

        self.eval()
        with torch.no_grad():
            control_cp_list = []
            control_ge_list = []
            target_cp_list = []
            target_ge_list = []
            cp_pred_list = []
            ge_pred_list = []
            smiles_list = []

            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                mol_features = mol_features.to(self.dev)

                cp_pred, ge_pred = self.forward(control_cp, control_ge, mol_features)

                control_cp_list.append(control_cp.cpu().numpy().astype(float))
                control_ge_list.append(control_ge.cpu().numpy().astype(float))
                target_cp_list.append(target_cp.cpu().numpy().astype(float))
                target_ge_list.append(target_ge.cpu().numpy().astype(float))
                cp_pred_list.append(cp_pred.cpu().numpy().astype(float))
                ge_pred_list.append(ge_pred.cpu().numpy().astype(float))
                smiles_list.extend(list(mol_id))

        control_cp_array = np.concatenate(control_cp_list, axis=0)
        control_ge_array = np.concatenate(control_ge_list, axis=0)
        target_cp_array = np.concatenate(target_cp_list, axis=0)
        target_ge_array = np.concatenate(target_ge_list, axis=0)
        cp_pred_array = np.concatenate(cp_pred_list, axis=0)
        ge_pred_array = np.concatenate(ge_pred_list, axis=0)
        smiles_array = np.array(smiles_list)

        return (
            control_cp_array,
            control_ge_array,
            target_cp_array,
            target_ge_array,
            cp_pred_array,
            ge_pred_array,
            smiles_array,
        )


    def predict_profile_emb(self, loader):
        """predict profiles."""

        test_size = 0
        # setup_seed(self.random_seed)
        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, _ in loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                mol_features = mol_features.to(self.dev)
                test_size += control_cp.shape[0]

                cp_pred, ge_pred, cp_emb, ge_emb, smiles_emb = self.forward(control_cp, control_ge, mol_features)

                try:
                    x1_array = torch.cat([x1_array, control_cp], dim=0)
                    x2_array = torch.cat([x2_array, control_ge], dim=0)
                    x2_cp_pred_array = torch.cat([x2_cp_pred_array, cp_pred], dim=0)
                    x2_ge_pred_array = torch.cat([x2_ge_pred_array, ge_pred], dim=0)
                    
                    cp_emb_array = torch.cat([cp_emb_array, cp_emb], dim=0)
                    ge_emb_array = torch.cat([ge_emb_array, ge_emb], dim=0)
                    smiles_emb_array = torch.cat([smiles_emb_array, smiles_emb], dim=0)
                except:
                    x1_array = control_cp.clone()
                    x2_array = control_ge.clone()
                    cp_emb_array = cp_emb.clone()
                    ge_emb_array = ge_emb.clone()
                    smiles_emb_array = smiles_emb.clone()
                    x2_cp_pred_array = cp_pred.clone()
                    x2_ge_pred_array = ge_pred.clone()

        x1_array = x1_array.cpu().numpy().astype(float)
        x2_array = x2_array.cpu().numpy().astype(float)
        smiles_emb_array = smiles_emb_array.cpu().numpy().astype(float)
        cp_emb_array = cp_emb_array.cpu().numpy().astype(float)
        ge_emb_array = ge_emb_array.cpu().numpy().astype(float)
        x2_cp_pred_array = x2_cp_pred_array.cpu().numpy().astype(float)
        x2_ge_pred_array = x2_ge_pred_array.cpu().numpy().astype(float)
        # mol_id_array = mol_id_array.numpy().astype(float)
        return x1_array, x2_array, cp_emb_array, ge_emb_array, smiles_emb_array, x2_cp_pred_array, x2_ge_pred_array


    def predict_profile_for_x1(self, loader):

        self.eval()
        with torch.no_grad():
            for control_cp, mol_features, mol_id, cid in loader:
                cid = np.array(list(cid))
                control_cp = control_cp.to(self.dev)
                mol_features = mol_features.to(self.dev)
                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.forward(control_cp, mol_features)
                try:
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)
                    # mol_id_array = torch.cat([mol_id_array, mol_id], dim=0)
                    mol_id_array = np.concatenate((mol_id_array, mol_id), axis=0)
                    cid_array = np.concatenate((cid_array, cid), axis=0)
                except:
                    x2_pred_array = x2_pred.clone()
                    mol_id_array = list(mol_id).copy()
                    # mol_id_array = mol_id.clone()
                    cid_array = cid.copy()

        x2_pred_array = x2_pred_array.cpu().numpy().astype(float)
        # mol_id_array = mol_id_array.cpu().numpy().astype(float)

        return x2_pred_array, mol_id_array, cid_array

    def eval_x_reconstruction(self, control_cp, control_ge, target_cp, target_ge, cp_pred, ge_pred, metrics_func=['pearson'], perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        """
        Compute reconstruction evaluation metrics for x1_rec, x2_rec, x2_pred（批量/并行优化版）
        """
        control_cp_np = None  # 保留兼容变量名，但改为批量实现
        use_torch = control_cp.is_cuda or cp_pred.is_cuda
        cp_true = target_cp.float()
        cp_pred_ = cp_pred.float()
        ge_true = target_ge.float()
        ge_pred_ = ge_pred.float()
        ccp = control_cp.float()
        cge = control_ge.float()
        device = cp_true.device

        if perturbed_centroid_CP is None:
            perturbed_centroid_CP = cp_true.mean(dim=0)
            perturbed_centroid_GE = ge_true.mean(dim=0)
        else:
            if use_torch:
                perturbed_centroid_CP = torch.tensor(perturbed_centroid_CP, dtype=cp_true.dtype, device=device)
                perturbed_centroid_GE = torch.tensor(perturbed_centroid_GE, dtype=ge_true.dtype, device=device)
            else:
                perturbed_centroid_CP = np.asarray(perturbed_centroid_CP, float)
                perturbed_centroid_GE = np.asarray(perturbed_centroid_GE, float)

        DEG_cp = cp_true - ccp
        DEG_cp_pred = cp_pred_ - ccp
        DEG_ge = ge_true - cge
        DEG_ge_pred = ge_pred_ - cge

        Delta_sys_CP_true = cp_true - perturbed_centroid_CP
        Delta_sys_CP_pred = cp_pred_ - perturbed_centroid_CP
        Delta_sys_GE_true = ge_true - perturbed_centroid_GE
        Delta_sys_GE_pred = ge_pred_ - perturbed_centroid_GE

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)

        def batch_pearson(x, y):
            x0 = x - x.mean(dim=1, keepdim=True)
            y0 = y - y.mean(dim=1, keepdim=True)
            num = (x0 * y0).sum(dim=1)
            den = torch.norm(x0, dim=1) * torch.norm(y0, dim=1) + 1e-8
            v = num / den
            return v.cpu().numpy() if use_torch else v

        def batch_rmse(x, y):
            se = ((x - y) ** 2).mean(dim=1)
            v = torch.sqrt(se)
            return v.cpu().numpy() if use_torch else np.sqrt(se)

        def batch_precision_at_k(x_true, x_pred, k=100, pos_num=100, neg_num=100):
            if use_torch:
                topk_pred = torch.topk(x_pred, k, dim=1).indices
                top_pos_true = torch.topk(x_true, pos_num, dim=1).indices
                top_neg_true = torch.topk(-x_true, neg_num, dim=1).indices
                pred_mask = torch.zeros((*x_pred.shape[:1], x_pred.shape[1]), dtype=torch.bool, device=x_pred.device)
                pos_mask = torch.zeros_like(pred_mask)
                neg_mask = torch.zeros_like(pred_mask)
                pred_mask.scatter_(1, topk_pred, True)
                pos_mask.scatter_(1, top_pos_true, True)
                neg_mask.scatter_(1, top_neg_true, True)
                pos_hits = (pred_mask & pos_mask).sum(dim=1)
                neg_hits = (pred_mask & neg_mask).sum(dim=1)
                return (neg_hits.float() / k, pos_hits.float() / k)
            else:
                x_true = x_true.cpu().numpy() if hasattr(x_true, 'cpu') else x_true
                x_pred = x_pred.cpu().numpy() if hasattr(x_pred, 'cpu') else x_pred
                N, D = x_true.shape
                topk_pred = np.argpartition(x_pred, -k, axis=1)[:, -k:]
                top_pos_true = np.argpartition(x_true, -pos_num, axis=1)[:, -pos_num:]
                top_neg_true = np.argpartition(x_true, neg_num, axis=1)[:, :neg_num]
                pred_mask = np.zeros((N, D), dtype=bool)
                pred_mask[np.arange(N)[:, None], topk_pred] = True
                pos_mask = np.zeros((N, D), dtype=bool)
                pos_mask[np.arange(N)[:, None], top_pos_true] = True
                neg_mask = np.zeros((N, D), dtype=bool)
                neg_mask[np.arange(N)[:, None], top_neg_true] = True
                pos_hits = (pred_mask & pos_mask).sum(axis=1)
                neg_hits = (pred_mask & neg_mask).sum(axis=1)
                return (neg_hits / k, pos_hits / k)

        if 'pearson' in metrics_func:
            p_cp_sys = batch_pearson(Delta_sys_CP_true, Delta_sys_CP_pred)
            p_ge_sys = batch_pearson(Delta_sys_GE_true, Delta_sys_GE_pred)
            metrics_dict['systema_pearson_CP'] = float(np.nansum(p_cp_sys))
            metrics_dict['systema_pearson_GE'] = float(np.nansum(p_ge_sys))
            metrics_dict_ls['systema_pearson_CP'] = p_cp_sys.tolist()
            metrics_dict_ls['systema_pearson_GE'] = p_ge_sys.tolist()
        if 'rmse' in metrics_func:
            r_cp_sys = batch_rmse(Delta_sys_CP_true, Delta_sys_CP_pred)
            r_ge_sys = batch_rmse(Delta_sys_GE_true, Delta_sys_GE_pred)
            metrics_dict['systema_rmse_CP'] = float(np.nansum(r_cp_sys))
            metrics_dict['systema_rmse_GE'] = float(np.nansum(r_ge_sys))
            metrics_dict_ls['systema_rmse_CP'] = r_cp_sys.tolist()
            metrics_dict_ls['systema_rmse_GE'] = r_ge_sys.tolist()

        if 'pearson' in metrics_func:
            cp_pred_pcc = batch_pearson(cp_true, cp_pred_)
            ge_pred_pcc = batch_pearson(ge_true, ge_pred_)
            metrics_dict_ls['cp_pred_pearson'] = cp_pred_pcc.tolist()
            metrics_dict_ls['ge_pred_pearson'] = ge_pred_pcc.tolist()
            deg_cp_pred_pcc = batch_pearson(DEG_cp, DEG_cp_pred)
            deg_ge_pred_pcc = batch_pearson(DEG_ge, DEG_ge_pred)
            metrics_dict_ls['DEG_cp_pred_pearson'] = deg_cp_pred_pcc.tolist()
            metrics_dict_ls['DEG_ge_pred_pearson'] = deg_ge_pred_pcc.tolist()
        if 'rmse' in metrics_func:
            cp_pred_rmse = batch_rmse(cp_true, cp_pred_)
            ge_pred_rmse = batch_rmse(ge_true, ge_pred_)
            metrics_dict_ls['cp_pred_rmse'] = cp_pred_rmse.tolist()
            metrics_dict_ls['ge_pred_rmse'] = ge_pred_rmse.tolist()
            deg_cp_pred_rmse = batch_rmse(DEG_cp, DEG_cp_pred)
            deg_ge_pred_rmse = batch_rmse(DEG_ge, DEG_ge_pred)
            metrics_dict_ls['DEG_cp_pred_rmse'] = deg_cp_pred_rmse.tolist()
            metrics_dict_ls['DEG_ge_pred_rmse'] = deg_ge_pred_rmse.tolist()

        def add_precision_metrics(metric_name, true, pred, prefix):
            if metric_name.startswith('precision'):
                k = int(metric_name[len('precision'):])
                neg, pos = batch_precision_at_k(true, pred, k)
                if use_torch:
                    metrics_dict['%s_neg_%s' % (prefix, metric_name)] = float(neg.sum().item())
                    metrics_dict['%s_pos_%s' % (prefix, metric_name)] = float(pos.sum().item())
                    metrics_dict_ls['%s_neg_%s' % (prefix, metric_name)] = neg.tolist()
                    metrics_dict_ls['%s_pos_%s' % (prefix, metric_name)] = pos.tolist()
                else:
                    metrics_dict['%s_neg_%s' % (prefix, metric_name)] = float(neg.sum())
                    metrics_dict['%s_pos_%s' % (prefix, metric_name)] = float(pos.sum())
                    metrics_dict_ls['%s_neg_%s' % (prefix, metric_name)] = neg.tolist()
                    metrics_dict_ls['%s_pos_%s' % (prefix, metric_name)] = pos.tolist()

        for m in metrics_func:
            add_precision_metrics(m, cp_true, cp_pred_, 'cp_pred')
        for m in metrics_func:
            add_precision_metrics(m, ge_true, ge_pred_, 'ge_pred')
        for m in metrics_func:
            add_precision_metrics(m, DEG_cp, DEG_cp_pred, 'DEG_cp_pred')
        for m in metrics_func:
            add_precision_metrics(m, DEG_ge, DEG_ge_pred, 'DEG_ge_pred')

        return metrics_dict, metrics_dict_ls


class MVCModel_GatedPCC(MVCModel):
    """Task-aware gated multimodal fusion with PCC-oriented auxiliary objectives."""

    def __init__(self, n_genes, n_images, n_emd, n_latent, n_en_hidden, n_de_hidden, molecule_feature_dim, molecule_hidden, **kwargs):
        super().__init__(
            n_genes=n_genes,
            n_images=n_images,
            n_emd=n_emd,
            n_latent=n_latent,
            n_en_hidden=n_en_hidden,
            n_de_hidden=n_de_hidden,
            molecule_feature_dim=molecule_feature_dim,
            molecule_hidden=molecule_hidden,
            **kwargs,
        )

        self.modal_embed_dim = self.n_en_hidden[0]
        self.mol_embed_dim = self.molecule_hidden if self.molecule_hidden is not None else self.molecule_feature_dim
        self.fusion_input_dim = self.modal_embed_dim * 2 + self.mol_embed_dim
        self.fusion_hidden_dim = self.modal_embed_dim
        gate_hidden_dim = max(self.fusion_input_dim // 2, 32)
        interaction_input_dim = self.modal_embed_dim * 4

        self.cp_gate_network = nn.Sequential(
            nn.Linear(self.fusion_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.ge_gate_network = nn.Sequential(
            nn.Linear(self.fusion_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 3),
        )

        self.cp_base_fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_base_fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.cp_interaction_fusion = nn.Sequential(
            nn.Linear(interaction_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_interaction_fusion = nn.Sequential(
            nn.Linear(interaction_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_residual_fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.cp_task_fusion = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim * 2, self.n_latent),
            nn.LayerNorm(self.n_latent),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_direct_fusion = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.n_latent),
            nn.LayerNorm(self.n_latent),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_task_fusion = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim * 2, self.n_latent),
            nn.LayerNorm(self.n_latent),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.pcc_weight = kwargs.get('pcc_weight', 0.2)
        self.delta_pcc_weight = kwargs.get('delta_pcc_weight', 0.1)
        self.mse_cp_weight = kwargs.get('mse_cp_weight', 1.0)
        self.mse_ge_weight = kwargs.get('mse_ge_weight', 1.0)

        if self.init_w:
            self.cp_gate_network.apply(self._init_weights)
            self.ge_gate_network.apply(self._init_weights)
            self.cp_base_fusion.apply(self._init_weights)
            self.ge_base_fusion.apply(self._init_weights)
            self.cp_interaction_fusion.apply(self._init_weights)
            self.ge_interaction_fusion.apply(self._init_weights)
            self.ge_residual_fusion.apply(self._init_weights)
            self.cp_task_fusion.apply(self._init_weights)
            self.ge_direct_fusion.apply(self._init_weights)
            self.ge_task_fusion.apply(self._init_weights)

    @staticmethod
    def pearson_loss(pred, target):
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = (
            torch.sqrt((pred_centered ** 2).sum(dim=1)) *
            torch.sqrt((target_centered ** 2).sum(dim=1)) + 1e-8
        )
        return 1.0 - (numerator / denominator).mean()

    def _compute_task_gates(self, z_cp, z_ge, feat_embed):
        modal_context = torch.cat([z_cp, z_ge, feat_embed], dim=1)
        cp_gate = torch.softmax(self.cp_gate_network(modal_context), dim=1)
        ge_gate = torch.softmax(self.ge_gate_network(modal_context), dim=1)
        return cp_gate, ge_gate

    def _apply_gate(self, gate_weights, z_cp, z_ge, feat_embed):
        return torch.cat(
            [
                gate_weights[:, 0:1] * z_cp,
                gate_weights[:, 1:2] * z_ge,
                gate_weights[:, 2:3] * feat_embed,
            ],
            dim=1,
        )

    def _build_pairwise_interactions(self, z_cp, z_ge, feat_embed):
        return torch.cat(
            [
                z_cp * z_ge,
                z_cp * feat_embed,
                z_ge * feat_embed,
                torch.abs(z_cp - z_ge),
            ],
            dim=1,
        )

    def _build_task_context(self, cp_gate, ge_gate, z_cp, z_ge, feat_embed):
        cp_gated = self._apply_gate(cp_gate, z_cp, z_ge, feat_embed)
        ge_gated = self._apply_gate(ge_gate, z_cp, z_ge, feat_embed)
        pairwise = self._build_pairwise_interactions(z_cp, z_ge, feat_embed)

        cp_context = torch.cat(
            [self.cp_base_fusion(cp_gated), self.cp_interaction_fusion(pairwise)],
            dim=1,
        )
        ge_context = torch.cat(
            [self.ge_base_fusion(ge_gated), self.ge_residual_fusion(ge_gated)],
            dim=1,
        )
        return cp_context, ge_context

    def forward(self, x_cp, x_ge, features):
        if torch.isnan(x_cp).any() or torch.isnan(x_ge).any() or torch.isnan(features).any():
            print("⚠️ 输入里有 NaN")

        if self.molecule_hidden is not None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features

        z_cp = self.encoder_cp(x_cp)
        z_ge = self.encoder_ge(x_ge)
        cp_gate, ge_gate = self._compute_task_gates(z_cp, z_ge, feat_embed)

        cp_fused = self._apply_gate(cp_gate, z_cp, z_ge, feat_embed)
        ge_fused = self._apply_gate(ge_gate, z_cp, z_ge, feat_embed)
        cp_context, ge_context = self._build_task_context(cp_gate, ge_gate, z_cp, z_ge, feat_embed)

        cp_latent = self.cp_task_fusion(cp_context)
        ge_latent = self.ge_direct_fusion(ge_fused)

        cp_pred = self.decoder_cp(cp_latent)
        ge_pred = self.decoder_ge(ge_latent)
        return cp_pred, ge_pred

    def loss(
        self,
        target_cp,
        target_ge,
        cp_pred,
        ge_pred,
        control_cp=None,
        control_ge=None,
        features=None,
        mol_ids=None,
        weight_cp=1.0,
        weight_ge=1.0,
    ):
        mse_cp = F.mse_loss(cp_pred, target_cp, reduction="mean")
        mse_ge = F.mse_loss(ge_pred, target_ge, reduction="mean")

        total_loss = self.mse_cp_weight * mse_cp + self.mse_ge_weight * mse_ge

        if self.pcc_weight != 0.0:
            total_loss = total_loss + self.pcc_weight * (
                self.pearson_loss(cp_pred, target_cp) + self.pearson_loss(ge_pred, target_ge)
            )

        if self.delta_pcc_weight != 0.0 and control_cp is not None and control_ge is not None:
            total_loss = total_loss + self.delta_pcc_weight * (
                self.pearson_loss(cp_pred - control_cp, target_cp - control_cp)
                + self.pearson_loss(ge_pred - control_ge, target_ge - control_ge)
            )

        total_loss = total_loss + self._compute_dose_ordinal_aux_loss(
            ge_pred=ge_pred,
            target_ge=target_ge,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        return total_loss

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        """Train MVCModel_GatedPCC with control-aware auxiliary losses."""
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ['loss']

        best_value = np.inf
        best_epoch = 0
        for epoch in range(n_epochs):
            self.train()
            for control_cp, control_ge, target_cp, target_ge, features, mol_id in train_loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                features = features.to(self.dev)
                if control_cp.shape[0] == 1:
                    continue

                optimizer.zero_grad()
                cp_pred, ge_pred = self.forward(control_cp, control_ge, features)
                loss = self.loss(
                    target_cp,
                    target_ge,
                    cp_pred,
                    ge_pred,
                    control_cp=control_cp,
                    control_ge=control_ge,
                    features=features,
                    mol_ids=mol_id,
                )
                loss.backward()
                optimizer.step()

            train_dict, train_metrics_dict, _ = self.test_model(
                loader=train_loader,
                loss_item=loss_item,
                metrics_func=metrics_func,
            )
            train_loss = train_dict['loss']

            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)
            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            test_dict, test_metrics_dict, _ = self.test_model(
                loader=test_loader,
                loss_item=loss_item,
                metrics_func=metrics_func,
            )
            test_loss = test_dict['loss']

            for k, v in test_dict.items():
                epoch_hist['valid_' + k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)

            if epoch % 10 == 0:
                print('[Epoch %d] | train loss: %.5f, valid_loss: %.5f' % (epoch, train_loss, test_loss), flush=True)

            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None, perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        """Evaluate MVCModel_GatedPCC with control-aware auxiliary losses."""
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                mol_features = mol_features.to(self.dev)
                test_size += control_cp.shape[0]

                cp_pred, ge_pred = self.forward(control_cp, control_ge, mol_features)
                loss_value = self.loss(
                    target_cp,
                    target_ge,
                    cp_pred,
                    ge_pred,
                    control_cp=control_cp,
                    control_ge=control_ge,
                    features=mol_features,
                    mol_ids=mol_id,
                )

                if loss_item is not None:
                    for k in loss_item:
                        test_dict[k] += loss_value.item() * control_cp.shape[0]

                if metrics_func is not None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(
                        control_cp,
                        control_ge,
                        target_cp,
                        target_ge,
                        cp_pred,
                        ge_pred,
                        metrics_func=metrics_func,
                        perturbed_centroid_CP=perturbed_centroid_CP,
                        perturbed_centroid_GE=perturbed_centroid_GE,
                    )
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls['cp_id'] += list(mol_id)

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size

        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls


class MVCModel_MaskCPInput(MVCModel):
    """Fair ablation: zero the CP input while keeping the dual-head prediction task unchanged."""

    def forward(self, x_cp, x_ge, features):
        masked_cp = torch.zeros_like(x_cp)
        return super().forward(masked_cp, x_ge, features)


class MVCModel_MaskGEInput(MVCModel):
    """Fair ablation: zero the GE input while keeping the dual-head prediction task unchanged."""

    def forward(self, x_cp, x_ge, features):
        masked_ge = torch.zeros_like(x_ge)
        return super().forward(x_cp, masked_ge, features)


class MVCModel_GatedPCC_MaskCPInput(MVCModel_GatedPCC):
    """Fair ablation for gated fusion: zero the CP input while keeping the dual-head task unchanged."""

    def forward(self, x_cp, x_ge, features):
        masked_cp = torch.zeros_like(x_cp)
        return super().forward(masked_cp, x_ge, features)


class MVCModel_GatedPCC_MaskGEInput(MVCModel_GatedPCC):
    """Fair ablation for gated fusion: zero the GE input while keeping the dual-head task unchanged."""

    def forward(self, x_cp, x_ge, features):
        masked_ge = torch.zeros_like(x_ge)
        return super().forward(x_cp, masked_ge, features)


class MVCModel_NoCP(torch.nn.Module):
    """消融模型：完全去掉CP模态，只预测GE模态"""
    def __init__(self, n_genes, n_images, n_emd, n_latent, n_en_hidden, n_de_hidden, molecule_feature_dim, molecule_hidden, **kwargs):
        """
        Initialize MVCModel_NoCP model (without CP modality).
        :param n_genes: Number of genes in the dataset.
        :param n_images: Number of images in the dataset (not used in this ablation model).
        :param n_emd: Embedding dimension for the input data.
        :param n_latent: Latent space dimension.
        :param n_en_hidden: List of hidden layer sizes for the encoder.
        :param n_de_hidden: List of hidden layer sizes for the decoder.
        :param molecule_feature_dim: Dimension of the molecule features.
        :param molecule_hidden: Hidden dimension for molecule features.
        :param kwargs: Additional keyword arguments.
        """
        super(MVCModel_NoCP, self).__init__()
        self.n_genes = n_genes
        self.n_images = n_images
        self.out_genes = n_genes
        self.n_emb = n_emd
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.molecule_feature_dim = molecule_feature_dim
        self.molecule_hidden = molecule_hidden
        self.init_w = kwargs.get('init_w', False)
        self.path_model = kwargs.get('path_model', 'trained_model')
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.2)
        self.random_seed = kwargs.get('random_seed', 1234)
        
        # 只保留GE编码器
        encoder_ge = [
            nn.Linear(self.n_genes, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        self.encoder_ge = nn.Sequential(*encoder_ge)
        
        # 只保留GE解码器
        decoder_ge = [
                nn.Linear(self.n_en_hidden[0] + self.molecule_hidden, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout)
                ]
        decoder_ge.append(nn.Linear(self.n_de_hidden[-1], self.out_genes))
        
        self.decoder_ge = nn.Sequential(*decoder_ge)
        
        if self.molecule_hidden != None:
            feat_embeddings = [
                                nn.Linear(self.molecule_feature_dim, self.molecule_hidden),
                                nn.BatchNorm1d(self.molecule_hidden),
                                nn.ReLU(),
                                nn.Dropout(self.dropout)
                                ]
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

        if self.init_w:
            self.encoder_ge.apply(self._init_weights)
            self.decoder_ge.apply(self._init_weights)

    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    @staticmethod
    def _move_batch_to_device(control_cp, control_ge, target_cp, target_ge, features, device):
        control_ge = control_ge.to(device)
        target_ge = target_ge.to(device)
        features = features.to(device)
        return control_ge, target_ge, features

    def forward(self, x_cp, x_ge, features):
        """ Forward pass through network without CP modality"""
        if torch.isnan(x_ge).any() or torch.isnan(features).any():
            print("⚠️ 输入里有 NaN")

        if self.molecule_hidden != None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features
        
        # 只使用GE模态和分子特征
        z_ge = self.encoder_ge(x_ge)
        z1_feat = torch.cat([z_ge, feat_embed], dim=1)
        ge_pred = self.decoder_ge(z1_feat)
        
        return ge_pred

    def loss(self, target_ge, ge_pred, weight_ge=1.0):
        mse_ge = F.mse_loss(target_ge, ge_pred, reduction="sum")
        return mse_ge
    
    def get_feat_embdding(self, features):
        feat_embed = self.feat_embeddings(features)
        return feat_embed

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        """ Train MVCModel_NoCP """
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ['loss']

        # Train
        best_value = np.inf
        best_epoch = 0
        for epoch in range(n_epochs):
            train_size = 0
            loss_value = 0
            self.train()
            for control_cp, control_ge, target_cp, target_ge, features, _ in train_loader:
                control_ge, target_ge, features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, features, self.dev
                )
                if control_cp.shape[0] == 1:
                    continue
                train_size += control_cp.shape[0]
                optimizer.zero_grad()

                ge_pred = self.forward(control_cp, control_ge, features)
                
                loss = self.loss(target_ge, ge_pred)

                loss_value += loss.item()
                loss.backward()
                optimizer.step()

            # Eval
            train_dict, train_metrics_dict, train_metrics_dict_ls= self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict['loss']

            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)

            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            test_dict, test_metrics_dict, test_metricsdict_ls= self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_loss = test_dict['loss']

            for k, v in test_dict.items():
                epoch_hist['valid_'+k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)
            if epoch % 10 == 0:
                print('[Epoch %d] | train loss: %.5f, valid_loss: %.5f' % (epoch, train_loss, test_loss), flush=True)

            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model_NoCP.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None, perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        """Test model on input loader."""
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_ge, target_ge, mol_features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, mol_features, self.dev
                )
                test_size += control_cp.shape[0]
                
                ge_pred = self.forward(control_cp, control_ge, mol_features)
                loss_ls = self.loss(target_ge, ge_pred)
                
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls.item()

                if metrics_func != None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(control_ge, target_ge, ge_pred, \
                        metrics_func=metrics_func, perturbed_centroid_GE=perturbed_centroid_GE)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls['cp_id'] += list(mol_id)

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size

        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls

    def predict_profile(self, loader):
        """predict profiles."""
        self.eval()
        with torch.no_grad():
            control_ge_list = []
            target_ge_list = []
            ge_pred_list = []
            smiles_list = []

            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_ge, target_ge, mol_features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, mol_features, self.dev
                )

                ge_pred = self.forward(control_cp, control_ge, mol_features)

                control_ge_list.append(control_ge.cpu().numpy().astype(float))
                target_ge_list.append(target_ge.cpu().numpy().astype(float))
                ge_pred_list.append(ge_pred.cpu().numpy().astype(float))
                smiles_list.extend(list(mol_id))

        control_ge_array = np.concatenate(control_ge_list, axis=0)
        target_ge_array = np.concatenate(target_ge_list, axis=0)
        ge_pred_array = np.concatenate(ge_pred_list, axis=0)
        smiles_array = np.array(smiles_list)

        return control_ge_array, target_ge_array, ge_pred_array, smiles_array

    def eval_x_reconstruction(self, control_ge, target_ge, ge_pred, metrics_func=['pearson'], perturbed_centroid_GE=None):
        """Compute reconstruction evaluation metrics for GE prediction（批量/并行优化版）"""
        use_torch = ge_pred.is_cuda
        ge_true = target_ge.float()
        ge_pred_ = ge_pred.float()
        cge = control_ge.float()
        device = ge_true.device

        if perturbed_centroid_GE is None:
            perturbed_centroid_GE = ge_true.mean(dim=0)
        else:
            if use_torch:
                perturbed_centroid_GE = torch.tensor(perturbed_centroid_GE, dtype=ge_true.dtype, device=device)
            else:
                perturbed_centroid_GE = np.asarray(perturbed_centroid_GE, float)

        DEG_ge = ge_true - cge
        DEG_ge_pred = ge_pred_ - cge
        Delta_sys_GE_true = ge_true - perturbed_centroid_GE
        Delta_sys_GE_pred = ge_pred_ - perturbed_centroid_GE

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)

        def batch_pearson(x, y):
            x0 = x - x.mean(dim=1, keepdim=True)
            y0 = y - y.mean(dim=1, keepdim=True)
            num = (x0 * y0).sum(dim=1)
            den = torch.norm(x0, dim=1) * torch.norm(y0, dim=1) + 1e-8
            v = num / den
            return v.cpu().numpy() if use_torch else v

        def batch_rmse(x, y):
            se = ((x - y) ** 2).mean(dim=1)
            v = torch.sqrt(se)
            return v.cpu().numpy() if use_torch else np.sqrt(se)

        def batch_precision_at_k(x_true, x_pred, k=100, pos_num=100, neg_num=100):
            if use_torch:
                topk_pred = torch.topk(x_pred, k, dim=1).indices
                top_pos_true = torch.topk(x_true, pos_num, dim=1).indices
                top_neg_true = torch.topk(-x_true, neg_num, dim=1).indices
                pred_mask = torch.zeros((*x_pred.shape[:1], x_pred.shape[1]), dtype=torch.bool, device=x_pred.device)
                pos_mask = torch.zeros_like(pred_mask)
                neg_mask = torch.zeros_like(pred_mask)
                pred_mask.scatter_(1, topk_pred, True)
                pos_mask.scatter_(1, top_pos_true, True)
                neg_mask.scatter_(1, top_neg_true, True)
                pos_hits = (pred_mask & pos_mask).sum(dim=1)
                neg_hits = (pred_mask & neg_mask).sum(dim=1)
                return (neg_hits.float() / k, pos_hits.float() / k)
            else:
                x_true = x_true.cpu().numpy() if hasattr(x_true, 'cpu') else x_true
                x_pred = x_pred.cpu().numpy() if hasattr(x_pred, 'cpu') else x_pred
                N, D = x_true.shape
                topk_pred = np.argpartition(x_pred, -k, axis=1)[:, -k:]
                top_pos_true = np.argpartition(x_true, -pos_num, axis=1)[:, -pos_num:]
                top_neg_true = np.argpartition(x_true, neg_num, axis=1)[:, :neg_num]
                pred_mask = np.zeros((N, D), dtype=bool)
                pred_mask[np.arange(N)[:, None], topk_pred] = True
                pos_mask = np.zeros((N, D), dtype=bool)
                pos_mask[np.arange(N)[:, None], top_pos_true] = True
                neg_mask = np.zeros((N, D), dtype=bool)
                neg_mask[np.arange(N)[:, None], top_neg_true] = True
                pos_hits = (pred_mask & pos_mask).sum(axis=1)
                neg_hits = (pred_mask & neg_mask).sum(axis=1)
                return (neg_hits / k, pos_hits / k)

        if 'pearson' in metrics_func:
            p_ge_sys = batch_pearson(Delta_sys_GE_true, Delta_sys_GE_pred)
            metrics_dict['systema_pearson_GE'] = float(np.nansum(p_ge_sys))
            metrics_dict_ls['systema_pearson_GE'] = p_ge_sys.tolist()
        if 'rmse' in metrics_func:
            r_ge_sys = batch_rmse(Delta_sys_GE_true, Delta_sys_GE_pred)
            metrics_dict['systema_rmse_GE'] = float(np.nansum(r_ge_sys))
            metrics_dict_ls['systema_rmse_GE'] = r_ge_sys.tolist()

        if 'pearson' in metrics_func:
            ge_pred_pcc = batch_pearson(ge_true, ge_pred_)
            metrics_dict_ls['ge_pred_pearson'] = ge_pred_pcc.tolist()
            deg_ge_pred_pcc = batch_pearson(DEG_ge, DEG_ge_pred)
            metrics_dict_ls['DEG_ge_pred_pearson'] = deg_ge_pred_pcc.tolist()
        if 'rmse' in metrics_func:
            ge_pred_rmse = batch_rmse(ge_true, ge_pred_)
            metrics_dict_ls['ge_pred_rmse'] = ge_pred_rmse.tolist()
            deg_ge_pred_rmse = batch_rmse(DEG_ge, DEG_ge_pred)
            metrics_dict_ls['DEG_ge_pred_rmse'] = deg_ge_pred_rmse.tolist()

        for m in metrics_func:
            if m.startswith('precision'):
                k = int(m[len('precision'):])
                neg, pos = batch_precision_at_k(ge_true, ge_pred_, k)
                if use_torch:
                    metrics_dict['ge_pred_neg_%s' % m] = float(neg.sum().item())
                    metrics_dict['ge_pred_pos_%s' % m] = float(pos.sum().item())
                    metrics_dict_ls['ge_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['ge_pred_pos_%s' % m] = pos.tolist()
                else:
                    metrics_dict['ge_pred_neg_%s' % m] = float(neg.sum())
                    metrics_dict['ge_pred_pos_%s' % m] = float(pos.sum())
                    metrics_dict_ls['ge_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['ge_pred_pos_%s' % m] = pos.tolist()

        for m in metrics_func:
            if m.startswith('precision'):
                k = int(m[len('precision'):])
                neg, pos = batch_precision_at_k(DEG_ge, DEG_ge_pred, k)
                if use_torch:
                    metrics_dict['DEG_ge_pred_neg_%s' % m] = float(neg.sum().item())
                    metrics_dict['DEG_ge_pred_pos_%s' % m] = float(pos.sum().item())
                    metrics_dict_ls['DEG_ge_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['DEG_ge_pred_pos_%s' % m] = pos.tolist()
                else:
                    metrics_dict['DEG_ge_pred_neg_%s' % m] = float(neg.sum())
                    metrics_dict['DEG_ge_pred_pos_%s' % m] = float(pos.sum())
                    metrics_dict_ls['DEG_ge_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['DEG_ge_pred_pos_%s' % m] = pos.tolist()

        return metrics_dict, metrics_dict_ls


class MVCModel_NoGE(torch.nn.Module):
    """消融模型：完全去掉GE模态，只预测CP模态"""
    def __init__(self, n_genes, n_images, n_emd, n_latent, n_en_hidden, n_de_hidden, molecule_feature_dim, molecule_hidden, **kwargs):
        """
        Initialize MVCModel_NoGE model (without GE modality).
        :param n_genes: Number of genes in the dataset (not used in this ablation model).
        :param n_images: Number of images in the dataset.
        :param n_emd: Embedding dimension for the input data.
        :param n_latent: Latent space dimension.
        :param n_en_hidden: List of hidden layer sizes for the encoder.
        :param n_de_hidden: List of hidden layer sizes for the decoder.
        :param molecule_feature_dim: Dimension of the molecule features.
        :param molecule_hidden: Hidden dimension for molecule features.
        :param kwargs: Additional keyword arguments.
        """
        super(MVCModel_NoGE, self).__init__()
        self.n_genes = n_genes
        self.n_images = n_images
        self.out_images = n_images
        self.n_emb = n_emd
        self.n_latent = n_latent
        self.n_en_hidden = n_en_hidden
        self.n_de_hidden = n_de_hidden
        self.molecule_feature_dim = molecule_feature_dim
        self.molecule_hidden = molecule_hidden
        self.init_w = kwargs.get('init_w', False)
        self.path_model = kwargs.get('path_model', 'trained_model')
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.dropout = kwargs.get('dropout', 0.2)
        self.random_seed = kwargs.get('random_seed', 1234)
        
        # 只保留CP编码器
        encoder_cp = [
            nn.Linear(self.n_images, self.n_en_hidden[0]),
            nn.BatchNorm1d(self.n_en_hidden[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        self.encoder_cp = nn.Sequential(*encoder_cp)
        
        # 只保留CP解码器
        decoder_cp = [
                nn.Linear(self.n_en_hidden[0] + self.molecule_hidden, self.n_de_hidden[0]),
                nn.BatchNorm1d(self.n_de_hidden[0]),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout)
                ]
        decoder_cp.append(nn.Linear(self.n_de_hidden[-1], self.out_images))
        
        self.decoder_cp = nn.Sequential(*decoder_cp)
        
        if self.molecule_hidden != None:
            feat_embeddings = [
                                nn.Linear(self.molecule_feature_dim, self.molecule_hidden),
                                nn.BatchNorm1d(self.molecule_hidden),
                                nn.ReLU(),
                                nn.Dropout(self.dropout)
                                ]
            self.feat_embeddings = nn.Sequential(*feat_embeddings)

        if self.init_w:
            self.encoder_cp.apply(self._init_weights)
            self.decoder_cp.apply(self._init_weights)

    def _init_weights(self, layer):
        """ Initialize weights of layer with Xavier uniform"""
        if type(layer)==nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
        return

    @staticmethod
    def _move_batch_to_device(control_cp, control_ge, target_cp, target_ge, features, device):
        control_cp = control_cp.to(device)
        target_cp = target_cp.to(device)
        features = features.to(device)
        return control_cp, target_cp, features

    def forward(self, x_cp, x_ge, features):
        """ Forward pass through network without GE modality"""
        if torch.isnan(x_cp).any() or torch.isnan(features).any():
            print("⚠️ 输入里有 NaN")

        if self.molecule_hidden != None:
            feat_embed = self.feat_embeddings(features)
        else:
            feat_embed = features
        
        # 只使用CP模态和分子特征
        z_cp = self.encoder_cp(x_cp)
        z1_feat = torch.cat([z_cp, feat_embed], dim=1)
        cp_pred = self.decoder_cp(z1_feat)
        
        return cp_pred

    def loss(self, target_cp, cp_pred, weight_cp=1.0):
        mse_cp = F.mse_loss(target_cp, cp_pred, reduction="sum")
        return mse_cp
    
    def get_feat_embdding(self, features):
        feat_embed = self.feat_embeddings(features)
        return feat_embed

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        """ Train MVCModel_NoGE """
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ['loss']

        # Train
        best_value = np.inf
        best_epoch = 0
        for epoch in range(n_epochs):
            train_size = 0
            loss_value = 0
            self.train()
            for control_cp, control_ge, target_cp, target_ge, features, _ in train_loader:
                control_cp, target_cp, features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, features, self.dev
                )
                if control_cp.shape[0] == 1:
                    continue
                train_size += control_cp.shape[0]
                optimizer.zero_grad()

                cp_pred = self.forward(control_cp, control_ge, features)
                
                loss = self.loss(target_cp, cp_pred)

                loss_value += loss.item()
                loss.backward()
                optimizer.step()

            # Eval
            train_dict, train_metrics_dict, train_metrics_dict_ls= self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict['loss']

            for k, v in train_dict.items():
                epoch_hist['train_' + k].append(v)

            for k, v in train_metrics_dict.items():
                epoch_hist['train_' + k].append(v)

            test_dict, test_metrics_dict, test_metricsdict_ls= self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_loss = test_dict['loss']

            for k, v in test_dict.items():
                epoch_hist['valid_'+k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist['valid_' + k].append(v)
            if epoch % 10 == 0:
                print('[Epoch %d] | train loss: %.5f, valid_loss: %.5f' % (epoch, train_loss, test_loss), flush=True)

            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + 'best_model_NoGE.pt')

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None, perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        """Test model on input loader."""
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp, target_cp, mol_features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, mol_features, self.dev
                )
                test_size += control_cp.shape[0]
                
                cp_pred = self.forward(control_cp, control_ge, mol_features)
                loss_ls = self.loss(target_cp, cp_pred)
                
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls.item()

                if metrics_func != None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(control_cp, target_cp, cp_pred, \
                        metrics_func=metrics_func, perturbed_centroid_CP=perturbed_centroid_CP)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls['cp_id'] += list(mol_id)

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size

        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls

    def predict_profile(self, loader):
        """predict profiles."""
        self.eval()
        with torch.no_grad():
            control_cp_list = []
            target_cp_list = []
            cp_pred_list = []
            smiles_list = []

            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp, target_cp, mol_features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, mol_features, self.dev
                )

                cp_pred = self.forward(control_cp, control_ge, mol_features)

                control_cp_list.append(control_cp.cpu().numpy().astype(float))
                target_cp_list.append(target_cp.cpu().numpy().astype(float))
                cp_pred_list.append(cp_pred.cpu().numpy().astype(float))
                smiles_list.extend(list(mol_id))

        control_cp_array = np.concatenate(control_cp_list, axis=0)
        target_cp_array = np.concatenate(target_cp_list, axis=0)
        cp_pred_array = np.concatenate(cp_pred_list, axis=0)
        smiles_array = np.array(smiles_list)

        return control_cp_array, target_cp_array, cp_pred_array, smiles_array

    def eval_x_reconstruction(self, control_cp, target_cp, cp_pred, metrics_func=['pearson'], perturbed_centroid_CP=None):
        """Compute reconstruction evaluation metrics for CP prediction（批量/并行优化版）"""
        use_torch = cp_pred.is_cuda
        cp_true = target_cp.float()
        cp_pred_ = cp_pred.float()
        ccp = control_cp.float()
        device = cp_true.device

        if perturbed_centroid_CP is None:
            perturbed_centroid_CP = cp_true.mean(dim=0)
        else:
            if use_torch:
                perturbed_centroid_CP = torch.tensor(perturbed_centroid_CP, dtype=cp_true.dtype, device=device)
            else:
                perturbed_centroid_CP = np.asarray(perturbed_centroid_CP, float)

        DEG_cp = cp_true - ccp
        DEG_cp_pred = cp_pred_ - ccp
        Delta_sys_CP_true = cp_true - perturbed_centroid_CP
        Delta_sys_CP_pred = cp_pred_ - perturbed_centroid_CP

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)

        def batch_pearson(x, y):
            x0 = x - x.mean(dim=1, keepdim=True)
            y0 = y - y.mean(dim=1, keepdim=True)
            num = (x0 * y0).sum(dim=1)
            den = torch.norm(x0, dim=1) * torch.norm(y0, dim=1) + 1e-8
            v = num / den
            return v.cpu().numpy() if use_torch else v

        def batch_rmse(x, y):
            se = ((x - y) ** 2).mean(dim=1)
            v = torch.sqrt(se)
            return v.cpu().numpy() if use_torch else np.sqrt(se)

        def batch_precision_at_k(x_true, x_pred, k=100, pos_num=100, neg_num=100):
            if use_torch:
                topk_pred = torch.topk(x_pred, k, dim=1).indices
                top_pos_true = torch.topk(x_true, pos_num, dim=1).indices
                top_neg_true = torch.topk(-x_true, neg_num, dim=1).indices
                pred_mask = torch.zeros((*x_pred.shape[:1], x_pred.shape[1]), dtype=torch.bool, device=x_pred.device)
                pos_mask = torch.zeros_like(pred_mask)
                neg_mask = torch.zeros_like(pred_mask)
                pred_mask.scatter_(1, topk_pred, True)
                pos_mask.scatter_(1, top_pos_true, True)
                neg_mask.scatter_(1, top_neg_true, True)
                pos_hits = (pred_mask & pos_mask).sum(dim=1)
                neg_hits = (pred_mask & neg_mask).sum(dim=1)
                return (neg_hits.float() / k, pos_hits.float() / k)
            else:
                x_true = x_true.cpu().numpy() if hasattr(x_true, 'cpu') else x_true
                x_pred = x_pred.cpu().numpy() if hasattr(x_pred, 'cpu') else x_pred
                N, D = x_true.shape
                topk_pred = np.argpartition(x_pred, -k, axis=1)[:, -k:]
                top_pos_true = np.argpartition(x_true, -pos_num, axis=1)[:, -pos_num:]
                top_neg_true = np.argpartition(x_true, neg_num, axis=1)[:, :neg_num]
                pred_mask = np.zeros((N, D), dtype=bool)
                pred_mask[np.arange(N)[:, None], topk_pred] = True
                pos_mask = np.zeros((N, D), dtype=bool)
                pos_mask[np.arange(N)[:, None], top_pos_true] = True
                neg_mask = np.zeros((N, D), dtype=bool)
                neg_mask[np.arange(N)[:, None], top_neg_true] = True
                pos_hits = (pred_mask & pos_mask).sum(axis=1)
                neg_hits = (pred_mask & neg_mask).sum(axis=1)
                return (neg_hits / k, pos_hits / k)

        if 'pearson' in metrics_func:
            p_cp_sys = batch_pearson(Delta_sys_CP_true, Delta_sys_CP_pred)
            metrics_dict['systema_pearson_CP'] = float(np.nansum(p_cp_sys))
            metrics_dict_ls['systema_pearson_CP'] = p_cp_sys.tolist()
        if 'rmse' in metrics_func:
            r_cp_sys = batch_rmse(Delta_sys_CP_true, Delta_sys_CP_pred)
            metrics_dict['systema_rmse_CP'] = float(np.nansum(r_cp_sys))
            metrics_dict_ls['systema_rmse_CP'] = r_cp_sys.tolist()

        if 'pearson' in metrics_func:
            cp_pred_pcc = batch_pearson(cp_true, cp_pred_)
            metrics_dict_ls['cp_pred_pearson'] = cp_pred_pcc.tolist()
            deg_cp_pred_pcc = batch_pearson(DEG_cp, DEG_cp_pred)
            metrics_dict_ls['DEG_cp_pred_pearson'] = deg_cp_pred_pcc.tolist()
        if 'rmse' in metrics_func:
            cp_pred_rmse = batch_rmse(cp_true, cp_pred_)
            metrics_dict_ls['cp_pred_rmse'] = cp_pred_rmse.tolist()
            deg_cp_pred_rmse = batch_rmse(DEG_cp, DEG_cp_pred)
            metrics_dict_ls['DEG_cp_pred_rmse'] = deg_cp_pred_rmse.tolist()

        for m in metrics_func:
            if m.startswith('precision'):
                k = int(m[len('precision'):])
                neg, pos = batch_precision_at_k(cp_true, cp_pred_, k)
                if use_torch:
                    metrics_dict['cp_pred_neg_%s' % m] = float(neg.sum().item())
                    metrics_dict['cp_pred_pos_%s' % m] = float(pos.sum().item())
                    metrics_dict_ls['cp_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['cp_pred_pos_%s' % m] = pos.tolist()
                else:
                    metrics_dict['cp_pred_neg_%s' % m] = float(neg.sum())
                    metrics_dict['cp_pred_pos_%s' % m] = float(pos.sum())
                    metrics_dict_ls['cp_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['cp_pred_pos_%s' % m] = pos.tolist()

        for m in metrics_func:
            if m.startswith('precision'):
                k = int(m[len('precision'):])
                neg, pos = batch_precision_at_k(DEG_cp, DEG_cp_pred, k)
                if use_torch:
                    metrics_dict['DEG_cp_pred_neg_%s' % m] = float(neg.sum().item())
                    metrics_dict['DEG_cp_pred_pos_%s' % m] = float(pos.sum().item())
                    metrics_dict_ls['DEG_cp_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['DEG_cp_pred_pos_%s' % m] = pos.tolist()
                else:
                    metrics_dict['DEG_cp_pred_neg_%s' % m] = float(neg.sum())
                    metrics_dict['DEG_cp_pred_pos_%s' % m] = float(pos.sum())
                    metrics_dict_ls['DEG_cp_pred_neg_%s' % m] = neg.tolist()
                    metrics_dict_ls['DEG_cp_pred_pos_%s' % m] = pos.tolist()

        return metrics_dict, metrics_dict_ls
