#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch import nn, optim
import copy
from collections import defaultdict
import numpy as np
from utils import *

# ===================== 兼容层 (保持不变) =====================
class HNNConv(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, *args, **kwargs):
        super().__init__()
        if in_dim is not None and out_dim is not None:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.linear = nn.LazyLinear(out_features=out_dim or 1)
        self.activation = nn.GELU()

    def forward(self, x, *args, **kwargs):
        return self.activation(self.linear(x))


# --- 超图特征修正模块 (保持不变) ---
class HypergraphRefinementBlock(nn.Module):
    def __init__(self, embed_dim, context_dim, num_hyperedges=8, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.base_hyperedges = nn.Parameter(torch.randn(1, num_hyperedges, embed_dim))
        self.edge_generator = nn.Sequential(
            nn.Linear(context_dim, embed_dim * num_hyperedges),
            nn.LayerNorm(embed_dim * num_hyperedges),
            nn.GELU(),
        )
        self.attn_n2e = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_e = nn.LayerNorm(embed_dim)
        self.attn_e2n = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_update = nn.LayerNorm(embed_dim)
        self.gate = nn.Parameter(torch.tensor([-4.0]))
        self.edge_gate = nn.Parameter(torch.tensor([-2.0]))

    def build_hyperedges(self, context):
        batch_size = context.shape[0]
        dynamic_edges = self.edge_generator(context).view(batch_size, self.num_hyperedges, -1)
        dynamic_scale = torch.sigmoid(self.edge_gate).view(1, 1, 1)
        return self.base_hyperedges.expand(batch_size, -1, -1) + dynamic_scale * dynamic_edges

    def forward(self, nodes, context):
        edges = self.build_hyperedges(context)
        e_update, _ = self.attn_n2e(query=edges, key=nodes, value=nodes)
        edges = self.norm_e(edges + e_update)
        n_update, _ = self.attn_e2n(query=nodes, key=edges, value=edges)
        n_update = self.norm_update(n_update)
        alpha = torch.sigmoid(self.gate).view(1, 1, 1)
        nodes_refined = nodes + alpha * n_update
        return nodes_refined


class MVCModel_HyperGate(torch.nn.Module):
    """HyperGate gated fusion with normalized reconstruction and PCC auxiliaries."""
    def __init__(
        self,
        n_genes,
        n_images,
        n_emd,
        n_latent,
        n_en_hidden,
        n_de_hidden,
        molecule_feature_dim,
        molecule_hidden,
        **kwargs,
    ):
        super(MVCModel_HyperGate, self).__init__()
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
        self.init_w = kwargs.get("init_w", False)
        self.path_model = kwargs.get("path_model", "trained_model")
        self.dev = kwargs.get("device", torch.device("cpu"))
        self.dropout = kwargs.get("dropout", 0.2)
        self.use_dose_feature = bool(kwargs.get("use_dose_feature", False))
        self.dose_ordinal_weight = float(kwargs.get("dose_ordinal_weight", 0.0))
        self.dose_ordinal_temperature = float(kwargs.get("dose_ordinal_temperature", 0.2))
        self.dose_ordinal_margin_scale = float(kwargs.get("dose_ordinal_margin_scale", 0.25))
        
        # Keep these attributes for compatibility with older checkpoints and logs.
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.loss_amplification = kwargs.get("loss_amplification", None)
        self.grad_clip_norm = kwargs.get("grad_clip_norm", 5.0)
        shared_pcc_weight = kwargs.get("pcc_weight", 0.6)
        shared_delta_pcc_weight = kwargs.get("delta_pcc_weight", 0.2)
        self.cp_pcc_weight = shared_pcc_weight if kwargs.get("cp_pcc_weight") is None else kwargs.get("cp_pcc_weight")
        self.ge_pcc_weight = shared_pcc_weight if kwargs.get("ge_pcc_weight") is None else kwargs.get("ge_pcc_weight")
        self.cp_delta_pcc_weight = shared_delta_pcc_weight if kwargs.get("cp_delta_pcc_weight") is None else kwargs.get("cp_delta_pcc_weight")
        self.ge_delta_pcc_weight = shared_delta_pcc_weight if kwargs.get("ge_delta_pcc_weight") is None else kwargs.get("ge_delta_pcc_weight")

        self.embed_dim = self.n_en_hidden[0]
        self.modal_embed_dim = self.embed_dim
        self.mol_embed_dim = self.molecule_hidden if self.molecule_hidden else self.molecule_feature_dim
        # shared_fusion concatenates z_cp, z_ge, z_mol after mol_to_embed, so the runtime
        # dimension is always 3 * embed_dim rather than depending on molecule_hidden.
        self.fusion_input_dim = self.embed_dim * 3
        self.tokens_per_modality = kwargs.get("tokens_per_modality", 2)
        self.n_attn_heads = kwargs.get("n_attn_heads", 8)
        self.n_fusion_layers = kwargs.get("n_fusion_layers", 2)
        self.n_hyper_heads = kwargs.get("n_hyper_heads", self.n_attn_heads)

        # --- Encoders ---
        encoder_cp = [
            nn.Linear(self.n_images, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        ]
        encoder_ge = [
            nn.Linear(self.n_genes, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        ]
        self.encoder_cp = nn.Sequential(*encoder_cp)
        self.encoder_ge = nn.Sequential(*encoder_ge)

        if self.molecule_hidden is not None:
            self.feat_embeddings = nn.Sequential(
                nn.Linear(self.molecule_feature_dim, self.molecule_hidden),
                nn.BatchNorm1d(self.molecule_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
        mol_input_dim = self.mol_embed_dim
        self.mol_to_embed = nn.Linear(mol_input_dim, self.embed_dim)

        token_dim = self.embed_dim * self.tokens_per_modality
        self.cp_tokenizer = nn.Sequential(
            nn.Linear(self.embed_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.ge_tokenizer = nn.Sequential(
            nn.Linear(self.embed_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.mol_tokenizer = nn.Sequential(
            nn.Linear(self.embed_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

        # --- Hypergraph Refinement ---
        self.num_hyperedges = kwargs.get("num_hyperedges", 4)
        self.hyper_insert_after = kwargs.get("hyper_insert_after", 1)
        self.disable_hyper_refinement = bool(kwargs.get("disable_hyper_refinement", False))
        self.hyper_refinement = None
        if not self.disable_hyper_refinement:
            self.hyper_refinement = HypergraphRefinementBlock(
                self.embed_dim,
                context_dim=self.embed_dim * 3,
                num_hyperedges=self.num_hyperedges,
                num_heads=self.n_hyper_heads,
                dropout=self.dropout,
            )

        # --- Fusion Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_attn_heads,
            dropout=self.dropout,
            batch_first=True,
            dim_feedforward=self.embed_dim * 4,
        )
        self.fusion_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.n_fusion_layers)])

        gate_input_dim = self.fusion_input_dim * 2
        gate_hidden_dim = max(self.fusion_input_dim // 2, 32)
        self.cp_task_gate_network = nn.Sequential(
            nn.Linear(self.fusion_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.ge_task_gate_network = nn.Sequential(
            nn.Linear(self.fusion_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.cp_modal_gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.ge_modal_gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.cp_private_residual = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_input_dim),
            nn.LayerNorm(self.fusion_input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_input_dim, self.fusion_input_dim),
        )
        self.cp_cross_context_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.fusion_input_dim),
            nn.LayerNorm(self.fusion_input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_input_dim, self.fusion_input_dim),
        )
        self.cp_cross_context_gate_logit = nn.Parameter(torch.tensor([-3.0]))
        self.ge_private_residual = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_input_dim),
            nn.LayerNorm(self.fusion_input_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_input_dim, self.fusion_input_dim),
        )
        # Keep CP close to the strong gated baseline, while allowing GE to use a larger dynamic correction.
        self.cp_residual_gate_logit = nn.Parameter(torch.tensor([-6.0]))
        self.ge_residual_gate_logit = nn.Parameter(torch.tensor([-3.0]))
        self.cp_output_projection = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.n_latent),
            nn.LayerNorm(self.n_latent),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_output_projection = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.n_latent),
            nn.LayerNorm(self.n_latent),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # --- Decoders ---
        decoder_cp = [
            nn.Linear(self.n_latent, self.n_de_hidden[0]),
            nn.BatchNorm1d(self.n_de_hidden[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
        ]
        decoder_ge = [
            nn.Linear(self.n_latent, self.n_de_hidden[0]),
            nn.BatchNorm1d(self.n_de_hidden[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
        ]
        decoder_cp.append(nn.Linear(self.n_de_hidden[-1], self.out_images))
        decoder_ge.append(nn.Linear(self.n_de_hidden[-1], self.out_genes))

        self.decoder_cp = nn.Sequential(*decoder_cp)
        self.decoder_ge = nn.Sequential(*decoder_ge)

        if self.init_w:
            self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, nn.TransformerEncoderLayer):
            for name, param in layer.named_parameters():
                if "weight" in name and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.constant_(param, 0.0)

    def forward(self, x_cp, x_ge, features):
        z_cp = self.encoder_cp(x_cp)
        z_ge = self.encoder_ge(x_ge)

        if hasattr(self, "feat_embeddings"):
            feat_embed_raw = self.feat_embeddings(features)
        else:
            feat_embed_raw = features
        z_mol = self.mol_to_embed(feat_embed_raw)

        token_sequence = self._build_modal_token_sequence(z_cp, z_ge, z_mol)
        hyper_context = self._build_hypergraph_context(z_cp, z_ge, z_mol)

        x = token_sequence
        for li, layer in enumerate(self.fusion_layers, start=1):
            x = layer(x)
            if self.hyper_refinement is not None and li == self.hyper_insert_after:
                x = self.hyper_refinement(x, hyper_context)

        cp_tokens, ge_tokens, mol_tokens = self._split_modal_tokens(x)
        refined_cp = cp_tokens.mean(dim=1)
        refined_ge = ge_tokens.mean(dim=1)
        refined_mol = mol_tokens.mean(dim=1)
        shared_fusion = self._build_shared_fusion(z_cp, z_ge, z_mol)
        cp_gate, ge_gate = self._compute_task_gate_weights(shared_fusion)
        cp_shared_fusion = self._apply_task_gate(cp_gate, z_cp, z_ge, z_mol)
        ge_shared_fusion = self._apply_task_gate(ge_gate, z_cp, z_ge, z_mol)

        cp_fused, ge_fused = self.build_task_fusions(
            z_cp=z_cp,
            z_ge=z_ge,
            feat_embed=z_mol,
            refined_cp=refined_cp,
            refined_ge=refined_ge,
            refined_mol=refined_mol,
            cp_shared_fusion=cp_shared_fusion,
            ge_shared_fusion=ge_shared_fusion,
        )
        cp_latent = self.cp_output_projection(cp_fused)
        ge_latent = self.ge_output_projection(ge_fused)
        return self.decoder_cp(cp_latent), self.decoder_ge(ge_latent)

    def _build_shared_fusion(self, z_cp, z_ge, feat_embed):
        return torch.cat([z_cp, z_ge, feat_embed], dim=1)

    def _tokenize_modal_summary(self, summary, tokenizer):
        token_tensor = tokenizer(summary)
        return token_tensor.view(summary.shape[0], self.tokens_per_modality, self.embed_dim)

    def _build_modal_token_sequence(self, z_cp, z_ge, z_mol):
        cp_tokens = self._tokenize_modal_summary(z_cp, self.cp_tokenizer)
        ge_tokens = self._tokenize_modal_summary(z_ge, self.ge_tokenizer)
        mol_tokens = self._tokenize_modal_summary(z_mol, self.mol_tokenizer)
        return torch.cat([cp_tokens, ge_tokens, mol_tokens], dim=1)

    def _split_modal_tokens(self, token_sequence):
        cp_end = self.tokens_per_modality
        ge_end = cp_end + self.tokens_per_modality
        mol_end = ge_end + self.tokens_per_modality
        return (
            token_sequence[:, :cp_end, :],
            token_sequence[:, cp_end:ge_end, :],
            token_sequence[:, ge_end:mol_end, :],
        )

    def _build_hypergraph_context(self, z_cp, z_ge, z_mol):
        return torch.cat([z_cp, z_ge, z_mol], dim=1)

    def _compute_task_gate_weights(self, shared_fusion, refined_cp=None, refined_ge=None, refined_mol=None):
        cp_gate = torch.softmax(self.cp_task_gate_network(shared_fusion), dim=1)
        ge_gate = torch.softmax(self.ge_task_gate_network(shared_fusion), dim=1)
        return cp_gate, ge_gate

    def _apply_task_gate(self, gate_weights, z_cp, z_ge, feat_embed):
        return torch.cat(
            [
                gate_weights[:, 0:1] * z_cp,
                gate_weights[:, 1:2] * z_ge,
                gate_weights[:, 2:3] * feat_embed,
            ],
            dim=1,
        )

    def _compute_modal_gates(self, shared_fusion, refined_cp, refined_ge, refined_mol, gate_network):
        dynamic_context = torch.cat([refined_ge, refined_cp, refined_mol], dim=1)
        gate_input = torch.cat([shared_fusion, dynamic_context], dim=1)
        gate_weights = torch.softmax(gate_network(gate_input), dim=1)
        gated_dynamic = torch.cat(
            [
                gate_weights[:, 0:1] * refined_ge,
                gate_weights[:, 1:2] * refined_cp,
                gate_weights[:, 2:3] * refined_mol,
            ],
            dim=1,
        )
        return gated_dynamic

    def build_task_fusions(
        self,
        z_cp,
        z_ge,
        feat_embed,
        refined_cp,
        refined_ge,
        refined_mol,
        cp_shared_fusion=None,
        ge_shared_fusion=None,
    ):
        if cp_shared_fusion is None:
            cp_shared_fusion = self._build_shared_fusion(z_cp, z_ge, feat_embed)
        if ge_shared_fusion is None:
            ge_shared_fusion = self._build_shared_fusion(z_cp, z_ge, feat_embed)
        cp_dynamic = self._compute_modal_gates(
            shared_fusion=cp_shared_fusion,
            refined_cp=refined_cp,
            refined_ge=refined_ge,
            refined_mol=refined_mol,
            gate_network=self.cp_modal_gate_network,
        )
        ge_dynamic = self._compute_modal_gates(
            shared_fusion=ge_shared_fusion,
            refined_cp=refined_cp,
            refined_ge=refined_ge,
            refined_mol=refined_mol,
            gate_network=self.ge_modal_gate_network,
        )
        cp_cross_context = torch.cat([refined_ge, refined_mol], dim=1)
        cp_cross_scale = torch.sigmoid(self.cp_cross_context_gate_logit).view(1, 1)
        cp_residual_scale = torch.sigmoid(self.cp_residual_gate_logit).view(1, 1)
        ge_residual_scale = torch.sigmoid(self.ge_residual_gate_logit).view(1, 1)
        cp_fused = (
            cp_shared_fusion
            + cp_residual_scale * self.cp_private_residual(cp_dynamic)
            + cp_cross_scale * self.cp_cross_context_proj(cp_cross_context)
        )
        ge_fused = ge_shared_fusion + ge_residual_scale * self.ge_private_residual(ge_dynamic)
        return cp_fused, ge_fused

    # --- Pearson Loss 计算 ---
    def pearson_loss(self, x, y):
        vx = x - torch.mean(x, dim=1, keepdim=True)
        vy = y - torch.mean(y, dim=1, keepdim=True)
        cost = (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8)
        return 1 - cost.mean()

    def _compute_dose_ordinal_aux_loss(self, ge_pred, target_ge, control_ge=None, features=None, mol_ids=None):
        if (
            self.dose_ordinal_weight <= 0.0
            or not self.use_dose_feature
            or features is None
            or mol_ids is None
            or features.shape[0] < 2
        ):
            return ge_pred.new_zeros(())

        aux = compute_same_compound_dose_ordinal_loss(
            pred_delta=ge_pred - control_ge if control_ge is not None else ge_pred,
            target_delta=target_ge - control_ge if control_ge is not None else target_ge,
            mol_ids=mol_ids,
            dose_feature=features[:, -1].reshape(-1),
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
        control_cp=None,
        control_ge=None,
        features=None,
        mol_ids=None,
    ):
        """Use normalized reconstruction with weighted profile/delta PCC auxiliaries."""
        mse_cp = F.mse_loss(cp_pred, target_cp, reduction="mean")
        mse_ge = F.mse_loss(ge_pred, target_ge, reduction="mean")

        cp_profile_pcc = self.pearson_loss(cp_pred, target_cp)
        ge_profile_pcc = self.pearson_loss(ge_pred, target_ge)

        cp_delta_pcc = cp_pred.new_tensor(0.0)
        ge_delta_pcc = ge_pred.new_tensor(0.0)
        if control_cp is not None and control_ge is not None:
            cp_delta_pcc = self.pearson_loss(cp_pred - control_cp, target_cp - control_cp)
            ge_delta_pcc = self.pearson_loss(ge_pred - control_ge, target_ge - control_ge)

        total_loss = (
            mse_cp
            + mse_ge
            + self.cp_pcc_weight * cp_profile_pcc
            + self.ge_pcc_weight * ge_profile_pcc
            + self.cp_delta_pcc_weight * cp_delta_pcc
            + self.ge_delta_pcc_weight * ge_delta_pcc
        )
        total_loss = total_loss + self._compute_dose_ordinal_aux_loss(
            ge_pred=ge_pred,
            target_ge=target_ge,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        return total_loss

    # --- 辅助与训练函数 ---
    def get_feat_embdding(self, features):
        if hasattr(self, "feat_embeddings"):
            return self.feat_embeddings(features)
        return features

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ["loss"]

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
                    control_cp=control_cp,
                    control_ge=control_ge,
                    features=features,
                    mol_ids=mol_id,
                )

                loss_value += loss.item()
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                optimizer.step()

            train_dict, train_metrics_dict, _ = self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict["loss"]
            
            # --- FIX: Calculate validation loss BEFORE logging ---
            test_dict, test_metrics_dict, _ = self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_loss = test_dict["loss"]

            if epoch % 10 == 0:
                print(
                    f"[Epoch {epoch}] Loss weights -> "
                    f"cp_pcc={self.cp_pcc_weight:.3f}, ge_pcc={self.ge_pcc_weight:.3f}, "
                    f"cp_delta={self.cp_delta_pcc_weight:.3f}, ge_delta={self.ge_delta_pcc_weight:.3f}"
                )
                print(f"[Epoch {epoch}] | train loss: {train_loss:.5f}, valid_loss: {test_loss:.5f}", flush=True)


            for k, v in train_dict.items():
                epoch_hist["train_" + k].append(v)
            for k, v in train_metrics_dict.items():
                epoch_hist["train_" + k].append(v)

            for k, v in test_dict.items():
                epoch_hist["valid_" + k].append(v)
            for k, v in test_metrics_dict.items():
                epoch_hist["valid_" + k].append(v)

            if test_loss < best_value:
                best_value = test_loss
                best_epoch = epoch
                if save_model:
                    torch.save(self, self.path_model + "best_model.pt")

        return epoch_hist, best_epoch

    def test_model(self, loader, loss_item=None, metrics_func=None, perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        x1_array, x2_array, x2_cp_pred_array, x2_ge_pred_array = None, None, None, None

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                mol_features = mol_features.to(self.dev)

                batch_size = control_cp.shape[0]
                test_size += batch_size

                cp_pred, ge_pred = self.forward(control_cp, control_ge, mol_features)
                
                loss_val = self.loss(
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
                        test_dict[k] += loss_val.item() * batch_size

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
                        metrics_dict_all[k] += metrics_dict[k] * batch_size
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                    metrics_dict_all_ls["cp_id"] += list(mol_id)

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size
        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size

        return test_dict, metrics_dict_all, metrics_dict_all_ls
    
    # ... predict_profile 和 eval_x_reconstruction 保持不变 ...
    def predict_profile(self, loader):
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

    def eval_x_reconstruction(self, control_cp, control_ge, target_cp, target_ge, cp_pred, ge_pred, metrics_func=['pearson'], perturbed_centroid_CP=None, perturbed_centroid_GE=None):
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
                return (neg_hits / k, pos_hits / k)
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


class MVCModel_HyperGate_MaskCPInput(MVCModel_HyperGate):
    """Fair ablation for HyperGate: zero the CP input while keeping the dual-head task unchanged."""

    def forward(self, x_cp, x_ge, features):
        masked_cp = torch.zeros_like(x_cp)
        return super().forward(masked_cp, x_ge, features)


class MVCModel_HyperGate_MaskGEInput(MVCModel_HyperGate):
    """Fair ablation for HyperGate: zero the GE input while keeping the dual-head task unchanged."""

    def forward(self, x_cp, x_ge, features):
        masked_ge = torch.zeros_like(x_ge)
        return super().forward(x_cp, masked_ge, features)
