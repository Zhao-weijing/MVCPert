#!/usr/bin/env python3

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from MVCModel_HyperGate import MVCModel_HyperGate


class GaussianHead(nn.Module):
    """Small diagonal-Gaussian parameter head."""

    def __init__(self, in_dim, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        logvar = self.logvar(h).clamp(min=-8.0, max=6.0)
        return self.mu(h), logvar


class ResidualDeltaDecoder(nn.Module):
    """Decode a target residual from condition context plus shared/private latents."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        nn.init.constant_(last.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class MVCModel_ResidualVAE(MVCModel_HyperGate):
    """
    Conditional residual VAE on top of the HyperGate condition encoder.

    Training uses q(z | control, molecule, target_delta). Validation and inference
    use p(z | control, molecule) with latent means by default, so target profiles are
    never leaked into reported predictions.
    """

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
        super(MVCModel_ResidualVAE, self).__init__(
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
        self.vae_latent_dim = int(kwargs.get("residual_vae_latent_dim", 128))
        self.vae_hidden_dim = int(kwargs.get("residual_vae_hidden_dim", max(self.n_latent, self.embed_dim * 2)))
        self.vae_kl_weight = float(kwargs.get("residual_vae_kl_weight", 1e-4))
        self.vae_kl_warmup_epochs = int(kwargs.get("residual_vae_kl_warmup_epochs", 10))
        self.vae_disentangle_weight = float(kwargs.get("residual_vae_disentangle_weight", 1e-3))
        self.vae_disentangle_mode = str(kwargs.get("residual_vae_disentangle_mode", "corr")).strip().lower()
        self.vae_hsic_sigma = float(kwargs.get("residual_vae_hsic_sigma", 1.0))
        self.vae_prior_recon_weight = float(kwargs.get("residual_vae_prior_recon_weight", 0.5))
        self.vae_modal_mask_prob = float(kwargs.get("residual_vae_modal_mask_prob", 0.15))
        self.vae_sample_train = bool(kwargs.get("residual_vae_sample_train", True))
        self.current_kl_scale = 1.0

        cond_dim = self.fusion_input_dim * 2
        task_cond_dim = self.fusion_input_dim
        delta_embed_dim = self.embed_dim

        self.cp_delta_encoder = nn.Sequential(
            nn.Linear(self.out_images, delta_embed_dim),
            nn.LayerNorm(delta_embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.ge_delta_encoder = nn.Sequential(
            nn.Linear(self.out_genes, delta_embed_dim),
            nn.LayerNorm(delta_embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.shared_prior = GaussianHead(cond_dim, self.vae_latent_dim, self.vae_hidden_dim, self.dropout)
        self.cp_prior = GaussianHead(task_cond_dim, self.vae_latent_dim, self.vae_hidden_dim, self.dropout)
        self.ge_prior = GaussianHead(task_cond_dim, self.vae_latent_dim, self.vae_hidden_dim, self.dropout)

        self.shared_posterior = GaussianHead(
            cond_dim + 2 * delta_embed_dim,
            self.vae_latent_dim,
            self.vae_hidden_dim,
            self.dropout,
        )
        self.cp_posterior = GaussianHead(
            task_cond_dim + delta_embed_dim,
            self.vae_latent_dim,
            self.vae_hidden_dim,
            self.dropout,
        )
        self.ge_posterior = GaussianHead(
            task_cond_dim + delta_embed_dim,
            self.vae_latent_dim,
            self.vae_hidden_dim,
            self.dropout,
        )

        decoder_in_dim = task_cond_dim + 2 * self.vae_latent_dim
        self.cp_delta_decoder = ResidualDeltaDecoder(
            in_dim=decoder_in_dim,
            hidden_dim=self.vae_hidden_dim,
            out_dim=self.out_images,
            dropout=self.dropout,
        )
        self.ge_delta_decoder = ResidualDeltaDecoder(
            in_dim=decoder_in_dim,
            hidden_dim=self.vae_hidden_dim,
            out_dim=self.out_genes,
            dropout=self.dropout,
        )

        self._cached_vae_state = {}
        self._last_vae_loss_components = {}

        if self.init_w:
            self.cp_delta_encoder.apply(self._init_weights)
            self.ge_delta_encoder.apply(self._init_weights)
            self.shared_prior.apply(self._init_weights)
            self.cp_prior.apply(self._init_weights)
            self.ge_prior.apply(self._init_weights)
            self.shared_posterior.apply(self._init_weights)
            self.cp_posterior.apply(self._init_weights)
            self.ge_posterior.apply(self._init_weights)
            self.cp_delta_decoder.apply(self._init_weights)
            self.ge_delta_decoder.apply(self._init_weights)
            nn.init.normal_(self.cp_delta_decoder.net[-1].weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.ge_delta_decoder.net[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.cp_delta_decoder.net[-1].bias, 0.0)
            nn.init.constant_(self.ge_delta_decoder.net[-1].bias, 0.0)

    def _maybe_mask_modal_inputs(self, x_cp, x_ge):
        if getattr(self, "_disable_modal_random_mask", False):
            return x_cp, x_ge, None
        if not self.training or self.vae_modal_mask_prob <= 0.0:
            return x_cp, x_ge, None

        prob = min(max(self.vae_modal_mask_prob, 0.0), 1.0)
        batch_size = x_cp.shape[0]
        rnd = torch.rand(batch_size, device=x_cp.device)
        cp_mask = rnd < (prob * 0.5)
        ge_mask = (rnd >= (prob * 0.5)) & (rnd < prob)

        if cp_mask.any():
            x_cp = x_cp.clone()
            x_cp[cp_mask] = 0.0
        if ge_mask.any():
            x_ge = x_ge.clone()
            x_ge[ge_mask] = 0.0
        return x_cp, x_ge, {"cp_mask_fraction": cp_mask.float().mean(), "ge_mask_fraction": ge_mask.float().mean()}

    def _encode_condition(self, x_cp, x_ge, features):
        z_cp = self.encoder_cp(x_cp)
        z_ge = self.encoder_ge(x_ge)

        if hasattr(self, "feat_embeddings"):
            feat_embed_raw = self.feat_embeddings(features)
        else:
            feat_embed_raw = features
        z_mol = self.mol_to_embed(feat_embed_raw)

        token_sequence = self._build_modal_token_sequence(z_cp, z_ge, z_mol)
        hyper_context = self._build_hypergraph_context(z_cp, z_ge, z_mol)

        fused_tokens = token_sequence
        for li, layer in enumerate(self.fusion_layers, start=1):
            fused_tokens = layer(fused_tokens)
            if self.hyper_refinement is not None and li == self.hyper_insert_after:
                fused_tokens = self.hyper_refinement(fused_tokens, hyper_context)

        cp_tokens, ge_tokens, mol_tokens = self._split_modal_tokens(fused_tokens)
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
        condition = torch.cat([cp_fused, ge_fused], dim=1)
        return condition, cp_fused, ge_fused

    def _reparameterize(self, mu, logvar, sample):
        if not sample:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def _gaussian_kl(self, post_mu, post_logvar, prior_mu, prior_logvar):
        post_var = torch.exp(post_logvar)
        prior_var = torch.exp(prior_logvar)
        kl = 0.5 * (
            prior_logvar
            - post_logvar
            + (post_var + (post_mu - prior_mu).pow(2)) / (prior_var + 1e-8)
            - 1.0
        )
        return kl.sum(dim=1).mean()

    def _cross_correlation_penalty(self, left, right):
        if left.shape[0] < 2:
            return left.new_zeros(())
        left = left - left.mean(dim=0, keepdim=True)
        right = right - right.mean(dim=0, keepdim=True)
        left = left / (left.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
        right = right / (right.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
        corr = left.transpose(0, 1).matmul(right) / max(left.shape[0] - 1, 1)
        return corr.pow(2).mean()

    def _gaussian_kernel(self, x, sigma):
        sq_dist = torch.cdist(x, x, p=2).pow(2)
        denom = 2.0 * max(float(sigma), 1e-6) ** 2
        return torch.exp(-sq_dist / denom)

    def _hsic_penalty(self, left, right):
        batch_size = left.shape[0]
        if batch_size < 4:
            return left.new_zeros(())
        sigma = self.vae_hsic_sigma
        k = self._gaussian_kernel(left, sigma=sigma)
        l = self._gaussian_kernel(right, sigma=sigma)
        h = torch.eye(batch_size, device=left.device, dtype=left.dtype)
        h = h - h.new_full((batch_size, batch_size), 1.0 / float(batch_size))
        kh = h.matmul(k).matmul(h)
        lh = h.matmul(l).matmul(h)
        norm = float((batch_size - 1) ** 2)
        return (kh * lh).sum() / max(norm, 1.0)

    def _compute_disentangle_penalty(self, shared_mu, cp_mu, ge_mu):
        mode = self.vae_disentangle_mode
        if mode == "none":
            return shared_mu.new_zeros(())
        if mode == "corr":
            return (
                self._cross_correlation_penalty(shared_mu, cp_mu)
                + self._cross_correlation_penalty(shared_mu, ge_mu)
                + 0.5 * self._cross_correlation_penalty(cp_mu, ge_mu)
            )
        if mode == "hsic":
            return (
                self._hsic_penalty(shared_mu, cp_mu)
                + self._hsic_penalty(shared_mu, ge_mu)
                + 0.5 * self._hsic_penalty(cp_mu, ge_mu)
            )
        raise ValueError(f"Unsupported residual_vae_disentangle_mode: {mode}")

    def _build_posterior_inputs(self, condition, cp_fused, ge_fused, control_cp, control_ge, target_cp, target_ge):
        cp_delta_embed = self.cp_delta_encoder(target_cp - control_cp)
        ge_delta_embed = self.ge_delta_encoder(target_ge - control_ge)
        shared_input = torch.cat([condition, cp_delta_embed, ge_delta_embed], dim=1)
        cp_input = torch.cat([cp_fused, cp_delta_embed], dim=1)
        ge_input = torch.cat([ge_fused, ge_delta_embed], dim=1)
        return shared_input, cp_input, ge_input

    def _build_latents(
        self,
        condition,
        cp_fused,
        ge_fused,
        control_cp,
        control_ge,
        target_cp=None,
        target_ge=None,
        sample=None,
    ):
        prior_shared_mu, prior_shared_logvar = self.shared_prior(condition)
        prior_cp_mu, prior_cp_logvar = self.cp_prior(cp_fused)
        prior_ge_mu, prior_ge_logvar = self.ge_prior(ge_fused)

        use_posterior = target_cp is not None and target_ge is not None
        if sample is None:
            sample = self.training and self.vae_sample_train

        if use_posterior:
            shared_input, cp_input, ge_input = self._build_posterior_inputs(
                condition=condition,
                cp_fused=cp_fused,
                ge_fused=ge_fused,
                control_cp=control_cp,
                control_ge=control_ge,
                target_cp=target_cp,
                target_ge=target_ge,
            )
            shared_mu, shared_logvar = self.shared_posterior(shared_input)
            cp_mu, cp_logvar = self.cp_posterior(cp_input)
            ge_mu, ge_logvar = self.ge_posterior(ge_input)
        else:
            shared_mu, shared_logvar = prior_shared_mu, prior_shared_logvar
            cp_mu, cp_logvar = prior_cp_mu, prior_cp_logvar
            ge_mu, ge_logvar = prior_ge_mu, prior_ge_logvar

        z_shared = self._reparameterize(shared_mu, shared_logvar, sample=sample)
        z_cp = self._reparameterize(cp_mu, cp_logvar, sample=sample)
        z_ge = self._reparameterize(ge_mu, ge_logvar, sample=sample)

        state = {
            "use_posterior": use_posterior,
            "shared_mu": shared_mu,
            "shared_logvar": shared_logvar,
            "cp_mu": cp_mu,
            "cp_logvar": cp_logvar,
            "ge_mu": ge_mu,
            "ge_logvar": ge_logvar,
            "prior_shared_mu": prior_shared_mu,
            "prior_shared_logvar": prior_shared_logvar,
            "prior_cp_mu": prior_cp_mu,
            "prior_cp_logvar": prior_cp_logvar,
            "prior_ge_mu": prior_ge_mu,
            "prior_ge_logvar": prior_ge_logvar,
            "z_shared": z_shared,
            "z_cp": z_cp,
            "z_ge": z_ge,
        }
        return z_shared, z_cp, z_ge, state

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        masked_cp, masked_ge, mask_state = self._maybe_mask_modal_inputs(x_cp, x_ge)
        condition, cp_fused, ge_fused = self._encode_condition(masked_cp, masked_ge, features)
        z_shared, z_cp, z_ge, state = self._build_latents(
            condition=condition,
            cp_fused=cp_fused,
            ge_fused=ge_fused,
            control_cp=x_cp,
            control_ge=x_ge,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
        )

        cp_delta = self.cp_delta_decoder(torch.cat([cp_fused, z_shared, z_cp], dim=1))
        ge_delta = self.ge_delta_decoder(torch.cat([ge_fused, z_shared, z_ge], dim=1))
        cp_pred = x_cp + cp_delta
        ge_pred = x_ge + ge_delta

        prior_cp_pred = None
        prior_ge_pred = None
        if target_cp is not None and target_ge is not None:
            prior_cp_delta = self.cp_delta_decoder(
                torch.cat([cp_fused, state["prior_shared_mu"], state["prior_cp_mu"]], dim=1)
            )
            prior_ge_delta = self.ge_delta_decoder(
                torch.cat([ge_fused, state["prior_shared_mu"], state["prior_ge_mu"]], dim=1)
            )
            prior_cp_pred = x_cp + prior_cp_delta
            prior_ge_pred = x_ge + prior_ge_delta

        state["mask_state"] = mask_state
        state["cp_delta_pred"] = cp_delta
        state["ge_delta_pred"] = ge_delta
        state["prior_cp_pred"] = prior_cp_pred
        state["prior_ge_pred"] = prior_ge_pred
        self._cached_vae_state = state
        return cp_pred, ge_pred

    def _compute_vae_regularization(self, reference_tensor):
        state = getattr(self, "_cached_vae_state", {})
        if not state or not state.get("use_posterior", False):
            zero = reference_tensor.new_zeros(())
            return zero, zero, {
                "kl_shared": zero,
                "kl_cp": zero,
                "kl_ge": zero,
                "kl_loss": zero,
                "disentangle_loss": zero,
            }

        kl_shared = self._gaussian_kl(
            state["shared_mu"],
            state["shared_logvar"],
            state["prior_shared_mu"],
            state["prior_shared_logvar"],
        )
        kl_cp = self._gaussian_kl(state["cp_mu"], state["cp_logvar"], state["prior_cp_mu"], state["prior_cp_logvar"])
        kl_ge = self._gaussian_kl(state["ge_mu"], state["ge_logvar"], state["prior_ge_mu"], state["prior_ge_logvar"])
        kl_loss = kl_shared + kl_cp + kl_ge

        disentangle_loss = self._compute_disentangle_penalty(
            state["shared_mu"],
            state["cp_mu"],
            state["ge_mu"],
        )

        return kl_loss, disentangle_loss, {
            "kl_shared": kl_shared.detach(),
            "kl_cp": kl_cp.detach(),
            "kl_ge": kl_ge.detach(),
            "kl_loss": kl_loss.detach(),
            "disentangle_loss": disentangle_loss.detach(),
        }

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
        base_loss = super().loss(
            target_cp=target_cp,
            target_ge=target_ge,
            cp_pred=cp_pred,
            ge_pred=ge_pred,
            control_cp=control_cp,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        state = getattr(self, "_cached_vae_state", {})
        prior_recon_loss = cp_pred.new_zeros(())
        if (
            self.vae_prior_recon_weight > 0.0
            and state
            and state.get("prior_cp_pred") is not None
            and state.get("prior_ge_pred") is not None
        ):
            prior_recon_loss = super().loss(
                target_cp=target_cp,
                target_ge=target_ge,
                cp_pred=state["prior_cp_pred"],
                ge_pred=state["prior_ge_pred"],
                control_cp=control_cp,
                control_ge=control_ge,
                features=features,
                mol_ids=mol_ids,
            )
        kl_loss, disentangle_loss, components = self._compute_vae_regularization(cp_pred)
        weighted_prior_recon = self.vae_prior_recon_weight * prior_recon_loss
        weighted_kl = self.vae_kl_weight * self.current_kl_scale * kl_loss
        weighted_disentangle = self.vae_disentangle_weight * disentangle_loss
        total = base_loss + weighted_prior_recon + weighted_kl + weighted_disentangle
        self._last_vae_loss_components = {
            "base_loss": float(base_loss.detach().cpu()),
            "prior_recon_loss": float(prior_recon_loss.detach().cpu()),
            "weighted_prior_recon": float(weighted_prior_recon.detach().cpu()),
            "weighted_kl": float(weighted_kl.detach().cpu()),
            "weighted_disentangle": float(weighted_disentangle.detach().cpu()),
            "disentangle_mode": self.vae_disentangle_mode,
            **{key: float(value.detach().cpu()) for key, value in components.items()},
        }
        return total

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ["loss"]

        best_value = np.inf
        best_epoch = 0

        for epoch in range(n_epochs):
            if self.vae_kl_warmup_epochs > 0:
                self.current_kl_scale = min(1.0, float(epoch + 1) / float(self.vae_kl_warmup_epochs))
            else:
                self.current_kl_scale = 1.0

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
                cp_pred, ge_pred = self.forward(
                    control_cp,
                    control_ge,
                    features,
                    target_cp=target_cp,
                    target_ge=target_ge,
                    sample=self.vae_sample_train,
                )
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
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                optimizer.step()

            train_dict, train_metrics_dict, _ = self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict["loss"]
            test_dict, test_metrics_dict, _ = self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_loss = test_dict["loss"]

            if epoch % 10 == 0:
                shared_dropout_desc = ""
                if hasattr(self, "vae_shared_dropout_prob_cp") and hasattr(self, "vae_shared_dropout_prob_ge"):
                    shared_dropout_desc = (
                        f", shared_drop_cp={self.vae_shared_dropout_prob_cp:.3f}, "
                        f"shared_drop_ge={self.vae_shared_dropout_prob_ge:.3f}"
                    )
                elif hasattr(self, "vae_shared_dropout_prob"):
                    shared_dropout_desc = f", shared_drop={self.vae_shared_dropout_prob:.3f}"
                print(
                    f"[Epoch {epoch}] ResidualVAE weights -> "
                    f"prior={self.vae_prior_recon_weight:.6g}, "
                    f"kl={self.vae_kl_weight:.6g}*{self.current_kl_scale:.3f}, "
                    f"disentangle={self.vae_disentangle_weight:.6g}"
                    f"({self.vae_disentangle_mode}), "
                    f"modal_mask={self.vae_modal_mask_prob:.3f}"
                    f"{shared_dropout_desc}"
                )
                if hasattr(self, "vae_factorized_correction"):
                    role_desc = [
                        f"factorized={self.vae_factorized_correction}",
                        f"private_orth={getattr(self, 'vae_private_orth_weight', 0.0):.6g}",
                        f"private_remainder={getattr(self, 'vae_private_remainder_weight', 0.0):.6g}",
                    ]
                    if hasattr(self, "vae_shared_infonce_weight"):
                        role_desc.append(
                            f"shared_infonce={getattr(self, 'vae_shared_infonce_weight', 0.0):.6g}"
                        )
                    if hasattr(self, "vae_shared_infonce_temperature"):
                        role_desc.append(
                            f"shared_infonce_temp={getattr(self, 'vae_shared_infonce_temperature', 0.1):.6g}"
                        )
                    print(
                        f"[Epoch {epoch}] ResidualVAE roles -> "
                        + ", ".join(role_desc)
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
                cp_pred, ge_pred = self.forward(control_cp, control_ge, mol_features, sample=False)
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
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]
                    metrics_dict_all_ls["cp_id"] += list(mol_id)

        if test_size == 0:
            return test_dict, metrics_dict_all, metrics_dict_all_ls

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size
        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size
        return test_dict, metrics_dict_all, metrics_dict_all_ls

    def generate_residual_samples(self, x_cp, x_ge, features, n_samples=4, use_mean=False):
        """Generate pseudo-replicate profiles from the conditional prior."""
        cp_samples = []
        ge_samples = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(int(n_samples)):
                cp_pred, ge_pred = self.forward(x_cp, x_ge, features, sample=not use_mean)
                cp_samples.append(cp_pred)
                ge_samples.append(ge_pred)
        if was_training:
            self.train()
        return torch.stack(cp_samples, dim=0), torch.stack(ge_samples, dim=0)


class MVCModel_HyperGateResidualVAE(MVCModel_ResidualVAE):
    """HyperGate deterministic prediction plus a small conditional VAE correction."""

    VIEW_FULL = "full"
    VIEW_GE = "ge_view"
    VIEW_CP = "cp_view"

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
        super(MVCModel_HyperGateResidualVAE, self).__init__(
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
        self.vae_base_recon_weight = float(kwargs.get("residual_vae_base_recon_weight", 1.0))
        self.vae_correction_scale = float(kwargs.get("residual_vae_correction_scale", 0.25))
        self.vae_correction_l2_weight = float(kwargs.get("residual_vae_correction_l2_weight", 0.05))
        # The current anchored ResidualVAE always uses centered factorized corrections.
        # Older toggles are intentionally ignored to keep the active model path simple.
        self.vae_centered_correction = True
        self.vae_factorized_correction = True
        self.vae_private_orth_weight = float(kwargs.get("residual_vae_private_orth_weight", 0.0))
        self.vae_private_remainder_weight = float(kwargs.get("residual_vae_private_remainder_weight", 0.0))
        self.vae_shared_infonce_weight = float(kwargs.get("residual_vae_shared_infonce_weight", 0.0))
        self.vae_shared_infonce_temperature = float(kwargs.get("residual_vae_shared_infonce_temperature", 0.1))
        self.residual_vae_training_stage = str(kwargs.get("residual_vae_training_stage", "full_only")).strip().lower()
        if self.residual_vae_training_stage not in {"full_only", "three_view"}:
            raise ValueError(
                f"Unsupported residual_vae_training_stage: {self.residual_vae_training_stage}"
            )
        self.residual_vae_geview_weight = float(kwargs.get("residual_vae_geview_weight", 0.2))
        self.residual_vae_cpview_weight = float(kwargs.get("residual_vae_cpview_weight", 0.2))
        self.residual_vae_aux_view_mode = str(kwargs.get("residual_vae_aux_view_mode", "random")).strip().lower()
        if self.residual_vae_aux_view_mode not in {"random", "ge", "cp", "alternate"}:
            raise ValueError(
                f"Unsupported residual_vae_aux_view_mode: {self.residual_vae_aux_view_mode}"
            )
        branch_in_dim = self.fusion_input_dim + self.vae_latent_dim
        self.cp_shared_delta_decoder_factorized = ResidualDeltaDecoder(
            in_dim=branch_in_dim,
            hidden_dim=self.vae_hidden_dim,
            out_dim=self.out_images,
            dropout=self.dropout,
        )
        self.cp_private_delta_decoder_factorized = ResidualDeltaDecoder(
            in_dim=branch_in_dim,
            hidden_dim=self.vae_hidden_dim,
            out_dim=self.out_images,
            dropout=self.dropout,
        )
        self.ge_shared_delta_decoder_factorized = ResidualDeltaDecoder(
            in_dim=branch_in_dim,
            hidden_dim=self.vae_hidden_dim,
            out_dim=self.out_genes,
            dropout=self.dropout,
        )
        self.ge_private_delta_decoder_factorized = ResidualDeltaDecoder(
            in_dim=branch_in_dim,
            hidden_dim=self.vae_hidden_dim,
            out_dim=self.out_genes,
            dropout=self.dropout,
        )
        role_embed_dim = self.embed_dim
        self.cp_role_projector = nn.Sequential(
            nn.Linear(self.out_images, role_embed_dim),
            nn.LayerNorm(role_embed_dim),
            nn.GELU(),
        )
        self.ge_role_projector = nn.Sequential(
            nn.Linear(self.out_genes, role_embed_dim),
            nn.LayerNorm(role_embed_dim),
            nn.GELU(),
        )

        last_cp = self.cp_delta_decoder.net[-1]
        last_ge = self.ge_delta_decoder.net[-1]
        nn.init.normal_(last_cp.weight, mean=0.0, std=1e-4)
        nn.init.normal_(last_ge.weight, mean=0.0, std=1e-4)
        nn.init.constant_(last_cp.bias, 0.0)
        nn.init.constant_(last_ge.bias, 0.0)
        factorized_decoders = [
            self.cp_shared_delta_decoder_factorized,
            self.cp_private_delta_decoder_factorized,
            self.ge_shared_delta_decoder_factorized,
            self.ge_private_delta_decoder_factorized,
        ]
        if self.init_w:
            for decoder in factorized_decoders:
                decoder.apply(self._init_weights)
            self.cp_role_projector.apply(self._init_weights)
            self.ge_role_projector.apply(self._init_weights)
        for decoder in factorized_decoders:
            last = decoder.net[-1]
            nn.init.normal_(last.weight, mean=0.0, std=1e-4)
            nn.init.constant_(last.bias, 0.0)
        self._residual_vae_aux_counter = 0
        self._cached_residual_vae_view_state = {}

    @classmethod
    def _canonical_view(cls, view):
        value = str(view).strip().lower()
        aliases = {
            "full": cls.VIEW_FULL,
            "full_view": cls.VIEW_FULL,
            "ge": cls.VIEW_GE,
            "geview": cls.VIEW_GE,
            "ge_view": cls.VIEW_GE,
            "cp": cls.VIEW_CP,
            "cpview": cls.VIEW_CP,
            "cp_view": cls.VIEW_CP,
        }
        if value not in aliases:
            raise ValueError(f"Unsupported view: {view}")
        return aliases[value]

    def _mask_inputs_for_view(self, x_cp, x_ge, view):
        view = self._canonical_view(view)
        if view == self.VIEW_FULL:
            return x_cp, x_ge
        if view == self.VIEW_GE:
            return torch.zeros_like(x_cp), x_ge
        if view == self.VIEW_CP:
            return x_cp, torch.zeros_like(x_ge)
        raise ValueError(f"Unsupported view: {view}")

    def _choose_aux_view(self, force_aux_view=None):
        if force_aux_view is not None:
            view = self._canonical_view(force_aux_view)
            if view == self.VIEW_FULL:
                raise ValueError("Auxiliary view cannot be full")
            return view
        if self.residual_vae_aux_view_mode == "ge":
            return self.VIEW_GE
        if self.residual_vae_aux_view_mode == "cp":
            return self.VIEW_CP
        if self.residual_vae_aux_view_mode == "alternate":
            view = self.VIEW_GE if self._residual_vae_aux_counter % 2 == 0 else self.VIEW_CP
            self._residual_vae_aux_counter += 1
            return view
        return self.VIEW_GE if torch.rand((), device=self.log_vars.device).item() < 0.5 else self.VIEW_CP

    def _private_orth_weight_value(self):
        return float(getattr(self, "vae_private_orth_weight", 0.0))

    def _private_remainder_weight_value(self):
        return float(getattr(self, "vae_private_remainder_weight", 0.0))

    def _shared_infonce_weight_value(self):
        return float(getattr(self, "vae_shared_infonce_weight", 0.0))

    def _shared_infonce_temperature_value(self):
        return float(getattr(self, "vae_shared_infonce_temperature", 0.1))

    def _decode_raw_correction_components(self, cp_fused, ge_fused, z_shared, z_cp, z_ge, z_shared_ge=None):
        if z_shared_ge is None:
            z_shared_ge = z_shared
        cp_shared_raw = self.cp_shared_delta_decoder_factorized(torch.cat([cp_fused, z_shared], dim=1))
        cp_private_raw = self.cp_private_delta_decoder_factorized(torch.cat([cp_fused, z_cp], dim=1))
        ge_shared_raw = self.ge_shared_delta_decoder_factorized(torch.cat([ge_fused, z_shared_ge], dim=1))
        ge_private_raw = self.ge_private_delta_decoder_factorized(torch.cat([ge_fused, z_ge], dim=1))
        cp_corr_raw = cp_shared_raw + cp_private_raw
        ge_corr_raw = ge_shared_raw + ge_private_raw
        return {
            "cp_corr_raw": cp_corr_raw,
            "ge_corr_raw": ge_corr_raw,
            "cp_shared_raw": cp_shared_raw,
            "cp_private_raw": cp_private_raw,
            "ge_shared_raw": ge_shared_raw,
            "ge_private_raw": ge_private_raw,
        }

    def _apply_correction_centering(self, cp_corr, ge_corr, zero_cp_corr, zero_ge_corr):
        return cp_corr - zero_cp_corr, ge_corr - zero_ge_corr

    def _center_correction_components(self, raw_components, zero_components):
        cp_shared_corr = raw_components["cp_shared_raw"] - zero_components["cp_shared_raw"]
        cp_private_corr = raw_components["cp_private_raw"] - zero_components["cp_private_raw"]
        ge_shared_corr = raw_components["ge_shared_raw"] - zero_components["ge_shared_raw"]
        ge_private_corr = raw_components["ge_private_raw"] - zero_components["ge_private_raw"]
        return {
            "cp_corr": cp_shared_corr + cp_private_corr,
            "ge_corr": ge_shared_corr + ge_private_corr,
            "cp_shared_corr": cp_shared_corr,
            "cp_private_corr": cp_private_corr,
            "ge_shared_corr": ge_shared_corr,
            "ge_private_corr": ge_private_corr,
        }

    def _project_role_corrections(self, cp_corr, ge_corr):
        cp_embed = self.cp_role_projector(cp_corr)
        ge_embed = self.ge_role_projector(ge_corr)
        return cp_embed, ge_embed

    @staticmethod
    def _cosine_similarity_squared(left, right):
        left = F.normalize(left, dim=1, eps=1e-6)
        right = F.normalize(right, dim=1, eps=1e-6)
        return (left * right).sum(dim=1).pow(2).mean()

    def _compute_private_orth_loss(self, reference_tensor):
        state = getattr(self, "_cached_vae_state", {})
        if (
            self._private_orth_weight_value() <= 0.0
            or not state
            or state.get("cp_private_correction") is None
            or state.get("ge_private_correction") is None
        ):
            return reference_tensor.new_zeros(())

        cp_private_embed, ge_private_embed = self._project_role_corrections(
            state["cp_private_correction"],
            state["ge_private_correction"],
        )
        return self._cosine_similarity_squared(cp_private_embed, ge_private_embed)

    def _compute_private_remainder_loss(self, target_cp, target_ge, reference_tensor):
        state = getattr(self, "_cached_vae_state", {})
        if (
            self._private_remainder_weight_value() <= 0.0
            or not state
            or state.get("base_cp_pred") is None
            or state.get("base_ge_pred") is None
            or state.get("cp_shared_correction") is None
            or state.get("cp_private_correction") is None
            or state.get("ge_shared_correction") is None
            or state.get("ge_private_correction") is None
        ):
            return reference_tensor.new_zeros(())

        scale = max(abs(float(self.vae_correction_scale)), 1e-6)
        target_cp_correction = (target_cp - state["base_cp_pred"].detach()) / scale
        target_ge_correction = (target_ge - state["base_ge_pred"].detach()) / scale
        cp_private_target = target_cp_correction - state["cp_shared_correction"].detach()
        ge_private_target = target_ge_correction - state["ge_shared_correction"].detach()
        cp_private_loss = F.mse_loss(state["cp_private_correction"], cp_private_target, reduction="mean")
        ge_private_loss = F.mse_loss(state["ge_private_correction"], ge_private_target, reduction="mean")
        return cp_private_loss + ge_private_loss

    def _contrastive_with_negative_pools(self, anchor, positive, negative_pools, temperature):
        if anchor.shape[0] < 2:
            return anchor.new_zeros(())
        anchor = F.normalize(anchor, dim=1, eps=1e-6)
        positive = F.normalize(positive, dim=1, eps=1e-6)
        logits = [anchor.matmul(positive.transpose(0, 1)) / temperature]
        for negative in negative_pools:
            negative = F.normalize(negative, dim=1, eps=1e-6)
            logits.append(anchor.matmul(negative.transpose(0, 1)) / temperature)
        logits = torch.cat(logits, dim=1)
        targets = torch.arange(anchor.shape[0], device=anchor.device)
        return F.cross_entropy(logits, targets)

    def _compute_shared_infonce_loss(self, target_cp, target_ge, reference_tensor):
        state = getattr(self, "_cached_vae_state", {})
        if (
            self._shared_infonce_weight_value() <= 0.0
            or not state
            or state.get("base_cp_pred") is None
            or state.get("base_ge_pred") is None
            or state.get("cp_shared_correction") is None
            or state.get("ge_shared_correction") is None
        ):
            return reference_tensor.new_zeros(())

        scale = max(abs(float(self.vae_correction_scale)), 1e-6)
        target_cp_correction = (target_cp - state["base_cp_pred"].detach()) / scale
        target_ge_correction = (target_ge - state["base_ge_pred"].detach()) / scale
        cp_private_target = target_cp_correction - state["cp_shared_correction"].detach()
        ge_private_target = target_ge_correction - state["ge_shared_correction"].detach()

        cp_shared_embed, ge_shared_embed = self._project_role_corrections(
            state["cp_shared_correction"],
            state["ge_shared_correction"],
        )
        cp_private_target_embed, ge_private_target_embed = self._project_role_corrections(
            cp_private_target.detach(),
            ge_private_target.detach(),
        )
        temperature = max(self._shared_infonce_temperature_value(), 1e-4)
        cp_to_ge = self._contrastive_with_negative_pools(
            anchor=cp_shared_embed,
            positive=ge_shared_embed,
            negative_pools=[cp_private_target_embed, ge_private_target_embed],
            temperature=temperature,
        )
        ge_to_cp = self._contrastive_with_negative_pools(
            anchor=ge_shared_embed,
            positive=cp_shared_embed,
            negative_pools=[cp_private_target_embed, ge_private_target_embed],
            temperature=temperature,
        )
        return 0.5 * (cp_to_ge + ge_to_cp)

    def _predict_base_from_fused(self, cp_fused, ge_fused):
        base_cp_latent = self.cp_output_projection(cp_fused)
        base_ge_latent = self.ge_output_projection(ge_fused)
        base_cp_pred = self.decoder_cp(base_cp_latent)
        base_ge_pred = self.decoder_ge(base_ge_latent)
        return base_cp_pred, base_ge_pred

    @staticmethod
    def _zero_latents_like(z_shared, z_cp, z_ge):
        return torch.zeros_like(z_shared), torch.zeros_like(z_cp), torch.zeros_like(z_ge)

    def forward_with_latent_mode(
        self,
        x_cp,
        x_ge,
        features,
        mode="all",
        target_cp=None,
        target_ge=None,
        sample=None,
        return_components=False,
    ):
        """
        Diagnostic-only forward path for latent ablations.

        Modes:
          - base: deterministic HyperGate head only
          - zero_latent: correction decoder with all latents zeroed
          - shared_only: only z_shared active
          - cp_private_only: only z_cp active on the CP branch
          - ge_private_only: only z_ge active on the GE branch
          - private_only: z_cp/z_ge active, z_shared zeroed
          - all: same path as the standard anchored ResidualVAE forward
        """
        valid_modes = {
            "base",
            "zero_latent",
            "shared_only",
            "cp_private_only",
            "ge_private_only",
            "private_only",
            "all",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unsupported latent ablation mode: {mode}. Expected one of {sorted(valid_modes)}")

        masked_cp, masked_ge, mask_state = self._maybe_mask_modal_inputs(x_cp, x_ge)
        condition, cp_fused, ge_fused = self._encode_condition(masked_cp, masked_ge, features)
        base_cp_pred, base_ge_pred = self._predict_base_from_fused(cp_fused, ge_fused)

        z_shared, z_cp, z_ge, state = self._build_anchor_latents(
            condition=condition,
            cp_fused=cp_fused,
            ge_fused=ge_fused,
            base_cp_pred=base_cp_pred,
            base_ge_pred=base_ge_pred,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
        )

        zero_shared, zero_cp, zero_ge = self._zero_latents_like(z_shared, z_cp, z_ge)
        zero_components = self._decode_raw_correction_components(
            cp_fused,
            ge_fused,
            zero_shared,
            zero_cp,
            zero_ge,
        )
        zero_cp_corr_raw = zero_components["cp_corr_raw"]
        zero_ge_corr_raw = zero_components["ge_corr_raw"]

        if mode == "base":
            raw_components = {
                "cp_corr_raw": torch.zeros_like(base_cp_pred),
                "ge_corr_raw": torch.zeros_like(base_ge_pred),
                "cp_shared_raw": torch.zeros_like(base_cp_pred),
                "cp_private_raw": torch.zeros_like(base_cp_pred),
                "ge_shared_raw": torch.zeros_like(base_ge_pred),
                "ge_private_raw": torch.zeros_like(base_ge_pred),
            }
        elif mode == "zero_latent":
            raw_components = zero_components
        elif mode == "shared_only":
            raw_components = self._decode_raw_correction_components(
                cp_fused,
                ge_fused,
                z_shared,
                zero_cp,
                zero_ge,
                z_shared_ge=z_shared,
            )
        elif mode == "cp_private_only":
            raw_components = self._decode_raw_correction_components(cp_fused, ge_fused, zero_shared, z_cp, zero_ge)
            raw_components["ge_corr_raw"] = zero_ge_corr_raw
            raw_components["ge_shared_raw"] = zero_components["ge_shared_raw"]
            raw_components["ge_private_raw"] = zero_components["ge_private_raw"]
        elif mode == "ge_private_only":
            raw_components = self._decode_raw_correction_components(cp_fused, ge_fused, zero_shared, zero_cp, z_ge)
            raw_components["cp_corr_raw"] = zero_cp_corr_raw
            raw_components["cp_shared_raw"] = zero_components["cp_shared_raw"]
            raw_components["cp_private_raw"] = zero_components["cp_private_raw"]
        elif mode == "private_only":
            raw_components = self._decode_raw_correction_components(cp_fused, ge_fused, zero_shared, z_cp, z_ge)
        else:
            raw_components = self._decode_raw_correction_components(
                cp_fused,
                ge_fused,
                z_shared,
                z_cp,
                z_ge,
                z_shared_ge=z_shared,
            )

        if mode == "base":
            centered_components = {
                "cp_corr": raw_components["cp_corr_raw"],
                "ge_corr": raw_components["ge_corr_raw"],
                "cp_shared_corr": raw_components["cp_shared_raw"],
                "cp_private_corr": raw_components["cp_private_raw"],
                "ge_shared_corr": raw_components["ge_shared_raw"],
                "ge_private_corr": raw_components["ge_private_raw"],
            }
        else:
            centered_components = self._center_correction_components(raw_components, zero_components)
        cp_corr = centered_components["cp_corr"]
        ge_corr = centered_components["ge_corr"]

        cp_pred = base_cp_pred + self.vae_correction_scale * cp_corr
        ge_pred = base_ge_pred + self.vae_correction_scale * ge_corr

        if not return_components:
            return cp_pred, ge_pred

        components = {
            "mode": mode,
            "mask_state": mask_state,
            "cp_fused": cp_fused,
            "ge_fused": ge_fused,
            "base_cp_pred": base_cp_pred,
            "base_ge_pred": base_ge_pred,
            "cp_correction": cp_corr,
            "ge_correction": ge_corr,
            "cp_shared_correction": centered_components["cp_shared_corr"],
            "cp_private_correction": centered_components["cp_private_corr"],
            "ge_shared_correction": centered_components["ge_shared_corr"],
            "ge_private_correction": centered_components["ge_private_corr"],
            "cp_correction_raw": raw_components["cp_corr_raw"],
            "ge_correction_raw": raw_components["ge_corr_raw"],
            "cp_shared_correction_raw": raw_components["cp_shared_raw"],
            "cp_private_correction_raw": raw_components["cp_private_raw"],
            "ge_shared_correction_raw": raw_components["ge_shared_raw"],
            "ge_private_correction_raw": raw_components["ge_private_raw"],
            "zero_cp_correction_raw": zero_cp_corr_raw,
            "zero_ge_correction_raw": zero_ge_corr_raw,
            "zero_cp_shared_correction_raw": zero_components["cp_shared_raw"],
            "zero_cp_private_correction_raw": zero_components["cp_private_raw"],
            "zero_ge_shared_correction_raw": zero_components["ge_shared_raw"],
            "zero_ge_private_correction_raw": zero_components["ge_private_raw"],
            "z_shared": z_shared,
            "z_cp": z_cp,
            "z_ge": z_ge,
            "state": state,
        }
        return cp_pred, ge_pred, components

    def _build_anchor_latents(self, condition, cp_fused, ge_fused, base_cp_pred, base_ge_pred, target_cp=None, target_ge=None, sample=None):
        prior_shared_mu, prior_shared_logvar = self.shared_prior(condition)
        prior_cp_mu, prior_cp_logvar = self.cp_prior(cp_fused)
        prior_ge_mu, prior_ge_logvar = self.ge_prior(ge_fused)

        use_posterior = target_cp is not None and target_ge is not None
        if sample is None:
            sample = self.training and self.vae_sample_train

        if use_posterior:
            cp_corr_target = target_cp - base_cp_pred
            ge_corr_target = target_ge - base_ge_pred
            cp_corr_embed = self.cp_delta_encoder(cp_corr_target)
            ge_corr_embed = self.ge_delta_encoder(ge_corr_target)
            shared_input = torch.cat([condition, cp_corr_embed, ge_corr_embed], dim=1)
            cp_input = torch.cat([cp_fused, cp_corr_embed], dim=1)
            ge_input = torch.cat([ge_fused, ge_corr_embed], dim=1)
            shared_mu, shared_logvar = self.shared_posterior(shared_input)
            cp_mu, cp_logvar = self.cp_posterior(cp_input)
            ge_mu, ge_logvar = self.ge_posterior(ge_input)
        else:
            shared_mu, shared_logvar = prior_shared_mu, prior_shared_logvar
            cp_mu, cp_logvar = prior_cp_mu, prior_cp_logvar
            ge_mu, ge_logvar = prior_ge_mu, prior_ge_logvar

        z_shared = self._reparameterize(shared_mu, shared_logvar, sample=sample)
        z_cp = self._reparameterize(cp_mu, cp_logvar, sample=sample)
        z_ge = self._reparameterize(ge_mu, ge_logvar, sample=sample)
        state = {
            "use_posterior": use_posterior,
            "shared_mu": shared_mu,
            "shared_logvar": shared_logvar,
            "cp_mu": cp_mu,
            "cp_logvar": cp_logvar,
            "ge_mu": ge_mu,
            "ge_logvar": ge_logvar,
            "prior_shared_mu": prior_shared_mu,
            "prior_shared_logvar": prior_shared_logvar,
            "prior_cp_mu": prior_cp_mu,
            "prior_cp_logvar": prior_cp_logvar,
            "prior_ge_mu": prior_ge_mu,
            "prior_ge_logvar": prior_ge_logvar,
            "z_shared": z_shared,
            "z_cp": z_cp,
            "z_ge": z_ge,
        }
        return z_shared, z_cp, z_ge, state

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        cp_pred, ge_pred, components = self.forward_with_latent_mode(
            x_cp=x_cp,
            x_ge=x_ge,
            features=features,
            mode="all",
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
            return_components=True,
        )
        state = components["state"]
        mask_state = components["mask_state"]
        cp_fused = components["cp_fused"]
        ge_fused = components["ge_fused"]
        base_cp_pred = components["base_cp_pred"]
        base_ge_pred = components["base_ge_pred"]
        cp_corr = components["cp_correction"]
        ge_corr = components["ge_correction"]

        prior_cp_pred = None
        prior_ge_pred = None
        if target_cp is not None and target_ge is not None:
            prior_raw_components = self._decode_raw_correction_components(
                cp_fused,
                ge_fused,
                state["prior_shared_mu"],
                state["prior_cp_mu"],
                state["prior_ge_mu"],
                z_shared_ge=state["prior_shared_mu"],
            )
            zero_components = {
                "cp_corr_raw": components["zero_cp_correction_raw"],
                "ge_corr_raw": components["zero_ge_correction_raw"],
                "cp_shared_raw": components["zero_cp_shared_correction_raw"],
                "cp_private_raw": components["zero_cp_private_correction_raw"],
                "ge_shared_raw": components["zero_ge_shared_correction_raw"],
                "ge_private_raw": components["zero_ge_private_correction_raw"],
            }
            prior_centered_components = self._center_correction_components(
                prior_raw_components,
                zero_components,
            )
            prior_cp_corr = prior_centered_components["cp_corr"]
            prior_ge_corr = prior_centered_components["ge_corr"]
            prior_cp_pred = base_cp_pred + self.vae_correction_scale * prior_cp_corr
            prior_ge_pred = base_ge_pred + self.vae_correction_scale * prior_ge_corr

        state["mask_state"] = mask_state
        state["base_cp_pred"] = base_cp_pred
        state["base_ge_pred"] = base_ge_pred
        state["cp_delta_pred"] = cp_pred - x_cp
        state["ge_delta_pred"] = ge_pred - x_ge
        state["cp_correction"] = cp_corr
        state["ge_correction"] = ge_corr
        state["cp_shared_correction"] = components["cp_shared_correction"]
        state["cp_private_correction"] = components["cp_private_correction"]
        state["ge_shared_correction"] = components["ge_shared_correction"]
        state["ge_private_correction"] = components["ge_private_correction"]
        state["cp_correction_raw"] = components["cp_correction_raw"]
        state["ge_correction_raw"] = components["ge_correction_raw"]
        state["cp_shared_correction_raw"] = components["cp_shared_correction_raw"]
        state["cp_private_correction_raw"] = components["cp_private_correction_raw"]
        state["ge_shared_correction_raw"] = components["ge_shared_correction_raw"]
        state["ge_private_correction_raw"] = components["ge_private_correction_raw"]
        zero_cp_correction, zero_ge_correction = self._apply_correction_centering(
            components["zero_cp_correction_raw"],
            components["zero_ge_correction_raw"],
            components["zero_cp_correction_raw"],
            components["zero_ge_correction_raw"],
        )
        state["zero_cp_correction"] = zero_cp_correction
        state["zero_ge_correction"] = zero_ge_correction
        state["zero_cp_correction_raw"] = components["zero_cp_correction_raw"]
        state["zero_ge_correction_raw"] = components["zero_ge_correction_raw"]
        state["zero_cp_shared_correction_raw"] = components["zero_cp_shared_correction_raw"]
        state["zero_cp_private_correction_raw"] = components["zero_cp_private_correction_raw"]
        state["zero_ge_shared_correction_raw"] = components["zero_ge_shared_correction_raw"]
        state["zero_ge_private_correction_raw"] = components["zero_ge_private_correction_raw"]
        state["prior_cp_pred"] = prior_cp_pred
        state["prior_ge_pred"] = prior_ge_pred
        self._cached_vae_state = state
        return cp_pred, ge_pred

    def forward_view(
        self,
        x_cp,
        x_ge,
        features,
        view=VIEW_FULL,
        target_cp=None,
        target_ge=None,
        sample=None,
    ):
        view = self._canonical_view(view)
        masked_cp, masked_ge = self._mask_inputs_for_view(x_cp, x_ge, view)
        previous_disable_mask = getattr(self, "_disable_modal_random_mask", False)
        self._disable_modal_random_mask = True
        try:
            cp_pred, ge_pred = self.forward(
                masked_cp,
                masked_ge,
                features,
                target_cp=target_cp,
                target_ge=target_ge,
                sample=sample,
            )
            state = dict(getattr(self, "_cached_vae_state", {}))
        finally:
            self._disable_modal_random_mask = previous_disable_mask

        result = {
            "view": view,
            "masked_cp_input": masked_cp,
            "masked_ge_input": masked_ge,
            "cp_pred": cp_pred,
            "ge_pred": ge_pred,
            "z_shared": state.get("z_shared"),
            "z_cp": state.get("z_cp"),
            "z_ge": state.get("z_ge"),
            "state": state,
        }
        self._cached_residual_vae_view_state[view] = result
        return result

    def compute_training_loss(
        self,
        control_cp,
        control_ge,
        target_cp,
        target_ge,
        features,
        mol_ids=None,
        force_aux_view=None,
    ):
        full = self.forward_view(
            control_cp,
            control_ge,
            features,
            view=self.VIEW_FULL,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=self.vae_sample_train,
        )
        full_loss = self.loss(
            target_cp=target_cp,
            target_ge=target_ge,
            cp_pred=full["cp_pred"],
            ge_pred=full["ge_pred"],
            control_cp=control_cp,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        full_components = dict(getattr(self, "_last_vae_loss_components", {}))

        total_loss = full_loss
        aux_view = "none"
        geview_loss = full_loss.new_zeros(())
        cpview_loss = full_loss.new_zeros(())
        weighted_geview = full_loss.new_zeros(())
        weighted_cpview = full_loss.new_zeros(())
        geview_components = None
        cpview_components = None

        if self.residual_vae_training_stage == "three_view":
            aux_view = self._choose_aux_view(force_aux_view=force_aux_view)
            aux = self.forward_view(
                control_cp,
                control_ge,
                features,
                view=aux_view,
                target_cp=target_cp,
                target_ge=target_ge,
                sample=self.vae_sample_train,
            )
            aux_loss = self.loss(
                target_cp=target_cp,
                target_ge=target_ge,
                cp_pred=aux["cp_pred"],
                ge_pred=aux["ge_pred"],
                control_cp=control_cp,
                control_ge=control_ge,
                features=features,
                mol_ids=mol_ids,
            )
            aux_components = dict(getattr(self, "_last_vae_loss_components", {}))
            if aux_view == self.VIEW_GE:
                geview_loss = aux_loss
                weighted_geview = self.residual_vae_geview_weight * geview_loss
                geview_components = aux_components
            else:
                cpview_loss = aux_loss
                weighted_cpview = self.residual_vae_cpview_weight * cpview_loss
                cpview_components = aux_components
            total_loss = full_loss + weighted_geview + weighted_cpview

        self._last_vae_loss_components = {
            "total_loss": float(total_loss.detach().cpu()),
            "full_loss": float(full_loss.detach().cpu()),
            "geview_loss": float(geview_loss.detach().cpu()),
            "cpview_loss": float(cpview_loss.detach().cpu()),
            "weighted_geview_loss": float(weighted_geview.detach().cpu()),
            "weighted_cpview_loss": float(weighted_cpview.detach().cpu()),
            "aux_view": aux_view,
            "stage": self.residual_vae_training_stage,
            "full_base_loss": float(full_components.get("base_loss", 0.0)),
            "full_prior_recon_loss": float(full_components.get("prior_recon_loss", 0.0)),
            "full_weighted_kl": float(full_components.get("weighted_kl", 0.0)),
            "full_weighted_disentangle": float(full_components.get("weighted_disentangle", 0.0)),
            "geview_base_loss": 0.0 if geview_components is None else float(geview_components.get("base_loss", 0.0)),
            "cpview_base_loss": 0.0 if cpview_components is None else float(cpview_components.get("base_loss", 0.0)),
        }
        return total_loss

    def _compute_complete_validation_loss(
        self,
        control_cp,
        control_ge,
        target_cp,
        target_ge,
        features,
        mol_ids=None,
    ):
        full = self.forward_view(
            control_cp,
            control_ge,
            features,
            view=self.VIEW_FULL,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=False,
        )
        full_loss = self.loss(
            target_cp=target_cp,
            target_ge=target_ge,
            cp_pred=full["cp_pred"],
            ge_pred=full["ge_pred"],
            control_cp=control_cp,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        full_components = dict(getattr(self, "_last_vae_loss_components", {}))

        geview = self.forward_view(
            control_cp,
            control_ge,
            features,
            view=self.VIEW_GE,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=False,
        )
        geview_loss = self.loss(
            target_cp=target_cp,
            target_ge=target_ge,
            cp_pred=geview["cp_pred"],
            ge_pred=geview["ge_pred"],
            control_cp=control_cp,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        geview_components = dict(getattr(self, "_last_vae_loss_components", {}))

        cpview = self.forward_view(
            control_cp,
            control_ge,
            features,
            view=self.VIEW_CP,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=False,
        )
        cpview_loss = self.loss(
            target_cp=target_cp,
            target_ge=target_ge,
            cp_pred=cpview["cp_pred"],
            ge_pred=cpview["ge_pred"],
            control_cp=control_cp,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        cpview_components = dict(getattr(self, "_last_vae_loss_components", {}))

        if self.residual_vae_training_stage == "three_view":
            weighted_geview = self.residual_vae_geview_weight * geview_loss
            weighted_cpview = self.residual_vae_cpview_weight * cpview_loss
            total_loss = full_loss + weighted_geview + weighted_cpview
        else:
            weighted_geview = full_loss.new_zeros(())
            weighted_cpview = full_loss.new_zeros(())
            total_loss = full_loss

        components = {
            "loss": total_loss,
            "full_loss": full_loss,
            "geview_loss": geview_loss,
            "cpview_loss": cpview_loss,
            "weighted_geview_loss": weighted_geview,
            "weighted_cpview_loss": weighted_cpview,
            "full_base_loss": full_loss.new_tensor(float(full_components.get("base_loss", 0.0))),
            "full_prior_recon_loss": full_loss.new_tensor(float(full_components.get("prior_recon_loss", 0.0))),
            "full_weighted_kl": full_loss.new_tensor(float(full_components.get("weighted_kl", 0.0))),
            "full_weighted_disentangle": full_loss.new_tensor(float(full_components.get("weighted_disentangle", 0.0))),
            "geview_base_loss": full_loss.new_tensor(float(geview_components.get("base_loss", 0.0))),
            "cpview_base_loss": full_loss.new_tensor(float(cpview_components.get("base_loss", 0.0))),
        }
        return full, geview, cpview, components

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
        total = super().loss(
            target_cp=target_cp,
            target_ge=target_ge,
            cp_pred=cp_pred,
            ge_pred=ge_pred,
            control_cp=control_cp,
            control_ge=control_ge,
            features=features,
            mol_ids=mol_ids,
        )
        state = getattr(self, "_cached_vae_state", {})
        base_loss = cp_pred.new_zeros(())
        if state and state.get("base_cp_pred") is not None and state.get("base_ge_pred") is not None:
            base_loss = MVCModel_HyperGate.loss(
                self,
                target_cp=target_cp,
                target_ge=target_ge,
                cp_pred=state["base_cp_pred"],
                ge_pred=state["base_ge_pred"],
                control_cp=control_cp,
                control_ge=control_ge,
                features=features,
                mol_ids=mol_ids,
            )
            total = total + self.vae_base_recon_weight * base_loss
        correction_l2 = cp_pred.new_zeros(())
        if state and state.get("cp_correction") is not None and state.get("ge_correction") is not None:
            correction_l2 = state["cp_correction"].pow(2).mean() + state["ge_correction"].pow(2).mean()
            total = total + self.vae_correction_l2_weight * correction_l2
        private_orth_loss = self._compute_private_orth_loss(cp_pred)
        private_remainder_loss = self._compute_private_remainder_loss(target_cp, target_ge, cp_pred)
        shared_infonce_loss = self._compute_shared_infonce_loss(target_cp, target_ge, cp_pred)
        private_orth_weight = self._private_orth_weight_value()
        private_remainder_weight = self._private_remainder_weight_value()
        shared_infonce_weight = self._shared_infonce_weight_value()
        total = total + private_orth_weight * private_orth_loss
        total = total + private_remainder_weight * private_remainder_loss
        total = total + shared_infonce_weight * shared_infonce_loss
        if hasattr(self, "_last_vae_loss_components"):
            self._last_vae_loss_components["base_loss"] = float(base_loss.detach().cpu())
            self._last_vae_loss_components["weighted_base_loss"] = float(
                (self.vae_base_recon_weight * base_loss).detach().cpu()
            )
            self._last_vae_loss_components["correction_l2"] = float(correction_l2.detach().cpu())
            self._last_vae_loss_components["weighted_correction_l2"] = float(
                (self.vae_correction_l2_weight * correction_l2).detach().cpu()
            )
            self._last_vae_loss_components["private_orth_loss"] = float(private_orth_loss.detach().cpu())
            self._last_vae_loss_components["weighted_private_orth_loss"] = float(
                (private_orth_weight * private_orth_loss).detach().cpu()
            )
            self._last_vae_loss_components["private_remainder_loss"] = float(private_remainder_loss.detach().cpu())
            self._last_vae_loss_components["weighted_private_remainder_loss"] = float(
                (private_remainder_weight * private_remainder_loss).detach().cpu()
            )
            self._last_vae_loss_components["shared_infonce_loss"] = float(shared_infonce_loss.detach().cpu())
            self._last_vae_loss_components["weighted_shared_infonce_loss"] = float(
                (shared_infonce_weight * shared_infonce_loss).detach().cpu()
            )
        return total

    @staticmethod
    def _batch_basic_metrics(true, pred, prefix):
        true = true.float()
        pred = pred.float()
        mse = ((pred - true) ** 2).mean(dim=1)
        rmse = torch.sqrt(mse)
        true_centered = true - true.mean(dim=1, keepdim=True)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        denom = torch.norm(true_centered, dim=1) * torch.norm(pred_centered, dim=1)
        pcc = (true_centered * pred_centered).sum(dim=1) / (denom + 1e-8)
        pcc = torch.where(denom > 1e-8, pcc, torch.zeros_like(pcc))
        return {
            f"{prefix}_mse": mse.detach().cpu().numpy(),
            f"{prefix}_rmse": rmse.detach().cpu().numpy(),
            f"{prefix}_pcc": pcc.detach().cpu().numpy(),
        }

    def _add_view_metrics(self, metrics_dict_all, metrics_dict_all_ls, target_cp, target_ge, view_result, prefix):
        for key, values in self._batch_basic_metrics(target_cp, view_result["cp_pred"], f"{prefix}_cp").items():
            metrics_dict_all[key] += float(np.nansum(values))
            metrics_dict_all_ls[key] += values.tolist()
        for key, values in self._batch_basic_metrics(target_ge, view_result["ge_pred"], f"{prefix}_ge").items():
            metrics_dict_all[key] += float(np.nansum(values))
            metrics_dict_all_ls[key] += values.tolist()

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_value = np.inf
        best_epoch = 0

        for epoch in range(n_epochs):
            if self.vae_kl_warmup_epochs > 0:
                self.current_kl_scale = min(1.0, float(epoch + 1) / float(self.vae_kl_warmup_epochs))
            else:
                self.current_kl_scale = 1.0

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
                loss = self.compute_training_loss(
                    control_cp=control_cp,
                    control_ge=control_ge,
                    target_cp=target_cp,
                    target_ge=target_ge,
                    features=features,
                    mol_ids=mol_id,
                )
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                optimizer.step()

            train_dict, train_metrics_dict, _ = self.test_model(
                loader=train_loader,
                loss_item=["loss"],
                metrics_func=metrics_func,
            )
            train_loss = train_dict["loss"]
            test_dict, test_metrics_dict, _ = self.test_model(
                loader=test_loader,
                loss_item=["loss"],
                metrics_func=metrics_func,
            )
            test_loss = test_dict["loss"]

            if epoch % 10 == 0:
                shared_dropout_desc = ""
                if hasattr(self, "vae_shared_dropout_prob_cp") and hasattr(self, "vae_shared_dropout_prob_ge"):
                    shared_dropout_desc = (
                        f", shared_drop_cp={self.vae_shared_dropout_prob_cp:.3f}, "
                        f"shared_drop_ge={self.vae_shared_dropout_prob_ge:.3f}"
                    )
                elif hasattr(self, "vae_shared_dropout_prob"):
                    shared_dropout_desc = f", shared_drop={self.vae_shared_dropout_prob:.3f}"
                print(
                    f"[Epoch {epoch}] ResidualVAE weights -> "
                    f"prior={self.vae_prior_recon_weight:.6g}, "
                    f"kl={self.vae_kl_weight:.6g}*{self.current_kl_scale:.3f}, "
                    f"disentangle={self.vae_disentangle_weight:.6g}"
                    f"({self.vae_disentangle_mode}), "
                    f"modal_mask={self.vae_modal_mask_prob:.3f}, "
                    f"stage={self.residual_vae_training_stage}, "
                    f"geview={self.residual_vae_geview_weight:.3f}, "
                    f"cpview={self.residual_vae_cpview_weight:.3f}, "
                    f"aux={self.residual_vae_aux_view_mode}"
                    f"{shared_dropout_desc}"
                )
                if hasattr(self, "vae_factorized_correction"):
                    role_desc = [
                        f"factorized={self.vae_factorized_correction}",
                        f"private_orth={getattr(self, 'vae_private_orth_weight', 0.0):.6g}",
                        f"private_remainder={getattr(self, 'vae_private_remainder_weight', 0.0):.6g}",
                    ]
                    if hasattr(self, "vae_shared_infonce_weight"):
                        role_desc.append(
                            f"shared_infonce={getattr(self, 'vae_shared_infonce_weight', 0.0):.6g}"
                        )
                    if hasattr(self, "vae_shared_infonce_temperature"):
                        role_desc.append(
                            f"shared_infonce_temp={getattr(self, 'vae_shared_infonce_temperature', 0.1):.6g}"
                        )
                    print(f"[Epoch {epoch}] ResidualVAE roles -> " + ", ".join(role_desc))
                print(
                    f"[Epoch {epoch}] | train loss: {train_loss:.5f}, "
                    f"valid_loss: {test_loss:.5f}, "
                    f"valid_full: {test_dict.get('full_loss', np.nan):.5f}, "
                    f"valid_GEview: {test_dict.get('geview_loss', np.nan):.5f}, "
                    f"valid_CPview: {test_dict.get('cpview_loss', np.nan):.5f}",
                    flush=True,
                )

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

                _, _, _, loss_components = self._compute_complete_validation_loss(
                    control_cp=control_cp,
                    control_ge=control_ge,
                    target_cp=target_cp,
                    target_ge=target_ge,
                    features=mol_features,
                    mol_ids=mol_id,
                )

                if loss_item is not None:
                    for k, v in loss_components.items():
                        test_dict[k] += float(v.detach().cpu()) * batch_size

                if metrics_func is not None:
                    # Metrics should match the paper-facing artifact path:
                    # target-free prior inference, not the posterior used for loss.
                    full = self.forward_view(
                        control_cp,
                        control_ge,
                        mol_features,
                        view=self.VIEW_FULL,
                        sample=False,
                    )
                    geview = self.forward_view(
                        control_cp,
                        control_ge,
                        mol_features,
                        view=self.VIEW_GE,
                        sample=False,
                    )
                    cpview = self.forward_view(
                        control_cp,
                        control_ge,
                        mol_features,
                        view=self.VIEW_CP,
                        sample=False,
                    )
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(
                        control_cp,
                        control_ge,
                        target_cp,
                        target_ge,
                        full["cp_pred"],
                        full["ge_pred"],
                        metrics_func=metrics_func,
                        perturbed_centroid_CP=perturbed_centroid_CP,
                        perturbed_centroid_GE=perturbed_centroid_GE,
                    )
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]
                    self._add_view_metrics(metrics_dict_all, metrics_dict_all_ls, target_cp, target_ge, full, "full")
                    self._add_view_metrics(metrics_dict_all, metrics_dict_all_ls, target_cp, target_ge, geview, "geview")
                    self._add_view_metrics(metrics_dict_all, metrics_dict_all_ls, target_cp, target_ge, cpview, "cpview")
                    metrics_dict_all_ls["cp_id"] += list(mol_id)

        if test_size == 0:
            return test_dict, metrics_dict_all, metrics_dict_all_ls

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size
        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size
        return test_dict, metrics_dict_all, metrics_dict_all_ls

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

                full = self.forward_view(control_cp, control_ge, mol_features, view=self.VIEW_FULL, sample=False)
                control_cp_list.append(control_cp.cpu().numpy().astype(float))
                control_ge_list.append(control_ge.cpu().numpy().astype(float))
                target_cp_list.append(target_cp.cpu().numpy().astype(float))
                target_ge_list.append(target_ge.cpu().numpy().astype(float))
                cp_pred_list.append(full["cp_pred"].cpu().numpy().astype(float))
                ge_pred_list.append(full["ge_pred"].cpu().numpy().astype(float))
                smiles_list.extend(list(mol_id))

        return (
            np.concatenate(control_cp_list, axis=0),
            np.concatenate(control_ge_list, axis=0),
            np.concatenate(target_cp_list, axis=0),
            np.concatenate(target_ge_list, axis=0),
            np.concatenate(cp_pred_list, axis=0),
            np.concatenate(ge_pred_list, axis=0),
            np.array(smiles_list),
        )

    def predict_all_views_and_latents(self, loader):
        profile_lists = defaultdict(list)
        latent_lists = defaultdict(list)
        smiles_list = []

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp = control_cp.to(self.dev)
                control_ge = control_ge.to(self.dev)
                target_cp = target_cp.to(self.dev)
                target_ge = target_ge.to(self.dev)
                mol_features = mol_features.to(self.dev)

                outputs = {
                    "full": self.forward_view(control_cp, control_ge, mol_features, view=self.VIEW_FULL, sample=False),
                    "geview": self.forward_view(control_cp, control_ge, mol_features, view=self.VIEW_GE, sample=False),
                    "cpview": self.forward_view(control_cp, control_ge, mol_features, view=self.VIEW_CP, sample=False),
                }

                profile_lists["control_cp"].append(control_cp.cpu().numpy().astype(float))
                profile_lists["control_ge"].append(control_ge.cpu().numpy().astype(float))
                profile_lists["target_cp"].append(target_cp.cpu().numpy().astype(float))
                profile_lists["target_ge"].append(target_ge.cpu().numpy().astype(float))
                for view_name, result in outputs.items():
                    profile_lists[f"{view_name}_cp_pred"].append(result["cp_pred"].cpu().numpy().astype(float))
                    profile_lists[f"{view_name}_ge_pred"].append(result["ge_pred"].cpu().numpy().astype(float))
                    for latent_key in ("z_shared", "z_cp", "z_ge"):
                        latent_tensor = result.get(latent_key)
                        if latent_tensor is not None:
                            latent_lists[f"{view_name}_{latent_key}"].append(
                                latent_tensor.cpu().numpy().astype(np.float32)
                            )
                smiles_list.extend(list(mol_id))

        profiles = {key: np.concatenate(value, axis=0) for key, value in profile_lists.items()}
        profiles["smiles"] = np.array(smiles_list)
        latents = {key: np.concatenate(value, axis=0) for key, value in latent_lists.items()}
        latents["smiles"] = np.array(smiles_list)
        return profiles, latents


class _MVCModel_HyperGateResidualVAE_SingleBranch(MVCModel_HyperGateResidualVAE):
    """Clean single-modality single-target control on the MVCPert architecture."""

    target_branch = None

    def _branch_config(self):
        if self.target_branch == "ge":
            return {
                "branch_name": "GE",
                "target_key": "target_ge",
                "control_key": "control_ge",
                "zero_input": "cp",
                "prior": self.ge_prior,
                "posterior": self.ge_posterior,
                "delta_encoder": self.ge_delta_encoder,
                "pcc_weight": float(self.ge_pcc_weight),
                "delta_pcc_weight": float(self.ge_delta_pcc_weight),
            }
        if self.target_branch == "cp":
            return {
                "branch_name": "CP",
                "target_key": "target_cp",
                "control_key": "control_cp",
                "zero_input": "ge",
                "prior": self.cp_prior,
                "posterior": self.cp_posterior,
                "delta_encoder": self.cp_delta_encoder,
                "pcc_weight": float(self.cp_pcc_weight),
                "delta_pcc_weight": float(self.cp_delta_pcc_weight),
            }
        raise ValueError(f"Unsupported target_branch: {self.target_branch!r}")

    def _compute_single_branch_disentangle(self, shared_mu, branch_mu):
        mode = self.vae_disentangle_mode
        if mode == "none":
            return shared_mu.new_zeros(())
        if mode == "corr":
            return self._cross_correlation_penalty(shared_mu, branch_mu)
        if mode == "hsic":
            return self._hsic_penalty(shared_mu, branch_mu)
        raise ValueError(f"Unsupported residual_vae_disentangle_mode: {mode}")

    def _build_single_branch_latents(self, condition, branch_fused, branch_target, base_branch_pred, branch):
        prior_shared_mu, prior_shared_logvar = self.shared_prior(condition)
        prior_branch_mu, prior_branch_logvar = branch["prior"](branch_fused)

        use_posterior = branch_target is not None
        sample = self.training and self.vae_sample_train

        if use_posterior:
            branch_corr_target = branch_target - base_branch_pred
            branch_corr_embed = branch["delta_encoder"](branch_corr_target)
            zero_embed = torch.zeros_like(branch_corr_embed)
            if self.target_branch == "ge":
                shared_input = torch.cat([condition, zero_embed, branch_corr_embed], dim=1)
            else:
                shared_input = torch.cat([condition, branch_corr_embed, zero_embed], dim=1)
            branch_input = torch.cat([branch_fused, branch_corr_embed], dim=1)
            shared_mu, shared_logvar = self.shared_posterior(shared_input)
            branch_mu, branch_logvar = branch["posterior"](branch_input)
        else:
            shared_mu, shared_logvar = prior_shared_mu, prior_shared_logvar
            branch_mu, branch_logvar = prior_branch_mu, prior_branch_logvar

        z_shared = self._reparameterize(shared_mu, shared_logvar, sample=sample)
        z_branch = self._reparameterize(branch_mu, branch_logvar, sample=sample)
        if self.target_branch == "ge":
            z_cp = torch.zeros_like(z_branch)
            z_ge = z_branch
        else:
            z_cp = z_branch
            z_ge = torch.zeros_like(z_branch)

        state = {
            "use_posterior": use_posterior,
            "shared_mu": shared_mu,
            "shared_logvar": shared_logvar,
            "branch_mu": branch_mu,
            "branch_logvar": branch_logvar,
            "prior_shared_mu": prior_shared_mu,
            "prior_shared_logvar": prior_shared_logvar,
            "prior_branch_mu": prior_branch_mu,
            "prior_branch_logvar": prior_branch_logvar,
            "z_shared": z_shared,
            "z_cp": z_cp,
            "z_ge": z_ge,
        }
        return z_shared, z_cp, z_ge, state

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        branch = self._branch_config()
        if sample is None:
            sample = self.training and self.vae_sample_train
        if branch["zero_input"] == "cp":
            x_cp = torch.zeros_like(x_cp)
        else:
            x_ge = torch.zeros_like(x_ge)

        condition, cp_fused, ge_fused = self._encode_condition(x_cp, x_ge, features)
        base_cp_pred, base_ge_pred = self._predict_base_from_fused(cp_fused, ge_fused)
        branch_fused = ge_fused if self.target_branch == "ge" else cp_fused
        base_branch_pred = base_ge_pred if self.target_branch == "ge" else base_cp_pred
        branch_target = target_ge if self.target_branch == "ge" else target_cp

        z_shared, z_cp, z_ge, state = self._build_single_branch_latents(
            condition=condition,
            branch_fused=branch_fused,
            branch_target=branch_target,
            base_branch_pred=base_branch_pred,
            branch=branch,
        )
        if not sample:
            z_shared = state["shared_mu"]
            if self.target_branch == "ge":
                z_ge = state["branch_mu"]
                z_cp = torch.zeros_like(z_ge)
            else:
                z_cp = state["branch_mu"]
                z_ge = torch.zeros_like(z_cp)

        zero_shared, zero_cp, zero_ge = self._zero_latents_like(z_shared, z_cp, z_ge)
        zero_components = self._decode_raw_correction_components(
            cp_fused,
            ge_fused,
            zero_shared,
            zero_cp,
            zero_ge,
        )
        raw_components = self._decode_raw_correction_components(
            cp_fused,
            ge_fused,
            z_shared,
            z_cp,
            z_ge,
            z_shared_ge=z_shared,
        )
        centered_components = self._center_correction_components(raw_components, zero_components)
        cp_corr = centered_components["cp_corr"]
        ge_corr = centered_components["ge_corr"]
        cp_pred = base_cp_pred + self.vae_correction_scale * cp_corr
        ge_pred = base_ge_pred + self.vae_correction_scale * ge_corr

        prior_cp_pred = None
        prior_ge_pred = None
        if branch_target is not None:
            if self.target_branch == "ge":
                prior_cp_mu = torch.zeros_like(state["prior_branch_mu"])
                prior_ge_mu = state["prior_branch_mu"]
            else:
                prior_cp_mu = state["prior_branch_mu"]
                prior_ge_mu = torch.zeros_like(state["prior_branch_mu"])
            prior_raw_components = self._decode_raw_correction_components(
                cp_fused,
                ge_fused,
                state["prior_shared_mu"],
                prior_cp_mu,
                prior_ge_mu,
                z_shared_ge=state["prior_shared_mu"],
            )
            prior_centered_components = self._center_correction_components(
                prior_raw_components,
                zero_components,
            )
            prior_cp_pred = base_cp_pred + self.vae_correction_scale * prior_centered_components["cp_corr"]
            prior_ge_pred = base_ge_pred + self.vae_correction_scale * prior_centered_components["ge_corr"]

        branch_pred = ge_pred if self.target_branch == "ge" else cp_pred
        branch_correction = ge_corr if self.target_branch == "ge" else cp_corr
        prior_branch_pred = prior_ge_pred if self.target_branch == "ge" else prior_cp_pred
        base_branch_pred = base_ge_pred if self.target_branch == "ge" else base_cp_pred

        state["base_cp_pred"] = base_cp_pred
        state["base_ge_pred"] = base_ge_pred
        state["cp_correction"] = cp_corr
        state["ge_correction"] = ge_corr
        state["branch_name"] = branch["branch_name"]
        state["branch_pred"] = branch_pred
        state["branch_correction"] = branch_correction
        state["base_branch_pred"] = base_branch_pred
        state["prior_branch_pred"] = prior_branch_pred
        self._cached_vae_state = state
        return branch_pred

    def _single_branch_loss(self, branch_target, branch_pred, branch_control=None, features=None, mol_ids=None):
        branch = self._branch_config()
        mse = F.mse_loss(branch_pred, branch_target, reduction="mean")
        profile_pcc = self.pearson_loss(branch_pred, branch_target)
        delta_pcc = branch_pred.new_zeros(())
        if branch_control is not None:
            delta_pcc = self.pearson_loss(branch_pred - branch_control, branch_target - branch_control)
        total = mse + branch["pcc_weight"] * profile_pcc + branch["delta_pcc_weight"] * delta_pcc
        dose_loss = branch_pred.new_zeros(())
        if self.target_branch == "ge":
            dose_loss = self._compute_dose_ordinal_aux_loss(
                ge_pred=branch_pred,
                target_ge=branch_target,
                control_ge=branch_control,
                features=features,
                mol_ids=mol_ids,
            )
            total = total + dose_loss
        return total, {
            "mse": mse.detach(),
            "profile_pcc": profile_pcc.detach(),
            "delta_pcc": delta_pcc.detach(),
            "dose_loss": dose_loss.detach(),
        }

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
        branch = self._branch_config()
        branch_target = target_ge if self.target_branch == "ge" else target_cp
        branch_pred = ge_pred if self.target_branch == "ge" else cp_pred
        branch_control = control_ge if self.target_branch == "ge" else control_cp
        total, branch_metrics = self._single_branch_loss(
            branch_target=branch_target,
            branch_pred=branch_pred,
            branch_control=branch_control,
            features=features,
            mol_ids=mol_ids,
        )

        state = getattr(self, "_cached_vae_state", {})
        zero = branch_pred.new_zeros(())

        prior_recon_loss = zero
        if state and state.get("prior_branch_pred") is not None:
            prior_recon_loss, _ = self._single_branch_loss(
                branch_target=branch_target,
                branch_pred=state["prior_branch_pred"],
                branch_control=branch_control,
                features=features,
                mol_ids=mol_ids,
            )

        kl_loss = zero
        disentangle_loss = zero
        if state and state.get("use_posterior", False):
            kl_shared = self._gaussian_kl(
                state["shared_mu"],
                state["shared_logvar"],
                state["prior_shared_mu"],
                state["prior_shared_logvar"],
            )
            kl_branch = self._gaussian_kl(
                state["branch_mu"],
                state["branch_logvar"],
                state["prior_branch_mu"],
                state["prior_branch_logvar"],
            )
            kl_loss = kl_shared + kl_branch
            disentangle_loss = self._compute_single_branch_disentangle(
                state["shared_mu"],
                state["branch_mu"],
            )

        base_loss = zero
        if state and state.get("base_branch_pred") is not None:
            base_loss, _ = self._single_branch_loss(
                branch_target=branch_target,
                branch_pred=state["base_branch_pred"],
                branch_control=branch_control,
                features=features,
                mol_ids=mol_ids,
            )

        correction_l2 = zero
        if state and state.get("branch_correction") is not None:
            correction_l2 = state["branch_correction"].pow(2).mean()

        total = (
            total
            + self.vae_prior_recon_weight * prior_recon_loss
            + self.vae_kl_weight * self.current_kl_scale * kl_loss
            + self.vae_disentangle_weight * disentangle_loss
            + self.vae_base_recon_weight * base_loss
            + self.vae_correction_l2_weight * correction_l2
        )

        self._last_vae_loss_components = {
            "branch": branch["branch_name"],
            "base_loss": float(base_loss.detach().cpu()),
            "weighted_base_loss": float((self.vae_base_recon_weight * base_loss).detach().cpu()),
            "prior_recon_loss": float(prior_recon_loss.detach().cpu()),
            "weighted_prior_recon": float((self.vae_prior_recon_weight * prior_recon_loss).detach().cpu()),
            "weighted_kl": float((self.vae_kl_weight * self.current_kl_scale * kl_loss).detach().cpu()),
            "weighted_disentangle": float((self.vae_disentangle_weight * disentangle_loss).detach().cpu()),
            "correction_l2": float(correction_l2.detach().cpu()),
            "weighted_correction_l2": float((self.vae_correction_l2_weight * correction_l2).detach().cpu()),
            "branch_mse": float(branch_metrics["mse"].cpu()),
            "branch_profile_pcc": float(branch_metrics["profile_pcc"].cpu()),
            "branch_delta_pcc": float(branch_metrics["delta_pcc"].cpu()),
            "branch_dose_loss": float(branch_metrics["dose_loss"].cpu()),
        }
        return total

    def _move_batch_to_device(self, control_cp, control_ge, target_cp, target_ge, features):
        return (
            control_cp.to(self.dev),
            control_ge.to(self.dev),
            target_cp.to(self.dev),
            target_ge.to(self.dev),
            features.to(self.dev),
        )

    def train_model(self, learning_rate, weight_decay, n_epochs, train_loader, test_loader, save_model=True, metrics_func=None):
        epoch_hist = defaultdict(list)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_item = ["loss"]

        best_value = np.inf
        best_epoch = 0

        for epoch in range(n_epochs):
            if self.vae_kl_warmup_epochs > 0:
                self.current_kl_scale = min(1.0, float(epoch + 1) / float(self.vae_kl_warmup_epochs))
            else:
                self.current_kl_scale = 1.0

            self.train()
            for control_cp, control_ge, target_cp, target_ge, features, mol_id in train_loader:
                control_cp, control_ge, target_cp, target_ge, features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, features
                )
                if control_cp.shape[0] == 1:
                    continue

                optimizer.zero_grad()
                branch_pred = self.forward(
                    control_cp,
                    control_ge,
                    features,
                    target_cp=target_cp,
                    target_ge=target_ge,
                    sample=self.vae_sample_train,
                )
                if self.target_branch == "ge":
                    cp_pred = torch.zeros_like(target_cp)
                    ge_pred = branch_pred
                else:
                    cp_pred = branch_pred
                    ge_pred = torch.zeros_like(target_ge)
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
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
                optimizer.step()

            train_dict, train_metrics_dict, _ = self.test_model(loader=train_loader, loss_item=loss_item, metrics_func=metrics_func)
            test_dict, test_metrics_dict, _ = self.test_model(loader=test_loader, loss_item=loss_item, metrics_func=metrics_func)
            train_loss = train_dict["loss"]
            test_loss = test_dict["loss"]

            if epoch % 10 == 0:
                print(
                    f"[Epoch {epoch}] Single-branch MVCPert ({self._branch_config()['branch_name']}-only) -> "
                    f"prior={self.vae_prior_recon_weight:.6g}, "
                    f"kl={self.vae_kl_weight:.6g}*{self.current_kl_scale:.3f}, "
                    f"disentangle={self.vae_disentangle_weight:.6g}({self.vae_disentangle_mode})"
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

    def _eval_single_branch_reconstruction(
        self,
        control_branch,
        target_branch,
        pred_branch,
        metrics_func=None,
        perturbed_centroid=None,
    ):
        metrics_func = metrics_func or ["pearson"]
        use_torch = pred_branch.is_cuda
        target_branch = target_branch.float()
        pred_branch = pred_branch.float()
        control_branch = control_branch.float()
        device = target_branch.device

        if perturbed_centroid is None:
            perturbed_centroid = target_branch.mean(dim=0)
        elif use_torch:
            perturbed_centroid = torch.tensor(perturbed_centroid, dtype=target_branch.dtype, device=device)
        else:
            perturbed_centroid = np.asarray(perturbed_centroid, float)

        delta_true = target_branch - control_branch
        delta_pred = pred_branch - control_branch
        systema_true = target_branch - perturbed_centroid
        systema_pred = pred_branch - perturbed_centroid

        metrics_dict = defaultdict(float)
        metrics_dict_ls = defaultdict(list)

        def batch_pearson(x, y):
            x0 = x - x.mean(dim=1, keepdim=True)
            y0 = y - y.mean(dim=1, keepdim=True)
            num = (x0 * y0).sum(dim=1)
            den = torch.norm(x0, dim=1) * torch.norm(y0, dim=1) + 1e-8
            out = num / den
            return out.cpu().numpy() if use_torch else out

        def batch_rmse(x, y):
            se = ((x - y) ** 2).mean(dim=1)
            out = torch.sqrt(se)
            return out.cpu().numpy() if use_torch else np.sqrt(se)

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
            x_true = x_true.cpu().numpy() if hasattr(x_true, "cpu") else x_true
            x_pred = x_pred.cpu().numpy() if hasattr(x_pred, "cpu") else x_pred
            n_rows, n_cols = x_true.shape
            topk_pred = np.argpartition(x_pred, -k, axis=1)[:, -k:]
            top_pos_true = np.argpartition(x_true, -pos_num, axis=1)[:, -pos_num:]
            top_neg_true = np.argpartition(x_true, neg_num, axis=1)[:, :neg_num]
            pred_mask = np.zeros((n_rows, n_cols), dtype=bool)
            pred_mask[np.arange(n_rows)[:, None], topk_pred] = True
            pos_mask = np.zeros((n_rows, n_cols), dtype=bool)
            pos_mask[np.arange(n_rows)[:, None], top_pos_true] = True
            neg_mask = np.zeros((n_rows, n_cols), dtype=bool)
            neg_mask[np.arange(n_rows)[:, None], top_neg_true] = True
            pos_hits = (pred_mask & pos_mask).sum(axis=1)
            neg_hits = (pred_mask & neg_mask).sum(axis=1)
            return (neg_hits / k, pos_hits / k)

        branch_upper = self._branch_config()["branch_name"]
        branch_lower = branch_upper.lower()

        if "pearson" in metrics_func:
            systema_pcc = batch_pearson(systema_true, systema_pred)
            metrics_dict[f"systema_pearson_{branch_upper}"] = float(np.nansum(systema_pcc))
            metrics_dict_ls[f"systema_pearson_{branch_upper}"] = systema_pcc.tolist()

            pred_pcc = batch_pearson(target_branch, pred_branch)
            metrics_dict_ls[f"{branch_lower}_pred_pearson"] = pred_pcc.tolist()

            delta_pcc = batch_pearson(delta_true, delta_pred)
            metrics_dict_ls[f"DEG_{branch_lower}_pred_pearson"] = delta_pcc.tolist()

        if "rmse" in metrics_func:
            systema_rmse = batch_rmse(systema_true, systema_pred)
            metrics_dict[f"systema_rmse_{branch_upper}"] = float(np.nansum(systema_rmse))
            metrics_dict_ls[f"systema_rmse_{branch_upper}"] = systema_rmse.tolist()

            pred_rmse = batch_rmse(target_branch, pred_branch)
            metrics_dict_ls[f"{branch_lower}_pred_rmse"] = pred_rmse.tolist()

            delta_rmse = batch_rmse(delta_true, delta_pred)
            metrics_dict_ls[f"DEG_{branch_lower}_pred_rmse"] = delta_rmse.tolist()

        for metric_name in metrics_func:
            if metric_name.startswith("precision"):
                k = int(metric_name[len("precision"):])
                neg_hits, pos_hits = batch_precision_at_k(target_branch, pred_branch, k)
                metrics_dict[f"{branch_lower}_pred_neg_{metric_name}"] = float(neg_hits.sum().item() if use_torch else neg_hits.sum())
                metrics_dict[f"{branch_lower}_pred_pos_{metric_name}"] = float(pos_hits.sum().item() if use_torch else pos_hits.sum())
                metrics_dict_ls[f"{branch_lower}_pred_neg_{metric_name}"] = neg_hits.tolist()
                metrics_dict_ls[f"{branch_lower}_pred_pos_{metric_name}"] = pos_hits.tolist()

                neg_hits, pos_hits = batch_precision_at_k(delta_true, delta_pred, k)
                metrics_dict[f"DEG_{branch_lower}_pred_neg_{metric_name}"] = float(neg_hits.sum().item() if use_torch else neg_hits.sum())
                metrics_dict[f"DEG_{branch_lower}_pred_pos_{metric_name}"] = float(pos_hits.sum().item() if use_torch else pos_hits.sum())
                metrics_dict_ls[f"DEG_{branch_lower}_pred_neg_{metric_name}"] = neg_hits.tolist()
                metrics_dict_ls[f"DEG_{branch_lower}_pred_pos_{metric_name}"] = pos_hits.tolist()

        return metrics_dict, metrics_dict_ls

    def test_model(self, loader, loss_item=None, metrics_func=None, perturbed_centroid_CP=None, perturbed_centroid_GE=None):
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        self.eval()
        with torch.no_grad():
            for control_cp, control_ge, target_cp, target_ge, mol_features, mol_id in loader:
                control_cp, control_ge, target_cp, target_ge, mol_features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, mol_features
                )
                batch_size = control_cp.shape[0]
                test_size += batch_size
                branch_pred = self.forward(control_cp, control_ge, mol_features, sample=False)
                if self.target_branch == "ge":
                    cp_pred = torch.zeros_like(target_cp)
                    ge_pred = branch_pred
                    branch_control = control_ge
                    branch_target = target_ge
                    perturbed_centroid = perturbed_centroid_GE
                else:
                    cp_pred = branch_pred
                    ge_pred = torch.zeros_like(target_ge)
                    branch_control = control_cp
                    branch_target = target_cp
                    perturbed_centroid = perturbed_centroid_CP

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
                    for key in loss_item:
                        test_dict[key] += loss_val.item() * batch_size

                if metrics_func is not None:
                    metrics_dict, metrics_dict_ls = self._eval_single_branch_reconstruction(
                        control_branch=branch_control,
                        target_branch=branch_target,
                        pred_branch=branch_pred,
                        metrics_func=metrics_func,
                        perturbed_centroid=perturbed_centroid,
                    )
                    for key, value in metrics_dict.items():
                        metrics_dict_all[key] += value
                    for key, value in metrics_dict_ls.items():
                        metrics_dict_all_ls[key] += value
                    metrics_dict_all_ls["cp_id"] += list(mol_id)

        if test_size == 0:
            return test_dict, metrics_dict_all, metrics_dict_all_ls
        for key in test_dict.keys():
            test_dict[key] = test_dict[key] / test_size
        for key in metrics_dict_all.keys():
            metrics_dict_all[key] = metrics_dict_all[key] / test_size
        return test_dict, metrics_dict_all, metrics_dict_all_ls

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
                control_cp, control_ge, target_cp, target_ge, mol_features = self._move_batch_to_device(
                    control_cp, control_ge, target_cp, target_ge, mol_features
                )
                branch_pred = self.forward(control_cp, control_ge, mol_features, sample=False)

                if self.target_branch == "ge":
                    cp_pred = torch.zeros_like(target_cp)
                    ge_pred = branch_pred
                else:
                    cp_pred = branch_pred
                    ge_pred = torch.zeros_like(target_ge)

                control_cp_list.append(control_cp.cpu().numpy().astype(float))
                control_ge_list.append(control_ge.cpu().numpy().astype(float))
                target_cp_list.append(target_cp.cpu().numpy().astype(float))
                target_ge_list.append(target_ge.cpu().numpy().astype(float))
                cp_pred_list.append(cp_pred.cpu().numpy().astype(float))
                ge_pred_list.append(ge_pred.cpu().numpy().astype(float))
                smiles_list.extend(list(mol_id))

        return (
            np.concatenate(control_cp_list, axis=0),
            np.concatenate(control_ge_list, axis=0),
            np.concatenate(target_cp_list, axis=0),
            np.concatenate(target_ge_list, axis=0),
            np.concatenate(cp_pred_list, axis=0),
            np.concatenate(ge_pred_list, axis=0),
            np.array(smiles_list),
        )


class MVCModel_HyperGateResidualVAE_NoCP(_MVCModel_HyperGateResidualVAE_SingleBranch):
    """Clean GE-only control: GE input to GE target only."""

    target_branch = "ge"


class MVCModel_HyperGateResidualVAE_NoGE(_MVCModel_HyperGateResidualVAE_SingleBranch):
    """Clean CP-only control: CP input to CP target only."""

    target_branch = "cp"


class MVCModel_HyperGateResidualVAE_MaskCPInput(MVCModel_HyperGateResidualVAE):
    """Fair MVCPert ablation: zero CP input while training the same dual-head task."""

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        masked_cp = torch.zeros_like(x_cp)
        return super().forward(
            masked_cp,
            x_ge,
            features,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
        )


class MVCModel_HyperGateResidualVAE_MaskGEInput(MVCModel_HyperGateResidualVAE):
    """Fair MVCPert ablation: zero GE input while training the same dual-head task."""

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        masked_ge = torch.zeros_like(x_ge)
        return super().forward(
            x_cp,
            masked_ge,
            features,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
        )


class MVCModel_HyperGateResidualVAE_RandomCPInput(MVCModel_HyperGateResidualVAE):
    """Fair MVCPert ablation: replace CP input with a batch-permuted CP control."""

    @staticmethod
    def _permute_batch(x):
        if x.shape[0] <= 1:
            return x
        perm = torch.roll(torch.arange(x.shape[0], device=x.device), shifts=1)
        return x.index_select(0, perm)

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        random_cp = self._permute_batch(x_cp)
        return super().forward(
            random_cp,
            x_ge,
            features,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
        )


class MVCModel_HyperGateResidualVAE_RandomGEInput(MVCModel_HyperGateResidualVAE):
    """Fair MVCPert ablation: replace GE input with a batch-permuted GE control."""

    @staticmethod
    def _permute_batch(x):
        if x.shape[0] <= 1:
            return x
        perm = torch.roll(torch.arange(x.shape[0], device=x.device), shifts=1)
        return x.index_select(0, perm)

    def forward(self, x_cp, x_ge, features, target_cp=None, target_ge=None, sample=None):
        random_ge = self._permute_batch(x_ge)
        return super().forward(
            x_cp,
            random_ge,
            features,
            target_cp=target_cp,
            target_ge=target_ge,
            sample=sample,
        )
