from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from model import BRANCHES, frozen
from losses import (
    kl_loss,
    classification_loss,
    direct_CMI_loss,
    discriminator_loss,
    encoder_objective,
    learning_relevant_zr,
    reconstruction_loss,
    tc_encoder_loss,
)

from sklearn.metrics import roc_auc_score
from losses import probabilities

@dataclass
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5

    beta: float = 100.0
    beta_warmup_epochs: int = 50

    lambda_cls: float = 1000.0
    lambda_tc: float = 1.0

    lambda_fair_yzy: float = 100.0
    lambda_fair_szs: float = 100.0
    lambda_fair_yzs: float = 1.0
    lambda_fair_szy: float = 1.0

    lri_alpha: float = 1.0
    lri_beta: float = 1.0

    # Enable CMI after the first teo training epochs.
    start_cmi_epoch: int = 3

    num_y_classes: int = 2
    num_s_classes: int = 3

    verbose: bool = True


def get_beta(epoch, beta, warmup_epochs):
    """Return the KL coefficient for a one-based epoch index."""
    if warmup_epochs <= 0:
        return float(beta)

    return float(beta) * min(1.0, epoch / warmup_epochs)

def macro_ovr_auc(y_true, probabilities, n_classes):
    """Compute macro one-vs-rest AUC over available classes."""
    class_aucs = []

    for class_id in range(n_classes):
        binary_true = (y_true == class_id).astype(int)

        if len(np.unique(binary_true)) == 2:
            class_aucs.append(
                roc_auc_score(
                    binary_true,
                    probabilities[:, class_id],
                )
            )

    return float(np.mean(class_aucs)) if class_aucs else np.nan

class CADVAETrainer:
    """Train CAD-VAE using batches ordered as (x, y, s)."""

    def __init__(self, model, cfg: TrainConfig, device=None):
        self.cfg = cfg

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)

        self.optimizer_vae = AdamW(
            self.model.vae_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.optimizer_main = AdamW(
            self.model.main_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.optimizer_opponent = AdamW(
            self.model.opponent_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.optimizer_discriminator = AdamW(
            self.model.discriminator.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.history = []
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None
        self.current_epoch = 0

    def _prepare_batch(self, batch):
        x, y, s = batch

        x = x.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.long).reshape(-1)
        s = s.to(self.device, dtype=torch.long).reshape(-1)

        return x, y, s

    @staticmethod
    def _accuracy(logits, labels, num_classes):
        if num_classes == 2:
            predictions = (logits.reshape(-1) >= 0).long()
        else:
            predictions = logits.argmax(dim=-1)

        return (predictions == labels).float().mean()

    def _forward_losses(
        self,
        x,
        y,
        s,
        sample,
        beta_kl,
        include_cmi,
    ):
        cfg = self.cfg

        recon_logits, z, mu, logvar = self.model(
            x,
            sample=sample,
        )

        loss_recon = reconstruction_loss(recon_logits, x)

        loss_kl = kl_loss(mu, logvar)

        # Main classification updates the encoder and the main heads.
        pred_y = self.model.cls_y(
            torch.cat([z["y"], z["r"]], dim=-1)
        )

        pred_s = self.model.cls_s(
            torch.cat([z["s"], z["r"]], dim=-1)
        )

        loss_y = classification_loss(
            pred_y,
            y,
            cfg.num_y_classes,
        )

        loss_s = classification_loss(
            pred_s,
            s,
            cfg.num_s_classes,
        )

        # Freeze opponents before the forward pass.
        with frozen(self.model.opp_y, self.model.opp_s):
            cmi_y, cmi_s = direct_CMI_loss(
                self.model.opp_y,
                self.model.opp_s,
                z["y"],
                z["s"],
                num_y_classes=cfg.num_y_classes,
                num_s_classes=cfg.num_s_classes,
            )

        # LRI uses fixed main heads and detached private representations.
        with frozen(self.model.cls_y, self.model.cls_s):
            relevance_y, relevance_s = learning_relevant_zr(
                self.model.cls_y,
                self.model.cls_s,
                z["y"].detach(),
                z["r"],
                z["s"].detach(),
                y,
                s,
                alpha=cfg.lri_alpha,
                beta=cfg.lri_beta,
                num_y_classes=cfg.num_y_classes,
                num_s_classes=cfg.num_s_classes,
            )

        # TC updates the encoder through a fixed discriminator.
        with frozen(self.model.discriminator):
            loss_tc = tc_encoder_loss(
                self.model.discriminator,
                z["y"],
                z["r"],
                z["s"],
            )

        loss = encoder_objective(
            loss_recon=loss_recon,
            loss_kl=loss_kl,
            loss_y=loss_y,
            loss_s=loss_s,
            cmi_y=cmi_y,
            cmi_s=cmi_s,
            relevance_y=relevance_y,
            relevance_s=relevance_s,
            loss_tc=loss_tc,
            beta_kl=beta_kl,
            lambda_cls=cfg.lambda_cls,
            lambda_tc=cfg.lambda_tc,
            lambda_fair_yzs=cfg.lambda_fair_yzs,
            lambda_fair_szy=cfg.lambda_fair_szy,
            lambda_fair_yzy=cfg.lambda_fair_yzy,
            lambda_fair_szs=cfg.lambda_fair_szs,
            include_cmi=include_cmi,
        )

        # Supervised opponent training must not update the encoder.
        pred_y_op = self.model.opp_y(z["s"].detach())
        pred_s_op = self.model.opp_s(z["y"].detach())

        loss_y_op = classification_loss(
            pred_y_op,
            y,
            cfg.num_y_classes,
        )

        loss_s_op = classification_loss(
            pred_s_op,
            s,
            cfg.num_s_classes,
        )

        loss_opponent = cfg.lambda_cls * (
            loss_y_op + loss_s_op
        )

        values = {
            "loss": loss,
            "recon": loss_recon,
            "kl": loss_kl,
            "cls_y": loss_y,
            "cls_s": loss_s,
            "cmi_y": cmi_y,
            "cmi_s": cmi_s,
            "relevance_y": relevance_y,
            "relevance_s": relevance_s,
            "tc": loss_tc,
            "opponent_y": loss_y_op,
            "opponent_s": loss_s_op,
            "opponent_loss": loss_opponent,
            "acc_y": self._accuracy(
                pred_y,
                y,
                cfg.num_y_classes,
            ),
            "acc_s": self._accuracy(
                pred_s,
                s,
                cfg.num_s_classes,
            ),
        }

        metrics = {
            name: value.detach().item()
            for name, value in values.items()
        }

        return loss, loss_opponent, metrics, pred_y, pred_s, pred_y_op, pred_s_op

    def _train_batch(
        self,
        x,
        y,
        s,
        beta_kl,
        include_cmi,
    ):
        self.optimizer_vae.zero_grad(set_to_none=True)
        self.optimizer_main.zero_grad(set_to_none=True)
        self.optimizer_opponent.zero_grad(set_to_none=True)
        self.optimizer_discriminator.zero_grad(set_to_none=True)

        loss, loss_opponent, metrics, _, _, _, _ = self._forward_losses(
            x,
            y,
            s,
            sample=True,
            beta_kl=beta_kl,
            include_cmi=include_cmi,
        )

        # These graphs are separated by freezing and detaching.
        loss.backward()
        loss_opponent.backward()

        self.optimizer_vae.step()
        self.optimizer_main.step()
        self.optimizer_opponent.step()

        with torch.no_grad():
            mu_d, logvar_d = self.model.encode(x)

            z_d = self.model.reparameterize(
                mu_d,
                logvar_d,
            )


        loss_d = self.cfg.lambda_cls * discriminator_loss(
            self.model.discriminator,
            z_d["y"],
            z_d["r"],
            z_d["s"],
        )

        self.optimizer_discriminator.zero_grad(set_to_none=True)

        loss_d.backward()

        self.optimizer_discriminator.step()

        metrics["discriminator_loss"] = loss_d.detach().item()

        return metrics

    def _run_epoch(
        self,
        loader,
        training,
        beta_kl,
        include_cmi,
    ):
        self.model.train(training)

        totals = {}
        num_batches = 0

        y_labels = []
        s_labels = []
        y_probabilities = []
        s_probabilities = []
        opponent_y_probabilities = []
        opponent_s_probabilities = []

        with torch.set_grad_enabled(training):
            for batch in loader:
                x, y, s = self._prepare_batch(batch)

                if training:
                    metrics = self._train_batch(
                        x,
                        y,
                        s,
                        beta_kl,
                        include_cmi,
                    )
                else:
                    _, _, metrics, pred_y, pred_s, pred_y_op, pred_s_op = self._forward_losses(
                        x,
                        y,
                        s,
                        sample=False,
                        beta_kl=beta_kl,
                        include_cmi=include_cmi,
                    )

                    # Collect labels and probabilities across all batches.
                    y_labels.append(y.detach().cpu().numpy())
                    s_labels.append(s.detach().cpu().numpy())

                    y_probabilities.append(
                        probabilities(
                            pred_y,
                            self.cfg.num_y_classes,
                        ).detach().cpu().numpy()
                    )

                    s_probabilities.append(
                        probabilities(
                            pred_s,
                            self.cfg.num_s_classes,
                        ).detach().cpu().numpy()
                    )
                    
                    opponent_y_probabilities.append(
                        probabilities(
                            pred_y_op,
                            self.cfg.num_y_classes,
                        ).detach().cpu().numpy()
                    )

                    opponent_s_probabilities.append(
                        probabilities(
                            pred_s_op,
                            self.cfg.num_s_classes,
                        ).detach().cpu().numpy()
                    )

                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value

                num_batches += 1

        if num_batches == 0:
            raise ValueError("The data loader contains no batches.")

        results = {
            name: value / num_batches
            for name, value in totals.items()
        }

        # Compute AUC once using the complete evaluation dataset.
        if not training:
            results["auc_y"] = macro_ovr_auc(
                np.concatenate(y_labels, axis=0),
                np.concatenate(y_probabilities, axis=0),
                self.cfg.num_y_classes,
            )

            results["auc_s"] = macro_ovr_auc(
                np.concatenate(s_labels, axis=0),
                np.concatenate(s_probabilities, axis=0),
                self.cfg.num_s_classes,
            )
            
            results["auc_opponent_y"] = macro_ovr_auc(
                np.concatenate(y_labels, axis=0),
                np.concatenate(opponent_y_probabilities, axis=0),
                self.cfg.num_y_classes,
            )

            results["auc_opponent_s"] = macro_ovr_auc(
                np.concatenate(s_labels, axis=0),
                np.concatenate(opponent_s_probabilities, axis=0),
                self.cfg.num_s_classes,
            )          

        return results

    def fit(self, train_loader, val_loader):
        """Train and retain the weights with the lowest validation loss."""
        self.history = []
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None

        for epoch in range(1, self.cfg.max_epochs + 1):
            self.current_epoch = epoch

            beta_kl = get_beta(
                epoch,
                self.cfg.beta,
                self.cfg.beta_warmup_epochs,
            )

            include_cmi = epoch >= self.cfg.start_cmi_epoch

            train_metrics = self._run_epoch(
                train_loader,
                training=True,
                beta_kl=beta_kl,
                include_cmi=include_cmi,
            )

            val_metrics = self._run_epoch(
                val_loader,
                training=False,
                beta_kl=beta_kl,
                include_cmi=include_cmi,
            )

            row = {
                "epoch": epoch,
                "beta": beta_kl,
                "include_cmi": int(include_cmi),
            }

            row.update({
                f"train_{name}": value
                for name, value in train_metrics.items()
            })

            row.update({
                f"val_{name}": value
                for name, value in val_metrics.items()
            })

            self.history.append(row)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch

                self.best_state_dict = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }

            if self.cfg.verbose:
                print(
                    f"Epoch {epoch:03d}/{self.cfg.max_epochs} "
                    f"beta={beta_kl:.4g} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc_y={val_metrics['acc_y']:.4f} "
                    f"val_acc_s={val_metrics['acc_s']:.4f}"
                )

        return self.history

    def evaluate(
        self,
        loader,
        epoch: Optional[int] = None,
    ):
        """Evaluate with mu; use final coefficients when epoch is omitted."""
        if epoch is None:
            beta_kl = self.cfg.beta
            include_cmi = True
        else:
            beta_kl = get_beta(
                epoch,
                self.cfg.beta,
                self.cfg.beta_warmup_epochs,
            )

            include_cmi = epoch >= self.cfg.start_cmi_epoch

        return self._run_epoch(
            loader,
            training=False,
            beta_kl=beta_kl,
            include_cmi=include_cmi,
        )

    def restore_best_model(self):
        """Restore model weights without restoring optimizer states."""
        if self.best_state_dict is None:
            raise RuntimeError(
                "No best model is available. Run fit() first."
            )

        self.model.load_state_dict(self.best_state_dict)

    @torch.no_grad()
    def encode_dataset(self, x, batch_size=None):
        """Return branch-wise posterior means in the original sample order."""
        if batch_size is None:
            batch_size = self.cfg.batch_size

        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
        )

        if x_tensor.size(0) == 0:
            raise ValueError(
                "The input dataset is empty."
            )

        loader = DataLoader(
            TensorDataset(x_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        was_training = self.model.training
        self.model.eval()

        outputs = {
            name: []
            for name in BRANCHES
        }

        try:
            for (x_batch,) in loader:
                mu, _ = self.model.encode(
                    x_batch.to(self.device)
                )

                for name in BRANCHES:
                    outputs[name].append(
                        mu[name].cpu().numpy()
                    )
        finally:
            self.model.train(was_training)

        return {
            name: np.concatenate(outputs[name], axis=0)
            for name in BRANCHES
        }