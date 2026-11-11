from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


# Fixed concatenation order: X, Y, S, R
BRANCHES = ("x", "y", "s", "r")


@dataclass
class ModelConfig:
    """Model architecture configuration, excluding training settings."""

    dim_x: int = 16
    dim_y: int = 16
    dim_s: int = 16
    dim_r: int = 16

    hidden_dims: tuple = (512, 128)
    classifier_hidden_dims: tuple = (64, 64)

    dropout: float = 0.1
    target_classes: int = 2
    sensitive_classes: int = 10


    @property
    def latent_dims(self):
        return {
            "x": self.dim_x,
            "y": self.dim_y,
            "s": self.dim_s,
            "r": self.dim_r,
        }

    @property
    def latent_dim(self):
        return sum(self.latent_dims.values())


def concatenate(values):
    """
    Concatenate latent branches in the fixed X, Y, S, R order.

    Default shapes:
        [B,16] + [B,16] + [B,16] + [B,16]
        -> [B,64]
    """
    return torch.cat(
        [values[name] for name in BRANCHES],
        dim=-1,
    )


def mlp(
    input_dim,
    hidden_dims,
    dropout=0.1,
    layer_norm=True,
):
    """Build MLP hidden layers without a final output layer."""
    layers = []

    for width in hidden_dims:
        layers.append(nn.Linear(input_dim, width))

        if layer_norm:
            layers.append(nn.LayerNorm(width))

        layers.append(nn.GELU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        input_dim = width

    return nn.Sequential(*layers)


class Classifier(nn.Module):
    """
    Return classification logits.

    Binary classification:
        Output shape [B,1], used with BCEWithLogitsLoss.

    Multiclass classification:
        Output shape [B,K], used with CrossEntropyLoss.
    """

    def __init__(
        self,
        input_dim,
        classes,
        hidden_dims=(64, 64),
    ):
        super().__init__()

        output_dim = 1 if classes == 2 else classes

        # Auxiliary networks use neither Dropout nor BatchNorm.
        self.network = nn.Sequential(
            mlp(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=0.0,
                layer_norm=False,
            ),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, x):
        return self.network(x)


class CADVAE(nn.Module):
    """
    MLP CAD-VAE for SNP data encoded with two channels per SNP.

    Default architecture:
        encoder: D -> 512 -> 128
        latent:  X16 / Y16 / S16 / R16
        decoder: 64 -> 128 -> 512 -> D

    """

    def __init__(self, input_dim, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = ModelConfig()

        self.input_dim = input_dim
        self.latent_dims = cfg.latent_dims
        self.target_classes = cfg.target_classes


        # --------------------------------------------------
        # 1. Shared encoder: D -> 512 -> 128
        # --------------------------------------------------
        self.encoder = mlp(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            layer_norm=True,
        )

        encoder_output_dim = cfg.hidden_dims[-1]

        # --------------------------------------------------
        # 2. Separate mu and logvar heads for each branch
        # --------------------------------------------------
        self.mu_heads = nn.ModuleDict({
            name: nn.Linear(encoder_output_dim, dim)
            for name, dim in self.latent_dims.items()
        })

        self.logvar_heads = nn.ModuleDict({
            name: nn.Linear(encoder_output_dim, dim)
            for name, dim in self.latent_dims.items()
        })

        # --------------------------------------------------
        # 3. Decoder: 64 -> 128 -> 512 -> D
        # --------------------------------------------------
        decoder_hidden_dims = tuple(
            reversed(cfg.hidden_dims)
        )

        self.decoder = nn.Sequential(
            mlp(
                input_dim=cfg.latent_dim,
                hidden_dims=decoder_hidden_dims,
                dropout=cfg.dropout,
                layer_norm=True,
            ),
            nn.Linear(
                decoder_hidden_dims[-1],
                input_dim,
            ),
        )

        # Return logits for BCEWithLogitsLoss.
        # Do not apply sigmoid here.

        d = self.latent_dims
        h = cfg.classifier_hidden_dims

        # --------------------------------------------------
        # 4. Main classifiers
        # --------------------------------------------------

        # Predict Y from [z_y,z_r]: 32 input dimensions by default.
        self.cls_y = Classifier(
            input_dim=d["y"] + d["r"],
            classes=cfg.target_classes,
            hidden_dims=h,
        )

        # Predict Island from [z_s,z_r]: 32 input dimensions by default.
        self.cls_s = Classifier(
            input_dim=d["s"] + d["r"],
            classes=cfg.sensitive_classes,
            hidden_dims=h,
        )

        # --------------------------------------------------
        # 5. Opponent classifiers
        # --------------------------------------------------

        # Predict Y from z_s.
        self.opp_y = Classifier(
            input_dim=d["s"],
            classes=cfg.target_classes,
            hidden_dims=h,
        )

        # Predict Island from z_y.
        self.opp_s = Classifier(
            input_dim=d["y"],
            classes=cfg.sensitive_classes,
            hidden_dims=h,
        )

        # --------------------------------------------------
        # 6. TC discriminator
        # --------------------------------------------------

        # Input: [z_y,z_r,z_s], with 48 dimensions by default.
        
        self.discriminator = Classifier(
            input_dim=d["y"] + d["r"] + d["s"],
            classes=2,
            hidden_dims=h,
        )

    def encode(self, x):
        """
        Encode input x with shape [B,D].

        Return two dictionaries, mu and logvar.

        Default mu shapes:
            mu["x"]: [B,16]
            mu["y"]: [B,16]
            mu["s"]: [B,16]
            mu["r"]: [B,16]

        Each logvar tensor has the same shape as its corresponding mu.
        """
        hidden = self.encoder(x)

        mu = {
            name: self.mu_heads[name](hidden)
            for name in BRANCHES
        }

        logvar = {
            name: self.logvar_heads[name](hidden)
            for name in BRANCHES
        }

        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Apply reparameterization to each branch:
            std = exp(0.5 * logvar)
            eps ~ N(0,I)
            z = mu + std * eps
        """
        z = {}

        for name in BRANCHES:
            std = torch.exp(0.5 * logvar[name])
            eps = torch.randn_like(mu[name])

            z[name] = mu[name] + std * eps

        return z

    def decode(self, z):
        """
        Accept either:
            a dictionary containing the four latent branches;
            a concatenated latent tensor of shape [B,64] by default.

        Return reconstruction logits with shape [B,D].
        """
        if isinstance(z, dict):
            z = concatenate(z)

        return self.decoder(z)

    def forward(self, x, sample=True):
        """
        sample=True:
            Use sampled latent variables for training.

        sample=False:
            Use mu for validation, testing, and feature export.

        Return:
            recon_logits, z, mu, logvar
        """
        mu, logvar = self.encode(x)

        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        recon_logits = self.decode(z)

        return recon_logits, z, mu, logvar

    # --------------------------------------------------
    # Parameter groups for separate optimizers
    # --------------------------------------------------

    def vae_parameters(self):
        """Yield encoder, Gaussian head, and decoder parameters."""
        modules = (
            self.encoder,
            self.mu_heads,
            self.logvar_heads,
            self.decoder,
        )

        for module in modules:
            yield from module.parameters()

    def main_parameters(self):
        """Yield parameters of the two main classifiers."""
        yield from self.cls_y.parameters()
        yield from self.cls_s.parameters()

    def opponent_parameters(self):
        """Yield parameters of the two opponent classifiers."""
        yield from self.opp_y.parameters()
        yield from self.opp_s.parameters()


@contextmanager
def frozen(*modules):
    """
    Temporarily freeze module parameters while retaining input gradients.

    Enter this context before the corresponding forward pass.
    Restore each parameter's original requires_grad state on exit.
    """
    states = [
        (parameter, parameter.requires_grad)
        for module in modules
        for parameter in module.parameters()
    ]

    try:
        for parameter, _ in states:
            parameter.requires_grad_(False)

        yield

    finally:
        for parameter, original_state in states:
            parameter.requires_grad_(original_state)