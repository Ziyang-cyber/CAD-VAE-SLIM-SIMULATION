import torch
import torch.nn.functional as F


BRANCHES = ("x", "y", "s", "r")

LOG_EPS = 1e-7


# --------------------------------------------------
# 1. Reconstruction and Gaussian KL
# --------------------------------------------------

def reconstruction_loss(logits, x):
    """
    Use the existing SNP reconstruction objective.

    Sum over input coordinates and average over samples.
    Both logits and x have shape [B, D].
    """
    return F.binary_cross_entropy_with_logits(
        logits,
        x,
        reduction="sum",
    ) / x.shape[0]


def kl_loss(mu, logvar):
    """Compute KL after concatenating all latent branches."""
    mu_all = torch.cat(
        [mu[name] for name in BRANCHES],
        dim=-1,
    )

    logvar_all = torch.cat(
        [logvar[name] for name in BRANCHES],
        dim=-1,
    )

    batch_size = mu_all.size(0)

    return -0.5 * (
        1
        + logvar_all
        - mu_all ** 2
        - torch.clamp(
            logvar_all.exp(),
            min=1e-6,
            max=1000.0,
        )
    ).sum() / batch_size

# --------------------------------------------------
# 2. Classification and predictive entropy
# --------------------------------------------------

def classification_loss(logits, labels, num_classes=2):
    """
    Use BCE for binary classification and CE for multiclass classification.

    Binary:
        logits: [B, 1]
        labels: [B] or [B, 1]

    Multiclass:
        logits: [B, K]
        labels: [B], encoded as 0, ..., K-1
    """
    if num_classes == 2:
        labels = labels.to(
            dtype=logits.dtype
        ).reshape(-1, 1)

        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
        )

    return F.cross_entropy(
        logits,
        labels.long().reshape(-1),
    )


def probabilities(logits, num_classes=2):
    """Return a complete class probability vector."""
    if num_classes == 2:
        p = torch.sigmoid(logits)

        # Columns correspond to class 0 and class 1.
        return torch.cat([1 - p, p], dim=-1)

    return torch.softmax(logits, dim=-1)


def entropy(probability):
    """
    Compute categorical entropy in nats.

    Sum over the final class dimension.
    """
    return -(
        probability
        * torch.log(
            torch.clamp(probability, min=LOG_EPS)
        )
    ).sum(dim=-1)


# --------------------------------------------------
# 3. Cross-prediction entropy
# --------------------------------------------------

def direct_CMI_loss(
    cls_y_zsr,
    cls_s_zyr,
    z_y,
    z_s,
    num_y_classes=2,
    num_s_classes=3,
):
    """

    The Y opponent receives z_s only.
    The S opponent receives z_y only.


    The caller freezes opponent parameters before this forward pass.

    Return:
        Negative Y prediction entropy from z_s.
        Negative S prediction entropy from z_y.
    """
    p_y = probabilities(
        cls_y_zsr(z_s),
        num_y_classes,
    )

    p_s = probabilities(
        cls_s_zyr(z_y),
        num_s_classes,
    )

    return (
        -entropy(p_y).mean(),
        -entropy(p_s).mean(),
    )


# --------------------------------------------------
# 4. Learning relevance of the shared branch
# --------------------------------------------------

def learning_relevance(
    classifier,
    private,
    shared,
    labels,
    num_classes,
    alpha,
):
    """
    Generalize the original paired/grouped entropy to K classes.

    Axis 0 indexes private samples.
    Axis 1 indexes shared samples.

    Group contributions are weighted by their sample counts.

    """
    n = private.shape[0]
    labels = labels.reshape(-1)

    assert shared.shape[0] == n == labels.shape[0]

    # Construct all N-by-N private/shared combinations.
    private_pairs = private.unsqueeze(1).expand(
        -1, n, -1
    )

    shared_pairs = shared.unsqueeze(0).expand(
        n, -1, -1
    )

    pairs = torch.cat(
        [private_pairs, shared_pairs],
        dim=-1,
    )

    pairs = pairs.reshape(n * n, -1)

    p = probabilities(
        classifier(pairs),
        num_classes,
    )

    p = p.reshape(n, n, num_classes)

    # Average over private samples for each fixed shared sample.
    p_shared = p.mean(dim=0)

    h_shared = entropy(p_shared).mean()

    h_grouped = shared.new_zeros(())

    for class_id in range(num_classes):
        mask = labels == class_id

        # An absent class contributes zero.
        if not mask.any():
            continue

        p_group = p[mask, :, :][:, mask, :].mean(dim=0)

        h_grouped = (
            h_grouped
            + entropy(p_group).sum() / n
        )

    return alpha * h_shared - h_grouped


def learning_relevant_zr(
    cls_y_zyr,
    cls_s_zsr,
    z_y,
    z_r,
    z_s,
    y_batch,
    s_batch,
    alpha=0.5,
    beta=0.5,
    num_y_classes=2,
    num_s_classes=3,
):
    """
    Return the Y and S learning relevance terms.
    """
    relevance_y = learning_relevance(
        classifier=cls_y_zyr,
        private=z_y,
        shared=z_r,
        labels=y_batch,
        num_classes=num_y_classes,
        alpha=alpha,
    )

    relevance_s = learning_relevance(
        classifier=cls_s_zsr,
        private=z_s,
        shared=z_r,
        labels=s_batch,
        num_classes=num_s_classes,
        alpha=beta,
    )

    return relevance_y, relevance_s


# --------------------------------------------------
# 5. Block permutation and TC
# --------------------------------------------------

def permute_zs(zs):
    """
    Independently permute sample rows within each latent block.
    """
    batch_size = zs[0].shape[0]
    permuted = []

    for block in zs:
        indices = torch.randperm(
            batch_size,
            device=block.device,
        )

        permuted.append(block[indices])

    return torch.cat(permuted, dim=-1)


def tc_encoder_loss(discriminator, z_y, z_r, z_s):
    """

    The caller freezes discriminator parameters before this forward.
    Latent inputs remain connected to the encoder.
    """
    joint = torch.cat(
        [z_y, z_r, z_s],
        dim=-1,
    )

    p = torch.sigmoid(discriminator(joint))

    denominator = torch.clamp(
        1 - p,
        min=LOG_EPS,
    )

    ratio = torch.clamp(
        p / denominator,
        min=LOG_EPS,
    )

    return torch.log(ratio).mean()


def discriminator_loss(discriminator, z_y, z_r, z_s):
    """
    Compute the joint/permuted discrimination loss, unweighted.

    The trainer multiplies this term by lambda_cls.
    """
    joint = torch.cat(
        [z_y, z_r, z_s],
        dim=-1,
    )

    permuted = permute_zs([z_y, z_r, z_s])

    p_joint = torch.sigmoid(discriminator(joint))
    p_permuted = torch.sigmoid(discriminator(permuted))

    return (
        -torch.log(p_joint + LOG_EPS).mean()
        -torch.log(1 - p_permuted + LOG_EPS).mean()
    )


# --------------------------------------------------
# 6. Encoder and main-classifier objective
# --------------------------------------------------

def encoder_objective(
    loss_recon,
    loss_kl,
    loss_y,
    loss_s,
    cmi_y,
    cmi_s,
    relevance_y,
    relevance_s,
    loss_tc,
    *,
    beta_kl,
    lambda_cls,
    lambda_tc,
    lambda_fair_yzs,
    lambda_fair_szy,
    lambda_fair_yzy,
    lambda_fair_szs,
    include_cmi,
):
    """
    objective loss
    Opponent and discriminator training use separate objectives.

    """
    total = loss_recon + beta_kl * loss_kl

    total = total + lambda_cls * (
        loss_y + loss_s
    )

    total = total + lambda_tc * loss_tc

    total = total - lambda_fair_yzy * relevance_y
    total = total - lambda_fair_szs * relevance_s

    if include_cmi:
        total = total + lambda_fair_yzs * cmi_y
        total = total + lambda_fair_szy * cmi_s

    return total