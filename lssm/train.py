import torch
import wandb
import einops
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from .memory import ReplayBuffer
from .utils import compute_consistency, bottle_mvn
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)

logger = logging.getLogger(__name__)


def train_autoencoder(
    config: DictConfig,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    """
        Stage 1: Train encoder and decoder with reconstruction loss.
    """

    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    decoder = Decoder(
        y_dim=train_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    wandb.watch([encoder, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    for update in tqdm(range(config.num_updates)):
        # train
        encoder.train()
        decoder.train()

        y, _, _, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        y = torch.as_tensor(y, device=device)
        y_flatten = einops.rearrange(y, "b l y -> (b l) y")

        a = encoder(y_flatten)
        y_recon = decoder(a)
        ae_loss = nn.MSELoss()(y_recon, y_flatten)

        optimizer.zero_grad()
        ae_loss.backward()
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/ae loss": ae_loss.item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()

                y, _, _, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                y = torch.as_tensor(y, device=device)
                y_flatten = einops.rearrange(y, "b l y -> (b l) y")

                a = encoder(y_flatten)
                y_recon = decoder(a)
                ae_loss = nn.MSELoss()(y_recon, y_flatten)

                wandb.log({
                    "test/ae loss": ae_loss.item(),
                    "global_step": update,
                })

    return encoder, decoder


def train_dynamics(
    config: DictConfig,
    encoder: nn.Module,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    """
        Stage 2: Train dynamics model with frozen encoder.
        Losses are computed in a-space (latent observation space).
    """

    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    dynamics_model = Dynamics(
        x_dim=config.x_dim,
        u_dim=train_buffer.u_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        min_var=config.min_var,
        max_var=config.max_var,
        locally_linear=config.locally_linear,
    ).to(device)

    wandb.watch([dynamics_model], log="all", log_freq=10)

    all_params = list(dynamics_model.parameters())

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    for update in tqdm(range(config.num_updates)):
        # train
        dynamics_model.train()

        y, u, _, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        priors, posteriors = dynamics_model(a=a, u=u)   # x0:T-1

        consistencies = compute_consistency(
            prior=bottle_mvn(priors),
            posterior=bottle_mvn(posteriors),
            free_nats=config.kl_free_nats
        )
        mean_consistency = consistencies[0]
        kl_consistency = consistencies[1]

        filter_a = dynamics_model.get_a(bottle_mvn(posteriors).loc)
        a_flatten = einops.rearrange(a, "l b a -> (l b) a")
        a_filter_loss = nn.MSELoss()(filter_a, a_flatten)

        a_pred_loss = 0.0
        for k in range(1, config.prediction_k+1):
            pred_dist = bottle_mvn(posteriors[0:config.chunk_length-k])
            for t in range(k):
                pred_dist = dynamics_model.prior(
                    dist=pred_dist,
                    u=einops.rearrange(u[t:config.chunk_length-k+t], "l b u -> (l b) u"),
                )
            pred_a = dynamics_model.get_a(pred_dist.loc)
            true_a = einops.rearrange(a[k:config.chunk_length], "l b a -> (l b) a")
            a_pred_loss += nn.MSELoss()(pred_a, true_a) * (config.chunk_length - k) / config.chunk_length

        a_pred_loss /= config.prediction_k

        total_loss = (
            a_pred_loss +
            config.filtering_weight * a_filter_loss +
            config.mean_consistency_weight * mean_consistency +
            config.kl_consistency_weight * kl_consistency
        )

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/a prediction loss": a_pred_loss.item(),
            "train/a filter loss": a_filter_loss.item(),
            "train/total loss": total_loss.item(),
            "train/mean consistency": mean_consistency.item(),
            "train/kl consistency": kl_consistency.item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                dynamics_model.eval()

                y, u, _, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                priors, posteriors = dynamics_model(a=a, u=u)   # x0:T-1

                consistencies = compute_consistency(
                    prior=bottle_mvn(priors),
                    posterior=bottle_mvn(posteriors),
                    free_nats=config.kl_free_nats
                )
                mean_consistency = consistencies[0]
                kl_consistency = consistencies[1]

                filter_a = dynamics_model.get_a(bottle_mvn(posteriors).loc)
                a_flatten = einops.rearrange(a, "l b a -> (l b) a")
                a_filter_loss = nn.MSELoss()(filter_a, a_flatten)

                a_pred_loss = 0.0
                for k in range(1, config.prediction_k+1):
                    pred_dist = bottle_mvn(posteriors[0:config.chunk_length-k])
                    for t in range(k):
                        pred_dist = dynamics_model.prior(
                            dist=pred_dist,
                            u=einops.rearrange(u[t:config.chunk_length-k+t], "l b u -> (l b) u"),
                        )
                    pred_a = dynamics_model.get_a(pred_dist.loc)
                    true_a = einops.rearrange(a[k:config.chunk_length], "l b a -> (l b) a")
                    a_pred_loss += nn.MSELoss()(pred_a, true_a) * (config.chunk_length - k) / config.chunk_length

                a_pred_loss /= config.prediction_k

                total_loss = (
                    a_pred_loss +
                    config.filtering_weight * a_filter_loss +
                    config.mean_consistency_weight * mean_consistency +
                    config.kl_consistency_weight * kl_consistency
                )

                wandb.log({
                    "test/a prediction loss": a_pred_loss.item(),
                    "test/a filter loss": a_filter_loss.item(),
                    "test/total loss": total_loss.item(),
                    "test/mean consistency": mean_consistency.item(),
                    "test/kl consistency": kl_consistency.item(),
                    "global_step": update,
                })

    return dynamics_model


def _evaluate_dynamics(
    config: DictConfig,
    encoder: nn.Module,
    dynamics_model: Dynamics,
    buffer: ReplayBuffer,
    prefix: str,
    device: str,
):
    """
        Evaluate dynamics model on a buffer. Computes the same metrics as the
        prediction-based training loop: a-filter loss, a-prediction loss, consistency.
    """
    with torch.no_grad():
        y, u, _, _ = buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        priors, posteriors = dynamics_model(a=a, u=u)

        consistencies = compute_consistency(
            prior=bottle_mvn(priors),
            posterior=bottle_mvn(posteriors),
            free_nats=config.kl_free_nats
        )
        mean_consistency = consistencies[0]
        kl_consistency = consistencies[1]

        filter_a = dynamics_model.get_a(bottle_mvn(posteriors).loc)
        a_flatten = einops.rearrange(a, "l b a -> (l b) a")
        a_filter_loss = nn.MSELoss()(filter_a, a_flatten)

        a_pred_loss = 0.0
        for k in range(1, config.prediction_k + 1):
            pred_dist = bottle_mvn(posteriors[0:config.chunk_length - k])
            for t in range(k):
                pred_dist = dynamics_model.prior(
                    dist=pred_dist,
                    u=einops.rearrange(u[t:config.chunk_length - k + t], "l b u -> (l b) u"),
                )
            pred_a = dynamics_model.get_a(pred_dist.loc)
            true_a = einops.rearrange(a[k:config.chunk_length], "l b a -> (l b) a")
            a_pred_loss += nn.MSELoss()(pred_a, true_a) * (config.chunk_length - k) / config.chunk_length

        a_pred_loss /= config.prediction_k

        total_loss = (
            a_pred_loss +
            config.filtering_weight * a_filter_loss +
            config.mean_consistency_weight * mean_consistency +
            config.kl_consistency_weight * kl_consistency
        )

        metrics = {
            f"{prefix}/a prediction loss": a_pred_loss.item(),
            f"{prefix}/a filter loss": a_filter_loss.item(),
            f"{prefix}/total loss": total_loss.item(),
            f"{prefix}/mean consistency": mean_consistency.item(),
            f"{prefix}/kl consistency": kl_consistency.item(),
        }

    return metrics


def train_dynamics_sid(
    config: DictConfig,
    encoder: nn.Module,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    """
        Stage 2 (SID variant): Identify dynamics via subspace identification.
        Uses PyPSID's IPSID with n1=0 (no preferential/behavioral signal).
        Sets identified A, B, C, Nx, Na as fixed buffers on the Dynamics model.
    """
    from PSID import IPSID

    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # encode all observations to a-space
    with torch.no_grad():
        all_y = torch.as_tensor(train_buffer.ys[:len(train_buffer)], device=device)
        all_a = encoder(all_y).cpu().numpy()
    all_u = train_buffer.us[:len(train_buffer)]
    all_done = train_buffer.done[:len(train_buffer)].flatten()

    # split into per-episode lists (IPSID natively supports list of segments)
    episode_ends = np.where(all_done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])

    a_episodes = []
    u_episodes = []
    for start, end in zip(episode_starts, episode_ends):
        a_episodes.append(all_a[start:end + 1])  # (T_i, a_dim)
        u_episodes.append(all_u[start:end + 1])  # (T_i, u_dim)

    total_samples = sum(len(ep) for ep in a_episodes)
    logger.info(f"Running SID on {len(a_episodes)} episodes ({total_samples} total samples) with horizon i={config.sid_horizon}")

    # run IPSID: vanilla SID with inputs (n1=0, no behavioral signal)
    # pass lists of per-episode arrays — IPSID handles boundaries internally
    id_sys = IPSID(
        Y=a_episodes,
        Z=None,
        U=u_episodes,
        nx=config.x_dim,
        n1=0,
        i=config.sid_horizon,
        remove_mean_Y=False,
        remove_mean_U=False,
    )

    # create Dynamics model with full noise covariances (via Cholesky)
    dynamics_model = Dynamics(
        x_dim=config.x_dim,
        u_dim=train_buffer.u_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        min_var=config.min_var,
        max_var=config.max_var,
        locally_linear=False,
        diagonal_noise=False,
    ).to(device)

    with torch.no_grad():
        dynamics_model.A.copy_(torch.as_tensor(id_sys.A, dtype=torch.float32))
        dynamics_model.B.copy_(torch.as_tensor(id_sys.B, dtype=torch.float32))
        dynamics_model.C.copy_(torch.as_tensor(id_sys.C, dtype=torch.float32))

        # set full noise covariances via Cholesky factors: Nx = Lx @ Lx^T
        Q = id_sys.Q.astype(np.float64)
        Q = (Q + Q.T) / 2
        Lx = np.linalg.cholesky(Q)
        dynamics_model.Lx.copy_(torch.as_tensor(Lx, dtype=torch.float32))

        R = id_sys.R.astype(np.float64)
        R = (R + R.T) / 2
        La = np.linalg.cholesky(R)
        dynamics_model.La.copy_(torch.as_tensor(La, dtype=torch.float32))

    # freeze all dynamics parameters
    for p in dynamics_model.parameters():
        p.requires_grad = False
    dynamics_model.eval()

    logger.info("SID identification complete.")
    logger.info(f"  A:\n{id_sys.A}")
    logger.info(f"  B:\n{id_sys.B}")
    logger.info(f"  C:\n{id_sys.C}")
    logger.info(f"  Q diag: {np.diag(id_sys.Q)}")
    logger.info(f"  R diag: {np.diag(id_sys.R)}")

    # evaluate on train and test data
    train_metrics = _evaluate_dynamics(config, encoder, dynamics_model, train_buffer, "train", device)
    test_metrics = _evaluate_dynamics(config, encoder, dynamics_model, test_buffer, "test", device)
    wandb.log({**train_metrics, **test_metrics, "global_step": 0})

    return dynamics_model


def train_cost(
    config: DictConfig,
    encoder: nn.Module,
    dynamics_model: Dynamics,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    """
        Stage 3: Train cost model with frozen encoder and dynamics.
    """

    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    cost_model = CostModel(
        x_dim=dynamics_model.x_dim,
        u_dim=dynamics_model.u_dim,
    ).to(device)

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False

    for p in dynamics_model.parameters():
        p.requires_grad = False

    encoder.eval()
    dynamics_model.eval()

    wandb.watch([cost_model], log="all", log_freq=10)

    all_params = list(cost_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):
        # train
        cost_model.train()

        y, u, c, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        c = torch.as_tensor(c, device=device)
        c = einops.rearrange(c, "b l 1 -> l b 1")

        _, posteriors = dynamics_model(a=a, u=u)  # x0:T-1
        # compute cost loss
        cost_loss = nn.MSELoss()(
            cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
            einops.rearrange(c, "l b 1 -> (l b) 1")
        )
        optimizer.zero_grad()
        cost_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/cost loss": cost_loss.item(),
            "global_step": update,
        })

        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                cost_model.eval()

                y, u, c, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")
                c = torch.as_tensor(c, device=device)
                c = einops.rearrange(c, "b l 1 -> l b 1")

                _, posteriors = dynamics_model(a=a, u=u)  # x0:T-1
                # compute cost loss
                cost_loss = nn.MSELoss()(
                    cost_model(x=bottle_mvn(posteriors).loc, u=einops.rearrange(u, "l b u -> (l b) u")),
                    einops.rearrange(c, "l b 1 -> (l b) 1")
                )

                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })

    return cost_model
