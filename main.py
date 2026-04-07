import os
import envs
import json
import wandb
import torch
import minari
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from minari import MinariDataset
from lssm.memory import ReplayBuffer
from envs.utils import collect_data
from lssm.train import train_autoencoder, train_dynamics
from lssm.evaluation import evaluate
from lssm.utils import jsonify
from lssm.models import IdentityEncoder, IdentityDecoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSSM")
    parser.add_argument("--config", type=str, help="path to the config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.run_id = datetime.now().strftime("%Y%m%d_%H%M")

    wandb.init(
        project="Manifolds control",
        name=config.run_name,
        notes=config.notes,
        config=OmegaConf.to_container(config, resolve=True)
    )

    # prepare logging
    save_dir = Path(config.log_dir) / config.run_id
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config, save_dir / "config.yaml")
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    logger = logging.getLogger(__name__)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # create env and collect data
    env = envs.make(config=config.env)
    logger.info("collecting data ...")
    collect_data(
        env=env,
        data_dir=save_dir / "data",
        num_episodes=config.data.num_episodes,
        action_repeat=config.data.action_repeat,
        dist=config.data.get("dist", "env"),
        u_std=config.data.get("u_std", 0.3),
        u_max=config.data.get("u_max", 0.3),
    )

    # create replay buffers
    dataset = MinariDataset(data=save_dir / "data")
    test_size = int(len(dataset) * config.data.test_ratio)
    train_size = len(dataset) - test_size
    train_data, test_data = minari.split_dataset(dataset=dataset, sizes=[train_size, test_size])
    train_buffer = ReplayBuffer.load_from_minari(dataset=train_data)
    test_buffer = ReplayBuffer.load_from_minari(dataset=test_data)

    use_autoencoder = config.train.use_autoencoder

    if use_autoencoder:
        # Stage 1: Train autoencoder
        logging.info("training autoencoder ...")
        encoder, decoder = train_autoencoder(
            config=config.train.autoencoder,
            train_buffer=train_buffer,
            test_buffer=test_buffer,
        )
        torch.save(encoder.state_dict(), save_dir / "encoder.pth")
        torch.save(decoder.state_dict(), save_dir / "decoder.pth")
    else:
        # Identity: no learned encoder/decoder
        logging.info("using identity encoder/decoder ...")
        device = "cuda" if (torch.cuda.is_available() and not config.train.dynamics.disable_gpu) else "cpu"
        encoder = IdentityEncoder().to(device)
        decoder = IdentityDecoder().to(device)

    # Stage 2: Train dynamics with frozen encoder/decoder
    logging.info("training dynamics ...")
    dynamics_model = train_dynamics(
        config=config.train.dynamics,
        encoder=encoder,
        decoder=decoder,
        train_buffer=train_buffer,
        test_buffer=test_buffer,
    )
    torch.save(dynamics_model.state_dict(), save_dir / "dynamics_model.pth")

    # Stage 3: Evaluate (includes per-target cost training)
    eval_results = evaluate(
        eval_config=config.evaluation,
        cost_train_config=config.train.cost,
        env=env,
        dynamics_model=dynamics_model,
        encoder=encoder,
        train_buffer=train_buffer,
        test_buffer=test_buffer
    )

    eval_results = [jsonify(er) for er in eval_results]
    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    wandb.save(save_dir / "eval_results.json")

    wandb.finish()
