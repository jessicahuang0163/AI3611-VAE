import yaml
import os
import argparse

import utils.pytorch_util as ptu
from utils.launcher_util import setup_logger
from src import vae
from utils.base.dataloader import loading_data
from src.core import train


def experiment(exp_specs, device):
    train_loader, test_loader = loading_data(
        os.path.dirname(os.path.abspath(__file__)),
        exp_specs["img_size"],
        exp_specs["batch_size"],
    )

    model = vae.VAE(exp_specs).to(device)

    train(
        model=model,
        dataloader=[train_loader, test_loader],
        exp_specs=exp_specs,
        device=device,
    )


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)

    # Set the random seed manually for reproducibility.
    seed = exp_specs["seed"]
    ptu.set_seed(seed)

    setup_logger(log_dir=exp_specs["log_dir"], variant=exp_specs)
    experiment(exp_specs, device)
