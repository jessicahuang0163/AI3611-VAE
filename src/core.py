import torch
import torchvision
import torch.nn.functional as F

from utils import logger
from utils.running_stat import RunningStats


def loss_fn(exp_specs, input, mu, log_sigma, recons):
    """ 
    Reconstruction loss + KL divergence
    """
    recon_loss = F.mse_loss(recons, input)
    kld_loss = 0.5 * (
        (torch.pow(mu, 2) + log_sigma.exp() - log_sigma).sum(dim=-1)
        - exp_specs["latent_dim"]
    ).mean(dim=0)

    loss = recon_loss + exp_specs["alpha"] * kld_loss
    return {
        "loss": loss,
        "recon_loss": recon_loss.clone().detach(),
        "kld_loss": -kld_loss.clone().detach(),
    }


def train(model, dataloader, exp_specs, device):
    train_loader, test_loader = dataloader
    opt = torch.optim.Adam(model.parameters(), lr=exp_specs["lr"])
    best_val_loss = None
    eval_metrics = {}
    for current_epoch in range(exp_specs["epoches"]):
        for i, batches in enumerate(train_loader):
            model.train()
            opt.zero_grad()

            imgs = batches[0].to(device)
            mu, log_sigma, recon = model(imgs)
            loss_dict = loss_fn(exp_specs, imgs, mu, log_sigma, recon)
            loss_dict["loss"].backward()
            opt.step()

            eval_metrics[0], eval_metrics[1], eval_metrics[2] = (
                RunningStats(),
                RunningStats(),
                RunningStats(),
            )
            eval_metrics[0].push(loss_dict["recon_loss"].cpu().detach().numpy())
            eval_metrics[1].push(loss_dict["kld_loss"].cpu().detach().numpy())
            eval_metrics[2].push(loss_dict["loss"].cpu().detach().numpy())

        logger.record_tabular("train_recon_loss", eval_metrics[0].mean())
        logger.record_tabular("kld_loss", eval_metrics[1].mean())
        logger.record_tabular("loss", eval_metrics[2].mean())

        model.eval()

        num = 20
        if exp_specs["latent_dim"] == 1:
            latents = (torch.arange(num ** 2) - num ** 2 / 2).reshape(-1, 1) / 40.0
        elif exp_specs["latent_dim"] == 2:
            indexes = (torch.arange(num) - num / 2) / 5.0
            x, y = torch.meshgrid(indexes, indexes, indexing="ij")
            latents = torch.stack([x.flatten(), y.flatten()]).T

        latents = latents.to(device)

        val_loss = val(exp_specs, model, test_loader, device)
        logger.record_tabular("val_recon_loss", val_loss)
        logger.record_tabular("Epoch", current_epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            logger.save_torch_model(model, "model.pt")
            best_val_loss = val_loss

    generate_imgs = model.generate(latents)
    grid = torchvision.utils.make_grid(generate_imgs, nrow=num)
    torchvision.utils.save_image(grid, "generate_image.png")


def val(exp_specs, model, dataloader, device):
    model.eval()
    for i, batches in enumerate(dataloader):
        imgs = batches[0].to(device)
        mu, log_sigma, recon = model(imgs)
        loss_dict = loss_fn(exp_specs, imgs, mu, log_sigma, recon)
        loss = RunningStats()
        loss.push(loss_dict["recon_loss"].cpu().detach().numpy())

    return loss.mean()
