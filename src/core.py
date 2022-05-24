import torch
import torchvision

def validate(model, dataloader, device):
    loss = []
    model.eval()
    for i, batches in enumerate(dataloader):
        imgs = batches[0].to(device)

        mu, log_var, out = model(imgs)

        loss_dict = model.loss(imgs, mu, log_var, out)
        loss.append(loss_dict["Reconstruction_Loss"])

    return sum(loss) / len(loss)


def train(model, dataloader, opt, logger, exp_specs, device):
    train_loader, test_loader = dataloader
    best_val_loss = None
    for current_epoch in range(exp_specs["epoches"]):
        for i, batches in enumerate(train_loader):
            model.train()
            opt.zero_grad()

            imgs = batches[0].to(device)
            # noise = 0.03 * torch.rand(imgs.shape).to(imgs.device)
            # imgs += noise

            mu, log_var, out = model(imgs)

            loss_dict = model.loss(imgs, mu, log_var, out)
            loss_dict["loss"].backward()
            opt.step()

        logger.record_tabular("loss", loss_dict["loss"].cpu().detach().numpy())
        logger.record_tabular(
            "train_reconstucion_loss",
            loss_dict["Reconstruction_Loss"].cpu().detach().numpy(),
        )
        logger.record_tabular("KLD", loss_dict["KLD"].cpu().detach().numpy())

        model.eval()

        num = 20
        if exp_specs["latent_dim"] == 1:
            latents = (torch.arange(num ** 2) - num ** 2 / 2).reshape(-1, 1) / 40.0
        elif exp_specs["latent_dim"] == 2:
            indexes = (torch.arange(num) - num / 2) / 5.0
            x, y = torch.meshgrid(indexes, indexes, indexing="ij")
            latents = torch.stack([x.flatten(), y.flatten()]).T
        else:
            latents = torch.randn(100, exp_specs["latent_dim"])
        latents = latents.to(device)

        val_loss = validate(model, test_loader, device)
        logger.record_tabular("val_reconstucion_loss", val_loss.cpu().detach().numpy())
        logger.record_tabular("Epoch", current_epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            logger.save_torch_model(model, "model.pt")
            best_val_loss = val_loss
    
    generate_imgs = model.generate(latents)
    grid = torchvision.utils.make_grid(generate_imgs, nrow=num)
    torchvision.utils.save_image(grid,'generate_image.jpg')
