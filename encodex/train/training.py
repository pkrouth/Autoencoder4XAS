from ..utils.imports import *


def train(
    num_epochs,
    dl,
    model,
    criterion,
    opt,
    save=True,
    filename=None,
    enable_neptune=False,
    enable_tensorboard=True,
):
    """
    :params :
    : num_epochs: obvious
    : dl: Dataloaders, dictionary which contain train and validation keys and dataloader as value.
    """
    liveloss = PlotLosses()
    if enable_tensorboard:
        writer = SummaryWriter()
    model = model.cuda()
    for epoch in range(num_epochs):
        logs = {}
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for batch_idx, (x_data, y_data) in enumerate(dl[phase]):
                # x_data = x_data.view(x_data.size(0),-1)
                x_data = x_data.float()
                x_data = Variable(x_data).cuda()
                # ===================forward=====================
                output = model(x_data)
                loss = criterion(output, x_data)
                # ===================backward====================
                if phase == "train":
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # ===================log========================
                # print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
                running_loss += loss.detach() * x_data.size(0)
            epoch_loss = running_loss / len(dl[phase].dataset)
            prefix = ""
            if phase == "validation":
                prefix = "val_"
                if enable_tensorboard:
                    writer.add_scalar("MSELoss/valid", epoch_loss.item(), epoch)
                if enable_neptune:
                    neptune.send_metric("valid_epoch_loss", epoch, epoch_loss.item())
            if phase == "train":
                if enable_tensorboard:
                    writer.add_scalar("MSELoss/train", epoch_loss.item(), epoch)
                if enable_neptune:
                    neptune.send_metric("train_epoch_loss", epoch, epoch_loss.item())
            logs[prefix + "loss"] = epoch_loss.item()
        liveloss.update(logs)
        liveloss.draw()
    if save:
        if filename:
            torch.save(model.state_dict(), str(filename))
        else:
            torch.save(model.state_dict(), "./conv_autoencoder_kld.pth")
