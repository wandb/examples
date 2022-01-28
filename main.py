import logging
import pathlib
import pprint
import sys

sys.path.append("5029e7a6e431bc04135de662326ea682")

import omegaconf
import torch
import torchvision
import tqdm

import dataset
import hydra
import hydra.utils
import model
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_test_predictions(images, labels, outputs, predicted, test_table, log_counter):

    # obtain confidence scores for all classes
    scores = torch.nn.functional.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # adding ids based on the order of the images
    _id = 0
    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
        img_id = str(_id) + "_" + str(log_counter)
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)


def test_loop(test_loader, net, epoch, job_id):

    net.eval()
    columns = ["id", "image", "guess", "truth"]
    for digit in range(10):
        columns.append("score_" + str(digit))
    test_table = wandb.Table(columns)
    correct = total = 0

    with torch.no_grad():
        for count, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if count < 10:
                log_test_predictions(
                    images, labels, outputs, predicted, test_table, count
                )
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        wandb.log({"epoch": epoch, "test/accuracy": acc})
        logging.warning(
            "[Job ID = %s] Test: Epoch %d: Accuracy: %f", job_id, epoch, acc
        )
        wandb.log({"test/predictions": test_table})


@hydra.main(config_path="configs", config_name="defaults")
def run_experiment(cfg: omegaconf.DictConfig) -> None:

    base_path = pathlib.Path(hydra.utils.get_original_cwd())
    (train_loader, test_loader), _ = dataset.get_dataset(
        cfg.dataset.name,
        (base_path / "data") / cfg.dataset.dir,
        cfg.dataset.train_batch,
        cfg.dataset.test_batch,
    )
    criterion = torch.nn.CrossEntropyLoss()
    wandb_cfg = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    pprint.pprint(wandb_cfg)

    runtime_cfg = hydra.core.hydra_config.HydraConfig.get()
    job_id = runtime_cfg.job.get("id", -1)
    ismultirun = job_id != -1
    group_key = f"{cfg.dataset.name}-{cfg.model.norm_type}"
    with wandb.init(**cfg.wandb.setup, group=group_key, config=wandb_cfg) as run:

        net = model.ConvNet(
            cfg.dataset.image_dim, cfg.dataset.num_classes, **cfg.model
        ).to(device)
        wandb.watch(net, **cfg.wandb.watch)
        optimizer_class = getattr(torch.optim, cfg.train.optimizer)
        optimizer = optimizer_class(net.parameters(), lr=cfg.train.lr)

        epoch_iter = range(cfg.train.epochs)
        train_iter = train_loader

        if not ismultirun:
            epoch_iter = tqdm.tqdm(epoch_iter, position=0, leave=False)
        else:
            logging.warning("[MULTIRUN] Logged Metrics only available on :%s", run.url)

        for epoch in epoch_iter:

            net.train()

            if not ismultirun:
                train_iter = tqdm.tqdm(train_iter, position=1, leave=False)

            for images, labels in train_iter:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                wandb.log({"train/loss": loss})
            test_loop(test_loader, net, epoch, job_id)


if __name__ == "__main__":
    run_experiment()  # pylint: disable=no-value-for-parameter
