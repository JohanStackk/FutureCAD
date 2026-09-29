import argparse
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from grounding_brep.runtime import load_config, load_model, dataset


def main():
    parser = argparse.ArgumentParser('Train BRepGround.')
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--initialize-from", help="Load weights only, without optimizer state"
    )
    parser.add_argument("--resume", help="Resume a full Lightning training checkpoint")
    parser.add_argument(
        "--smoke", action="store_true", help="One training and validation batch"
    )
    args, overrides = parser.parse_known_args()
    if args.initialize_from and args.resume:
        parser.error("Choose either --initialize-from or --resume")
    cfg = load_config(args.config, overrides)
    pl.seed_everything(cfg.train_seed, workers=True)
    loaders = [
        dataset(cfg, s).get_dataloader(
            cfg.data.batch_size,
            s == "train" and cfg.data.get("train_shuffle", False),
            cfg.data.num_workers,
        )
        for s in ("train", "val")
    ]
    if any(len(loader.dataset) == 0 for loader in loaders):
        raise ValueError("Training and validation datasets must be nonempty")
    model = load_model(cfg, args.initialize_from)
    out = Path(cfg.trainer.output_dir)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=out / "checkpoints",
        monitor="val/f1",
        mode="max",
        filename="epoch{epoch:03d}-f1{val/f1:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        save_last=True,
        save_weights_only=False,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        default_root_dir=str(out),
        callbacks=[callback],
        logger=CSVLogger(str(out), "logs"),
        fast_dev_run=args.smoke,
    )
    trainer.fit(
        model,
        train_dataloaders=loaders[0],
        val_dataloaders=loaders[1],
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    main()
