import os
import sys
import argparse
from tqdm import tqdm

import torch

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

from medico_sam.models.monai_models import get_monai_models


sys.path.append("..")


def train_swinunetr(args):
    """Train SwinUNETR model for semantic segmentation on medical imaging datasets.

    The scripts below are inspired from
    https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb.

    NOTE: I just copy-pasted the code. I didn't want to optimize the code / make it better at all.
    """
    from common import get_num_classes, DATASETS_2D, DATASETS_3D, get_dataloaders, get_in_channels

    # Some training settings.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    num_classes = get_num_classes(dataset)
    in_channels = get_in_channels(dataset)

    if dataset in DATASETS_2D:
        patch_shape = (1024, 1024)
        tile_shape = (96, 96)
        ndim = 2
    elif dataset in DATASETS_3D:
        patch_shape = (32, 512, 512)
        tile_shape = (32, 96, 96)
        ndim = 3
    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name or not part of our experiments yet.")

    model = get_monai_models(
        image_size=patch_shape, in_channels=in_channels, out_channels=num_classes, ndim=ndim,
    )
    model.to(device)

    # All stuff we need for training.
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.GradScaler("cuda")
    train_loader, val_loader = get_dataloaders(
        patch_shape=patch_shape, data_path=args.input_path, dataset_name=dataset, benchmark_models=True,
    )
    save_root = os.path.join(
        os.getcwd() if args.save_root is None else args.save_root, "checkpoints", f"{dataset}_swinunetr"
    )
    os.makedirs(save_root, exist_ok=True)

    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch[0].to("cuda"), batch[1].to("cuda"))
                with torch.autocast("cuda"):
                    val_outputs = sliding_window_inference(val_inputs, tile_shape, 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = batch
            x, y = x.to("cuda"), y.to("cuda")
            with torch.autocast("cuda"):
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
            )
            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(save_root, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            global_step += 1
        return global_step, dice_val_best, global_step_best

    max_iterations = args.iterations
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)

    print(f"Train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


def main():
    parser = argparse.ArgumentParser(description="Train SwinUNETR model for the semantic segmentation tasks.")
    parser.add_argument(
        "-d", "--dataset", required=True, help="The name of medical dataset for semantic segmentation."
    )
    parser.add_argument(
        "-i", "--input_path", default="/mnt/vast-nhr/projects/cidas/cca/data",
        help="The filepath to the medical data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e5), help="For how many iterations should the model be trained?"
    )
    args = parser.parse_args()
    train_swinunetr(args)


if __name__ == "__main__":
    main()
