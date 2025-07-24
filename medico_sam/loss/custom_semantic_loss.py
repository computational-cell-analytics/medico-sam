import torch

from micro_sam.training.semantic_sam_trainer import CustomDiceLoss


class CustomCombinedLoss(torch.nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()

        self.dice_weight = dice_weight
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dice_loss = CustomDiceLoss(num_classes=num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, pred, target):
        pred = pred.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        # Compute the dice loss.
        dice_loss = self.dice_loss(pred, target)

        # Compute cross entropy loss.
        ce_loss = self.ce_loss(pred, target.squeeze(1).long())

        # Get the overall computed loss.
        net_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss

        return net_loss
