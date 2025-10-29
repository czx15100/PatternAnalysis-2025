"""
Training script for U-Net on OASIS MRI Brain Segmentation Dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import OASISDataset
from modules import UNet

def dice_coefficient(pred, target, num_classes):
    """Calculate the Dice coefficient for segmentation."""
    pred = torch.argmax(pred, dim=1)
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).float().sum()
        union = pred_c.float().sum() + target_c.float().sum()
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_scores.append(dice)
    return torch.mean(torch.stack(dice_scores))

def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    """Train for one epoch."""
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs) # Forward pass
        loss = criterion(outputs, masks) # Compute loss
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        # Accumulate loss and dice
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks, num_classes).item()

    return total_loss / len(dataloader), total_dice / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device, num_classes):
    """Validate for one epoch."""
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks, num_classes).item()

    return total_loss / len(dataloader), total_dice / len(dataloader)


def main():
    """Main training loop."""
    # Hyperparameters
    batch_size = 4
    num_epochs = 80
    learning_rate = 0.0001
    num_classes = 3
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Datasets
    #train_ds = OASISDataset("keras_png_slices_train", "keras_png_slices_seg_train")
    #val_ds = OASISDataset("keras_png_slices_validate", "keras_png_slices_seg_validate")
    train_ds = OASISDataset("/home/groups/comp3710/OASIS/keras_png_slices_train", "/home/groups/comp3710/OASIS/keras_png_slices_seg_train")
    val_ds = OASISDataset("/home/groups/comp3710/OASIS/keras_png_slices_validate", "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model setup
    model = UNet(in_channels=1, num_classes=num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_dice = 0.0
    train_losses, val_losses, train_dice_scores, val_dice_scores = [], [], [], []

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, num_classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_scores.append(train_dice)
        val_dice_scores.append(val_dice)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")

    # Plot curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_dice_scores, label="Train Dice")
    plt.plot(val_dice_scores, label="Val Dice")
    plt.title("Dice Coefficient")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

# Run the training process
if __name__ == "__main__":
    main()