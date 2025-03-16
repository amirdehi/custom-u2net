import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler  # ✅ Fixed Autocast Import

from data_loader import RandomCrop, RescaleT, SalObjDataset, ToTensorLab
from model import U2NET, U2NETP

# ------- 1. Define Loss Function --------
bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # ✅ Fixed BCELoss issue


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    # ✅ Clamp predictions to prevent extreme values
    d0 = torch.clamp(d0, min=-10, max=10)
    d1 = torch.clamp(d1, min=-10, max=10)
    d2 = torch.clamp(d2, min=-10, max=10)
    d3 = torch.clamp(d3, min=-10, max=10)
    d4 = torch.clamp(d4, min=-10, max=10)
    d5 = torch.clamp(d5, min=-10, max=10)
    d6 = torch.clamp(d6, min=-10, max=10)

    losses = [bce_loss(d, labels_v) for d in [d0, d1, d2, d3, d4, d5, d6]]
    total_loss = sum(losses)

    # Debugging loss values
    if torch.isnan(total_loss):
        print("❌ NaN detected in loss!")

    return losses[0], total_loss


# ------- 2. Set Training Parameters --------
model_name = 'u2net'  # Change to 'u2netp' for lightweight model
data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = 'im_aug' + os.sep
tra_label_dir = 'gt_aug' + os.sep
image_ext, label_ext = '.JPG', '.jpg'
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 20
batch_size_train = 8

# ------- 3. Load Data --------
tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, '*' + image_ext))
tra_lbl_name_list = [os.path.join(data_dir, tra_label_dir, os.path.basename(img).replace(image_ext, label_ext))
                     for img in tra_img_name_list]

print(f"Training images: {len(tra_img_name_list)}, Training labels: {len(tra_lbl_name_list)}")

train_transforms = transforms.Compose([
    RescaleT(320),
    RandomCrop(288),
    ToTensorLab(flag=0)
])

salobj_dataset = SalObjDataset(tra_img_name_list, tra_lbl_name_list, transform=train_transforms)
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 4. Load Model & Freeze Layers --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = U2NET(3, 1) if model_name == 'u2net' else U2NETP(3, 1)

pretrained_path = os.path.join("saved_models/pretrained", "u2net.pth")
if os.path.exists(pretrained_path):
    print(f"Loading pretrained model from {pretrained_path}")
    net.load_state_dict(torch.load(pretrained_path, map_location=device))

net.to(device)

# Freeze first half of layers initially
for param in list(net.parameters())[:len(list(net.parameters())) // 2]:
    param.requires_grad = False

# ------- 5. Define Optimizer & Scheduler --------
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-6)  # ✅ Cosine Annealing for LR
scaler = GradScaler("cuda")  # ✅ Fixed GradScaler Import

# Check for NaNs in Labels
for i, data in enumerate(salobj_dataloader):
    inputs, labels = data['image'].to(device, dtype=torch.float32), data['label'].to(device, dtype=torch.float32)

    # ✅ Debug: Check if labels have NaNs
    if torch.isnan(labels).any():
        print(f"❌ NaN detected in labels at batch {i}")
        continue  # Skip bad batch

    # ✅ Debug: Check label range (should be between 0 and 1)
    if labels.min() < 0 or labels.max() > 1:
        print(f"⚠️ Warning: Label values out of range in batch {i}")

    break  # Stop after first check

# ------- 6. Training Loop --------
print("Starting Training...")
for epoch in range(epoch_num):
    net.train()

    # Gradually unfreeze layers at later epochs
    if epoch == 10:
        for param in net.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(net.parameters(), lr=5e-6, weight_decay=1e-5)  # ✅ Lowered LR

    running_loss, running_tar_loss = 0.0, 0.0
    for i, data in enumerate(salobj_dataloader):
        inputs, labels = data['image'].to(device, dtype=torch.float32), data['label'].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast("cuda"):  # ✅ Mixed precision
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

        # ✅ Replace NaNs in predictions with 0 before loss calculation
        d0, d1, d2, d3, d4, d5, d6 = [torch.nan_to_num(d) for d in [d0, d1, d2, d3, d4, d5, d6]]

        # ✅ Replace NaNs in labels (just in case)
        labels = torch.nan_to_num(labels)

        # Compute loss
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_tar_loss += loss2.item()

        print(
            f"[Epoch {epoch + 1}/{epoch_num}, Batch {i + 1}/{len(salobj_dataloader)}] Loss: {running_loss / (i + 1):.5f}")

    scheduler.step()

    # Save model at each epoch
    model_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch + 1}.pth")
    torch.save(net.state_dict(), model_path)
    print(f"Model saved at {model_path}")

print("Training Complete!")
