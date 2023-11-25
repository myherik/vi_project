import torch
import numpy as np
import glob
import os

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    Lambdad,
    EnsureChannelFirstd,
)
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss, DiceCELoss
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from functools import partial

from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

base_path = "Task01_BrainTumour/"

train_images = sorted(glob.glob(os.path.join(base_path, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(base_path, "labelsTr", "*.nii.gz")))

dataset_list = [
    {"img": image_name, "seg": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

"""
for file in os.listdir(os.path.join(base_path, "imagesTr")):

    data_dict = {
        "img": os.path.join(base_path, "imagesTr", file), 
        "seg": os.path.join(base_path, "labelsTr", file)
    }
    dataset_list.append(data_dict)"""

train_dataset_list, val_dataset_list = dataset_list[:-94], dataset_list[-94:]
print(len(train_dataset_list), len(val_dataset_list))

train_transform = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            #Lambdad(keys=["img", "seg"], func=lambda x: x.permute(3,2,0,1)),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Lambdad(keys=["img", "seg"], func=lambda x: x.permute(0,3,1,2)),
            CropForegroundd(
                keys=["img", "seg"],
                source_key="img",
                #k_divisible=[128, 128, 128],
                #in_place=True,
            ),
            RandSpatialCropd(
                keys=["img", "seg"],
                roi_size=[128, 128, 128],
                random_size=False,
                #in_place=True,
            ),
            #RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
            #RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=1),
            #RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
            #RandScaleIntensityd(keys="img", factors=0.1, prob=1.0),
            #RandShiftIntensityd(keys="img", offsets=0.1, prob=1.0),
        ]
    )

val_transform = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Lambdad(keys=["img", "seg"], func=lambda x: x.permute(0,3,1,2)),
            NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ]
    )

train_dataset = CacheDataset(train_dataset_list, transform=train_transform, cache_rate=1.0, num_workers=os.cpu_count() // 2)
val_dataset = CacheDataset(val_dataset_list, transform=val_transform, cache_rate=1.0, num_workers=os.cpu_count() // 2)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=os.cpu_count() // 2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=os.cpu_count() // 2)

model = SwinUNETR(
    in_channels=4,
    out_channels=4,
    img_size=128,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
).to(device)

def calculate_iou(predicted, target):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Args:
    - predicted (torch.Tensor): Predicted binary mask (0 or 1).
    - target (torch.Tensor): Ground truth binary mask (0 or 1).

    Returns:
    - float: IoU score.
    """
    intersection = torch.logical_and(predicted, target).sum().item()
    union = torch.logical_or(predicted, target).sum().item()

    iou = intersection / union if union != 0 else 0.0
    return iou

loss_function = DiceCELoss(to_onehot_y=True, softmax=True) #torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), 0.1)

torch.backends.cudnn.benchmark = True
loss_function = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_accuracy = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[128, 128, 128],
    sw_batch_size=4,
    predictor=model,
    overlap=0.5,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

num_epochs = 3
losses = []
test_losses = []
ious = []
test_ious = []
dice_accs = []
test_dice_accs = []

for epoch in range(num_epochs):

    epoch_loss = 0
    epoch_test_loss = 0
    epoch_iou = 0
    epoch_test_iou = 0

    model.train()
    for idx, batch_data in enumerate(tqdm(train_loader)):
        img, seg = batch_data["img"].to(device), batch_data["seg"].to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_function(output, seg)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        acc, not_nans = dice_accuracy(y_pred=post_pred(post_sigmoid(output)), y=seg)
        epoch_loss += loss.item()
        epoch_iou += calculate_iou(output, seg)
        dice_accs.append(acc.item())
        
        iou = calculate_iou(output, seg)
        ious.append(iou)

        
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            img, seg = batch_data["img"].to(device), batch_data["seg"].to(device)
            output = model_inferer(img)
            val_labels_list = decollate_batch(seg)
            val_outputs_list = decollate_batch(output)
            val_output_convert = [post_pred(post_sigmoid(el)) for el in val_outputs_list]
            dice_accuracy.reset()
            dice_accuracy(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = dice_accuracy.aggregate()
            test_dice_accs.append(acc.item())

            loss = loss_function(output, seg)
            test_losses.append(loss.item())
            epoch_test_loss += loss.item()

            iou = calculate_iou(output, seg)
            test_ious.append(iou)

            epoch_test_iou += calculate_iou(output, seg)
    
    print(f"""
            Epoch: {epoch} / {num_epochs} 
            | Train Loss: {epoch_loss / len(train_loader)} 
            | Test Loss: {epoch_test_loss / len(val_loader)} 
            | Train IoU: {ious[-1]} 
            | Test IoU: {test_ious[-1]}
            | Train Dice Accuracy: {dice_accs[-1]} 
            | Test Dice Accuracy: {test_dice_accs[-1]}
        """)

    scheduler.step()
        