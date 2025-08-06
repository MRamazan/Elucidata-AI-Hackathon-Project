import os
import random
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

from scipy.stats import spearmanr
from tqdm import tqdm
import imgaug
import wandb

CONFIG = {
    "seed": 42,
    "data_path": "el-hackathon-2025",
    "output_dir": "model_outputs",
    "batch_size": 32,
    "num_workers": 6,
    "learning_rate": 0.003,
    "weight_decay": 1e-1,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.1,
    "num_classes": 35,
    "image_size": (162, 162),
    "patch_size": 54,
    "max_epochs": 5,
    "checkpoint_epochs": [3,4,5,6,8,10,13,20,24], # epoch 3 performed best
    "use_wandb": False,  # Set to False if you don't want to use wandb
    "model_type": "resnet18",
    "mixed_precision": True,  # Use mixed precision training
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    imgaug.seed(seed)


def get_device():
    """Get the appropriate device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") 
    else:
        return torch.device("cpu")


def spearman_rank_correlation(x, y):
    """Calculate Spearman rank correlation"""

    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return spearmanr(x, y)[0]


def spearman_corr(preds, targets):
    """Calculate mean Spearman correlation across all samples"""
    correlations = []
    for i in range(len(preds)):
        corr = spearman_rank_correlation(preds[i], targets[i])
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations)


class HackhathonDataset(Dataset):
    """Dataset class for the hackathon competition"""

    def __init__(self, data_path, transform=None, mode="train"):
        self.data_path = data_path
        self.materials = []
        self.transform = transform

        train_slides = ["S_1","S_2","S_3", "S_4", "S_5"]
        val_slide = ["S_6"]
        test_slide = ["S_7"]
        self.mode = mode

        slide_list = train_slides if mode == "train" else val_slide if mode == "val" else test_slide

        with h5py.File(f"{self.data_path}/elucidata_ai_challenge_data.h5", "r") as h5file:
            images_group = "images/Train" if mode != "test" else "images/Test"
            spots_group = "spots/Train" if mode != "test" else "spots/Test"

            train_images = h5file[images_group]
            train_spots = h5file[spots_group]

            for slide_name in tqdm(slide_list, desc=f"Loading {mode} data"):
                if slide_name in train_images.keys():
                    image = np.array(train_images[slide_name])
                    spots = np.array(train_spots[slide_name])
                    df = pd.DataFrame(spots)
                    self._split_into_patches(image, df, CONFIG["patch_size"])

        print(f"{len(self.materials)} patches initialized for {mode} set")

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, idx):
        image, stats = self.materials[idx]

        if self.transform:
            image = self.transform(image)

        stats = torch.tensor(stats[2:], dtype=torch.float32)
        return image, stats


    def _split_into_patches(self, arr, df, patch_size):
        """Split the image into patches centered on spot coordinates"""
        h, w, c = arr.shape

        for idx in range(len(df)):
            row = df.iloc[idx]
            x, y = int(row["x"]), int(row["y"])

            half_size = patch_size // 2

            y_min = max(y - half_size, 0)
            y_max = min(y + half_size, h)
            x_min = max(x - half_size, 0)
            x_max = min(x + half_size, w)

            patch = arr[y_min:y_max, x_min:x_max, :]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                self.materials.append([patch, row])
            else:
                padded_patch = np.zeros((patch_size, patch_size, c), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                self.materials.append([padded_patch, row])


import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableSpearmanLoss(nn.Module):
    """Differentiable approximation of Spearman correlation loss"""

    def __init__(self, regularization_strength=1.0):
        super().__init__()
        self.regularization_strength = regularization_strength

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
      
        pred_rank = self._soft_rank(y_pred)
        true_rank = self._soft_rank(y_true)


        pred_rank = F.normalize(pred_rank, dim=1)
        true_rank = F.normalize(true_rank, dim=1)

        spearman = torch.sum(pred_rank * true_rank, dim=1)
        return 1 - spearman.mean()

    def _soft_rank(self, x, regularization_strength=None):
        if regularization_strength is None:
            regularization_strength = self.regularization_strength

        x = x.unsqueeze(-1)  # [batch, n, 1]
        diff = x - x.transpose(-1, -2)  # [batch, n, n]
        P = torch.sigmoid(-regularization_strength * diff)  # pairwise comparisons
        ranks = P.sum(dim=-1)  # approximate ranks
        return ranks


class CombinedLoss(nn.Module):
    """Combined loss function using L1 and Spearman correlation"""

    def __init__(self, alpha=0.5, regularization_strength=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.spearman = DifferentiableSpearmanLoss(regularization_strength)
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        l1_loss = self.l1(y_pred, y_true)
        spearman_loss = self.spearman(y_pred, y_true)
        return l1_loss + self.alpha * spearman_loss




def create_model(model_type, num_classes):
    """Create and initialize the model"""
    if model_type == "resnet18":
        # model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        MODEL_PATH = '/home/lm/Downloads/tenpercent_resnet18.ckpt'
        RETURN_PREACTIVATION = False  # return features from the model, if false return classification logits
        NUM_CLASSES = 35  # only used if RETURN_PREACTIVATION = False

        def load_model_weights(model, weights):

            model_dict = model.state_dict()
            weights = {k: v for k, v in weights.items() if k in model_dict}
            if weights == {}:
                print('No weight could be loaded..')
            model_dict.update(weights)
            model.load_state_dict(model_dict)

            return model

        model = models.resnet18(pretrained=False)

        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model = load_model_weights(model, state_dict)

        if RETURN_PREACTIVATION:
            model.fc = torch.nn.Sequential()
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transforms():
    """Get data transformations for training and validation"""
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize(CONFIG["image_size"]),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Resize(CONFIG["image_size"]),
    ])

    return train_transform, val_transform


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler=None):
    """Train model for one epoch"""
    model.train()

    epoch_loss = 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc="Training")

    for images,labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        if scaler:  # Using mixed precision
          with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()


        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})


    avg_loss = epoch_loss / len(dataloader)
    spearman_score = spearman_corr(all_preds, all_labels)

    return avg_loss, spearman_score, all_preds, all_labels


def validate(model, dataloader, loss_fn, device):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

  
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    avg_loss = val_loss / len(dataloader)
    spearman_score = spearman_corr(all_preds, all_labels)

    return avg_loss, spearman_score, all_preds, all_labels


def get_tta_transforms():
    """Get test-time augmentation transformations"""
    tta_transforms = [
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
        ]),
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomHorizontalFlip(p=1.0),
        ]),
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomVerticalFlip(p=1.0),
        ]),
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomRotation(degrees=(90, 90)),
        ]),
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomRotation(degrees=(180, 180)),
        ]),
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]),
    ]
    return tta_transforms

def predict_test_set_with_tta(model, data_path, device):
    """Generate predictions for the test set using Test Time Augmentation"""
    tta_transforms = get_tta_transforms()

    with h5py.File(f"{data_path}/elucidata_ai_challenge_data.h5", "r") as f:
        test_spots = f["spots/Test"]
        test_images = f["images/Test"]
        sample = 'S_7' 
        image = np.array(test_images[sample])
        spots = np.array(test_spots[sample])
        x, y = spots["x"], spots["y"]

        outputs = []
        model.eval()
        with torch.no_grad():
            patch_size = CONFIG["patch_size"]
            for x_, y_ in tqdm(zip(x, y), desc="Generating predictions with TTA", total=len(x)):
                half_size = patch_size // 2
              
                y_min = max(y_ - half_size, 0)
                y_max = min(y_ + half_size, image.shape[0])
                x_min = max(x_ - half_size, 0)
                x_max = min(x_ + half_size, image.shape[1])
                patch = image[y_min:y_max, x_min:x_max, :]

                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch

                patch_predictions = []
                for transform in tta_transforms:
                    patch_tensor = transform(patch)
                    patch_tensor = patch_tensor.to(device)
                    output = model(patch_tensor.unsqueeze(0)).cpu().numpy()
                    patch_predictions.append(output[0])

         
                avg_prediction = np.mean(patch_predictions, axis=0)
                outputs.append(avg_prediction)

    return np.array(outputs), x, y


def save_submission(predictions, data_path, epoch, model_name):
    """Save predictions to submission file"""
    example_df = pd.read_csv(f"{data_path}/submission (1).csv")
    ID = example_df["ID"]
    output_df = pd.DataFrame(predictions)
    submission_df = pd.concat([ID, output_df], axis=1)
    submission_df.columns = example_df.columns

    output_file = f"{CONFIG['output_dir']}/{model_name}_epoch_{epoch}.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")

    return output_file

def main():
    """Main training function"""
    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Using device: {device}")


    train_transform, val_transform = get_transforms()

    train_dataset = HackhathonDataset(
        CONFIG["data_path"],
        transform=train_transform,
        mode="train"
    )
    val_dataset = HackhathonDataset(
        CONFIG["data_path"],
        transform=val_transform,
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,

    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,

    )


    model = create_model(CONFIG["model_type"], CONFIG["num_classes"])
    model = model.to(device)

 
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG["scheduler_gamma"],
        patience=3,
    )


    scaler = torch.cuda.amp.GradScaler() if CONFIG["mixed_precision"] and device.type == "cuda" else None

  
    if CONFIG["use_wandb"]:
        wandb.init(
            project="hackathon-gene-expression",
            config=CONFIG,
            name=f"{CONFIG['model_type']}_run"
        )

        wandb.watch(model)


    best_val_spearman = -1
    for epoch in range(CONFIG["max_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['max_epochs']}")
    
        train_loss, train_spearman, _, _ = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )
    
        val_loss, val_spearman, _, _ = validate(
            model, val_loader, loss_fn, device
        )

  
        scheduler.step(val_loss)

 
        metrics = {
            "train_loss": train_loss,
            "train_spearman": train_spearman,
            "val_loss": val_loss,
            "val_spearman": val_spearman,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        print(f"Train Loss: {train_loss:.4f}, Train Spearman: {train_spearman:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Spearman: {val_spearman:.4f}")

        if CONFIG["use_wandb"]:
            wandb.log(metrics)

     
        if epoch in CONFIG["checkpoint_epochs"]:
            test_preds, _, _ = predict_test_set_with_tta(model, CONFIG["data_path"], device)

          
            submission_file = save_submission(
                test_preds, CONFIG["data_path"], epoch, CONFIG["model_type"]
            )

            if CONFIG["use_wandb"]:
                wandb.save(submission_file)

    if CONFIG["use_wandb"]:
        wandb.finish()

    print("Training completed!")


if __name__ == "__main__":
    main()
