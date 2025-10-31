#!/usr/bin/env python3
"""
main.py - Point-Supervised Semantic Segmentation for Massachusetts Buildings Dataset

Complete Assignment Implementation:
1. Partial Cross-Entropy Loss (Task 1)
2. Real Remote Sensing Dataset with Point Annotations (Task 2)
3. Transfer Learning + Semi-Supervised + TTA + Ensemble (Advanced Features)
4. Two Comprehensive Experiments (Task 3)


Usage:
    # Run all experiments
    python main.py --mode experiments --data_root /path/to/prepared
    
    # Train single model
    python main.py --mode train --data_root /path/to/prepared --points_per_image 200
    
    # Inference with ensemble
    python main.py --mode inference --data_root /path/to/prepared --ckpt_paths model1.pth model2.pth
"""

import os
import sys
import random
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Segmentation models
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("ERROR: segmentation_models_pytorch required. Install: pip install segmentation-models-pytorch")
    sys.exit(1)

# Optional libraries
try:
    from skimage.exposure import match_histograms
    HAS_HIST_MATCH = True
except ImportError:
    HAS_HIST_MATCH = False

# Metrics
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    """Create directory if doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def pil_loader(path):
    """Load image as RGB"""
    return Image.open(path).convert('RGB')

def pil_loader_mask(path):
    """Load mask as grayscale"""
    im = Image.open(path)
    if im.mode in ['RGB', 'RGBA']:
        im = im.convert('L')
    return im

# ============================================================
# TASK 1: PARTIAL CROSS-ENTROPY LOSS IMPLEMENTATION
# ============================================================

class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for point-supervised learning.
    
    This loss function is the CORE of point-supervised segmentation.
    It computes cross-entropy only on pixels with point annotations,
    completely ignoring unlabeled pixels during training.
    
    Mathematical Formula:
        L_PCE = -Σ(y_i * log(p_i) * m_i) / Σ(m_i)
    
    Where:
        y_i: One-hot encoded ground truth for pixel i
        p_i: Predicted probability for pixel i (after softmax)
        m_i: Binary mask (1 if pixel i is annotated, 0 otherwise)
        Σ(m_i): Total number of annotated pixels (normalization)
    
    Key Properties:
        1. Selective Computation: Only labeled pixels contribute gradients
        2. Normalized Loss: Prevents bias from varying annotation density
        3. Memory Efficient: No need to store full masks
        4. Flexible: Works with arbitrary point distributions
    
    Args:
        ignore_index (int): Value for unlabeled pixels (default: -1)
        focal_gamma (float): Focal loss parameter (0 = standard CE, >0 = focal)
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, ignore_index=-1, focal_gamma=0.0, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_gamma = focal_gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets, mask=None):
        """
        Compute partial cross-entropy loss.
        
        Args:
            predictions: (B, C, H, W) tensor of logits from model
            targets: (B, H, W) tensor of ground truth labels
            mask: (B, H, W) tensor, optional binary mask
        
        Returns:
            loss: Scalar tensor
        """
        device = predictions.device
        B, C, H, W = predictions.shape
        
        # Compute log probabilities
        log_probs = F.log_softmax(predictions, dim=1)
        probs = F.softmax(predictions, dim=1)
        
        # Create mask if not provided
        if mask is None:
            mask = (targets != self.ignore_index).float()
        
        mask = mask.to(device)
        targets = targets.to(device)
        
        # Flatten spatial dimensions
        log_probs_flat = log_probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        probs_flat = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        # Handle ignore_index
        valid_mask = (targets_flat != self.ignore_index).float()
        targets_safe = targets_flat.clone()
        targets_safe[targets_flat == self.ignore_index] = 0
        
        # Get log probability of true class
        log_probs_true = log_probs_flat.gather(1, targets_safe.unsqueeze(1)).squeeze(1)
        probs_true = probs_flat.gather(1, targets_safe.unsqueeze(1)).squeeze(1)
        
        # Focal loss weighting
        if self.focal_gamma > 0:
            focal_weight = (1 - probs_true) ** self.focal_gamma
        else:
            focal_weight = 1.0
        
        # Compute focal cross-entropy
        focal_loss = -focal_weight * log_probs_true
        
        # Apply mask
        masked_loss = focal_loss * mask_flat * valid_mask
        
        # Normalize
        num_labeled = (mask_flat * valid_mask).sum()
        
        if self.reduction == 'mean':
            if num_labeled > 0:
                return masked_loss.sum() / num_labeled
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return masked_loss


class ConsistencyLoss(nn.Module):
    """Consistency loss for semi-supervised learning"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred1, pred2, unlabeled_mask):
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)
        mse = F.mse_loss(prob1, prob2, reduction='none')
        mse = mse.mean(dim=1)
        masked_mse = (mse * unlabeled_mask).sum() / unlabeled_mask.sum().clamp(min=1)
        return masked_mse


# ============================================================
# TASK 2: MASSACHUSETTS BUILDINGS DATASET WITH POINT ANNOTATIONS
# ============================================================

class MassachusettsDataset(Dataset):
    """
    Massachusetts Buildings Dataset with Point Annotation Simulation.
    """
    def __init__(self, root, split='train', img_size=256, num_points=200,
                 augment=False, hist_ref=None, seed=42):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.num_points = num_points
        self.augment = augment
        self.seed = seed
        
        self.img_dir = self.root / 'images' / split
        self.mask_dir = self.root / 'masks' / split
        
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Masks folder not found: {self.mask_dir}")
        
        self.img_paths = sorted([p for p in self.img_dir.glob('*') 
                                if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
        self.mask_paths = sorted([p for p in self.mask_dir.glob('*')
                                 if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']])
        
        if len(self.img_paths) != len(self.mask_paths):
            mask_dict = {p.stem: p for p in self.mask_paths}
            paired = []
            for img_path in self.img_paths:
                mask_path = mask_dict.get(img_path.stem)
                if mask_path:
                    paired.append((img_path, mask_path))
            
            if not paired:
                raise RuntimeError(f"Cannot pair images and masks in {split} split")
            
            self.img_paths, self.mask_paths = zip(*paired)
        
        self.hist_ref = None
        if hist_ref and HAS_HIST_MATCH:
            try:
                ref_img = pil_loader(hist_ref)
                ref_img = ref_img.resize((img_size, img_size))
                self.hist_ref = np.array(ref_img).astype(np.uint8)
            except Exception as e:
                print(f"Warning: Could not load histogram reference: {e}")
        
        self.rng = np.random.RandomState(seed + hash(split) % 10000)
        
        print(f"{split.capitalize()} dataset: {len(self.img_paths)} images, "
              f"{num_points} points each, coverage: {(num_points/(img_size**2)*100):.3f}%")
    
    def __len__(self):
        return len(self.img_paths)
    
    def _hist_match(self, img_np):
        if not HAS_HIST_MATCH or self.hist_ref is None:
            return img_np
        try:
            matched = match_histograms(img_np, self.hist_ref, channel_axis=2)
            return matched.astype(np.float32)
        except Exception:
            return img_np
    
    def _sample_points(self, mask_np):
        H, W = mask_np.shape
        point_mask = np.full((H, W), -1, dtype=np.int64)
        point_loc = np.zeros((H, W), dtype=np.float32)
        
        bg_coords = np.argwhere(mask_np == 0)
        fg_coords = np.argwhere(mask_np == 1)
        
        n_fg = min(self.num_points // 2, len(fg_coords))
        n_bg = min(self.num_points - n_fg, len(bg_coords))
        
        if n_fg > 0 and len(fg_coords) > 0:
            fg_indices = self.rng.choice(len(fg_coords), n_fg, replace=False)
            for idx in fg_indices:
                y, x = fg_coords[idx]
                point_mask[y, x] = 1
                point_loc[y, x] = 1.0
        
        if n_bg > 0 and len(bg_coords) > 0:
            bg_indices = self.rng.choice(len(bg_coords), n_bg, replace=False)
            for idx in bg_indices:
                y, x = bg_coords[idx]
                point_mask[y, x] = 0
                point_loc[y, x] = 1.0
        
        sampled = int((point_loc == 1.0).sum())
        remaining = self.num_points - sampled
        
        if remaining > 0:
            all_pixels = H * W
            for _ in range(remaining * 3):
                if (point_loc == 1.0).sum() >= self.num_points:
                    break
                idx = self.rng.randint(0, all_pixels)
                y, x = idx // W, idx % W
                if point_loc[y, x] == 0:
                    point_mask[y, x] = int(mask_np[y, x])
                    point_loc[y, x] = 1.0
        
        return point_mask, point_loc
    
    def _load_pair(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = pil_loader(img_path)
        img_np = np.array(img).astype(np.float32) / 255.0
        
        mask = pil_loader_mask(mask_path)
        mask_np = np.array(mask).astype(np.int64)
        mask_np = (mask_np > 127).astype(np.int64)
        
        img_np = self._hist_match(img_np)
        
        return img_np, mask_np
    
    def _resize(self, img_np, mask_np):
        if img_np.shape[0] != self.img_size or img_np.shape[1] != self.img_size:
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 127).astype(np.int64)
        
        return img_np, mask_np
    
    def _augment(self, img, mask, point_mask, point_loc):
        if self.rng.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            point_mask = np.fliplr(point_mask).copy()
            point_loc = np.fliplr(point_loc).copy()
        
        if self.rng.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
            point_mask = np.flipud(point_mask).copy()
            point_loc = np.flipud(point_loc).copy()
        
        k = self.rng.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
            point_mask = np.rot90(point_mask, k).copy()
            point_loc = np.rot90(point_loc, k).copy()
        
        return img, mask, point_mask, point_loc
    
    def __getitem__(self, idx):
        img_np, mask_np = self._load_pair(idx)
        img_np, mask_np = self._resize(img_np, mask_np)
        point_mask_np, point_loc_np = self._sample_points(mask_np)
        
        if self.augment:
            img_np, mask_np, point_mask_np, point_loc_np = self._augment(
                img_np, mask_np, point_mask_np, point_loc_np
            )
        
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        full_mask = torch.from_numpy(mask_np).long()
        point_mask = torch.from_numpy(point_mask_np).long()
        point_loc = torch.from_numpy(point_loc_np).float()
        
        return img_tensor, full_mask, point_mask, point_loc


# ============================================================
# MODEL BUILDING WITH TRANSFER LEARNING
# ============================================================

def build_model(backbone='resnet34', pretrained=True, num_classes=2, freeze_encoder=False):
    """Build U-Net model with transfer learning"""
    encoder_weights = 'imagenet' if pretrained else None
    
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    if freeze_encoder and pretrained:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"✓ Encoder '{backbone}' frozen for transfer learning")
    
    return model


# ============================================================
# SEMI-SUPERVISED TRAINING
# ============================================================

def train_semi_supervised_epoch(model, dataloader, optimizer, device,
                                sup_criterion, cons_criterion=None,
                                semi_lambda=0.5, pseudo_thresh=0.95,
                                consistency_weight=0.1):
    """Train one epoch with semi-supervised learning"""
    model.train()
    total_sup_loss = 0
    total_pseudo_loss = 0
    total_cons_loss = 0
    
    for images, _, point_masks, point_locs in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        point_masks = point_masks.to(device)
        point_locs = point_locs.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Supervised loss
        labeled_mask = (point_masks != -1).float()
        sup_loss = sup_criterion(outputs, point_masks, labeled_mask)
        
        # Pseudo-label loss
        pseudo_loss = torch.tensor(0.0, device=device)
        if semi_lambda > 0:
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
            
            unlabeled_mask = (point_masks == -1)
            reliable_mask = (max_probs > pseudo_thresh) & unlabeled_mask
            
            if reliable_mask.sum() > 0:
                pseudo_loss = sup_criterion(outputs, pseudo_labels, reliable_mask.float())
        
        # Consistency loss
        cons_loss = torch.tensor(0.0, device=device)
        if cons_criterion and consistency_weight > 0:
            model.train()
            outputs2 = model(images)
            unlabeled_mask_float = (point_masks == -1).float()
            cons_loss = cons_criterion(outputs, outputs2, unlabeled_mask_float)
        
        total_loss = sup_loss + semi_lambda * pseudo_loss + consistency_weight * cons_loss
        
        total_loss.backward()
        optimizer.step()
        
        total_sup_loss += sup_loss.item()
        if isinstance(pseudo_loss, torch.Tensor):
            total_pseudo_loss += pseudo_loss.item()
        if isinstance(cons_loss, torch.Tensor):
            total_cons_loss += cons_loss.item()
    
    n = len(dataloader)
    return total_sup_loss / n, total_pseudo_loss / n, total_cons_loss / n


# ============================================================
# TEST-TIME AUGMENTATION
# ============================================================

class TTAWrapper:
    """Test-Time Augmentation wrapper"""
    def __init__(self, model):
        self.model = model
    
    def predict_with_tta(self, images, device):
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            images = images.to(device)
            
            pred = self.model(images)
            all_preds.append(F.softmax(pred, dim=1))
            
            pred = self.model(torch.flip(images, dims=[3]))
            pred = torch.flip(pred, dims=[3])
            all_preds.append(F.softmax(pred, dim=1))
            
            pred = self.model(torch.flip(images, dims=[2]))
            pred = torch.flip(pred, dims=[2])
            all_preds.append(F.softmax(pred, dim=1))
            
            pred = self.model(torch.rot90(images, k=1, dims=[2, 3]))
            pred = torch.rot90(pred, k=-1, dims=[2, 3])
            all_preds.append(F.softmax(pred, dim=1))
        
        avg_pred = torch.stack(all_preds).mean(dim=0)
        return avg_pred


def tta_predict_simple(model, images, device):
    """Simple TTA with horizontal flip only"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        out1 = F.softmax(model(images), dim=1)
        out2 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
        out2 = torch.flip(out2, dims=[3])
        avg = (out1 + out2) / 2.0
    return avg


# ============================================================
# ENSEMBLE LEARNING
# ============================================================

class EnsembleModel:
    """Ensemble of multiple models"""
    def __init__(self, models):
        self.models = models
        for model in self.models:
            model.eval()
    
    def predict(self, images, device, use_tta=False):
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                if use_tta:
                    probs = tta_predict_simple(model, images, device)
                else:
                    probs = F.softmax(model(images.to(device)), dim=1)
                all_preds.append(probs)
        
        avg_probs = torch.stack(all_preds).mean(dim=0)
        return avg_probs


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, dataloader, device, use_tta=False, tta_wrapper=None):
    """Evaluate model on full masks"""
    if use_tta and tta_wrapper is None:
        tta_wrapper = TTAWrapper(model)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, full_masks, _, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            if use_tta:
                outputs = tta_wrapper.predict_with_tta(images, device)
            else:
                outputs = F.softmax(model(images.to(device)), dim=1)
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            targets = full_masks.numpy().flatten()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    iou = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)
    
    try:
        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
    except:
        f1 = 0.0
    
    return accuracy, iou, f1, all_preds, all_targets


# ============================================================
# VISUALIZATION
# ============================================================

def plot_experiment_results(results, experiment_name, save_dir):
    """Plot comprehensive experiment results"""
    ensure_dir(save_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = [r['config'] for r in results]
    best_accs = [r['best_accuracy'] for r in results]
    final_accs = [r['final_accuracy_tta'] for r in results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax = axes[0, 0]
    ax.bar(x - width/2, best_accs, width, label='Best Acc (no TTA)', alpha=0.8)
    ax.bar(x + width/2, final_accs, width, label='Final Acc (with TTA)', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{experiment_name}: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    ious = [r['final_iou_tta'] for r in results]
    ax.plot(configs, ious, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('IoU Performance', fontsize=14, fontweight='bold')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for result in results:
        label = result['config']
        if 'history' in result and 'val_acc' in result['history']:
            ax.plot(result['history']['val_acc'], label=label, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    f1_scores = [r.get('final_f1_tta', 0) for r in results]
    ax.bar(configs, f1_scores, alpha=0.8, color='orange')
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f'{experiment_name}_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {save_dir}/{experiment_name}_results.png")


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {save_path}")


def visualize_predictions(model, dataset, device, num_samples=6, save_path=None):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, full_mask, point_mask, point_loc = dataset[idx]
            
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            img_np = img.permute(1, 2, 0).cpu().numpy()
            full_mask_np = full_mask.cpu().numpy()
            point_loc_np = point_loc.cpu().numpy()
            
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(full_mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(img_np)
            y_coords, x_coords = np.where(point_loc_np == 1)
            axes[i, 2].scatter(x_coords, y_coords, c='red', s=10, alpha=0.6)
            axes[i, 2].set_title(f'Points (n={len(x_coords)})')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred, cmap='gray')
            axes[i, 3].set_title('Prediction')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved predictions: {save_path}")
    plt.close()

# ---------------- Additional visualization helpers ----------------

def plot_training_history(history, out_dir, prefix='training'):
    """
    history: dict with keys 'train_loss', 'val_acc', 'val_iou', 'val_f1' (lists per epoch)
    """
    ensure_dir(out_dir)
    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    plt.figure(figsize=(10, 4))
    # Loss
    plt.subplot(1, 3, 1)
    if 'train_loss' in history:
        plt.plot(epochs, history['train_loss'], '-o', label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True)

    # Val Accuracy
    plt.subplot(1, 3, 2)
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], '-o', label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.grid(True)

    # Val IoU
    plt.subplot(1, 3, 3)
    if 'val_iou' in history:
        plt.plot(epochs, history['val_iou'], '-o', label='Val IoU')
    plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.title('Validation IoU'); plt.grid(True)

    plt.tight_layout()
    path = Path(out_dir) / f'{prefix}_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history: {path}")

def compute_per_class_iou(y_true, y_pred, num_classes):
    """Return per-class IoU array of length num_classes"""
    ious = []
    for c in range(num_classes):
        tp = np.logical_and(y_pred == c, y_true == c).sum()
        fn = np.logical_and(y_pred != c, y_true == c).sum()
        fp = np.logical_and(y_pred == c, y_true != c).sum()
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)
    return np.array(ious)

def plot_per_class_iou(y_true, y_pred, num_classes, out_path, class_names=None):
    """Plot per-class IoU bar chart and save to out_path"""
    ensure_dir(Path(out_path).parent)
    ious = compute_per_class_iou(y_true, y_pred, num_classes)
    labels = [f'Class {i}' for i in range(num_classes)] if class_names is None else class_names

    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=ious)
    plt.ylim(0, 1)
    plt.ylabel('IoU'); plt.title('Per-class IoU')
    plt.xticks(rotation=45, ha='right')
    for idx, v in enumerate(ious):
        plt.text(idx, v + 0.02, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class IoU: {out_path}")

def overlay_mask_on_image(img_np, mask_np, alpha=0.45, cmap='tab20'):
    """
    img_np: HxWx3 (float 0..1)
    mask_np: HxW with class ints
    return: HxWx3 overlay float
    """
    import matplotlib.cm as cm
    h, w = mask_np.shape
    cmap_inst = cm.get_cmap(cmap)
    mask_colored = cmap_inst((mask_np % cmap_inst.N) / (cmap_inst.N - 1))[:, :, :3]  # HxWx3
    overlay = img_np * (1 - alpha) + mask_colored * alpha
    overlay = np.clip(overlay, 0, 1)
    return overlay

def save_overlay_grid(model, dataset, device, out_path, num_samples=6, num_cols=3, use_tta=False):
    """
    Saves a grid of (image | ground-truth | pred | overlay) for random samples.
    """
    ensure_dir(Path(out_path).parent)
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    rows = []
    with torch.no_grad():
        for idx in indices:
            img, full_mask, point_mask, point_loc = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)
            if use_tta:
                probs = tta_predict_simple(model, img_batch, device)
            else:
                probs = F.softmax(model(img_batch), dim=1)
            pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
            img_np = img.permute(1, 2, 0).cpu().numpy()
            gt = full_mask.numpy()
            overlay = overlay_mask_on_image(img_np, pred)

            # Compose row: input, GT, pred, overlay
            rows.append((img_np, gt, pred, overlay))

    # plot grid
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(4*4, 3*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (img_np, gt, pred, overlay) in enumerate(rows):
        axes[i, 0].imshow(img_np); axes[i, 0].set_title('Input'); axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='gray'); axes[i, 1].set_title('GT'); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred, cmap='gray'); axes[i, 2].set_title('Pred'); axes[i, 2].axis('off')
        axes[i, 3].imshow(overlay); axes[i, 3].set_title('Overlay'); axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved overlay grid: {out_path}")


# ============================================================
# TASK 3: EXPERIMENTS
# ============================================================

def experiment_1_point_density(args, device):
    """EXPERIMENT 1: Effect of Point Annotation Density"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Effect of Point Annotation Density")
    print("="*70)
    print("\nPurpose: Determine optimal annotation density for point supervision")
    print("Hypothesis: Logarithmic scaling with diminishing returns\n")
    
    point_counts = [50, 200, 500]
    results = []
    
    for num_points in point_counts:
        print(f"\n{'='*70}")
        print(f"Configuration: {num_points} points per image")
        print(f"Coverage: {(num_points / (args.img_size ** 2)) * 100:.3f}%")
        print(f"{'='*70}\n")
        
        set_seed(args.seed)
        
        train_ds = MassachusettsDataset(
            args.data_root, 'train', args.img_size, num_points, 
            augment=True, seed=args.seed
        )
        val_ds = MassachusettsDataset(
            args.data_root, 'val', args.img_size, num_points,
            augment=False, seed=args.seed + 1000
        )
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)
        
        model = build_model(args.backbone, args.pretrained, args.num_classes,
                          args.freeze_encoder).to(device)
        
        sup_criterion = PartialCrossEntropyLoss(ignore_index=-1, focal_gamma=2.0)
        cons_criterion = ConsistencyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        best_acc = 0
        history = {'train_loss': [], 'val_acc': [], 'val_iou': [], 'val_f1': []}
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            sup_loss, pseudo_loss, cons_loss = train_semi_supervised_epoch(
                model, train_loader, optimizer, device, sup_criterion,
                cons_criterion, args.semi_lambda, args.pseudo_thresh,
                args.consistency_weight
            )
            
            val_acc, val_iou, val_f1, _, _ = evaluate_model(
                model, val_loader, device, use_tta=False
            )
            
            scheduler.step(val_acc)
            
            history['train_loss'].append(sup_loss)
            history['val_acc'].append(val_acc)
            history['val_iou'].append(val_iou)
            history['val_f1'].append(val_f1)
            
            print(f"  Loss: {sup_loss:.4f} (Pseudo: {pseudo_loss:.4f}, Cons: {cons_loss:.4f})")
            print(f"  Val - Acc: {val_acc:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = Path(args.out_dir) / 'checkpoints' / f'exp1_best_{num_points}pts.pth'
                torch.save(model.state_dict(), ckpt_path)
        
        print("\nFinal evaluation with TTA...")
        acc_tta, iou_tta, f1_tta, _, _ = evaluate_model(
            model, val_loader, device, use_tta=True
        )
        
        result = {
            'config': f'{num_points}pts',
            'num_points': num_points,
            'coverage_percent': (num_points / (args.img_size ** 2)) * 100,
            'best_accuracy': best_acc,
            'final_accuracy_tta': acc_tta,
            'final_iou_tta': iou_tta,
            'final_f1_tta': f1_tta,
            'history': history
        }
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Results for {num_points} points:")
        print(f"  Best Acc: {best_acc:.4f}")
        print(f"  With TTA - Acc: {acc_tta:.4f}, IoU: {iou_tta:.4f}, F1: {f1_tta:.4f}")
        print(f"{'='*70}")
    
    return results


def experiment_2_learning_rate(args, device):
    """EXPERIMENT 2: Effect of Learning Rate"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Effect of Learning Rate")
    print("="*70)
    print("\nPurpose: Find optimal LR for sparse supervision")
    print("Hypothesis: LR=0.001 provides best balance\n")
    
    learning_rates = [0.0001, 0.001, 0.01]
    results = []
    
    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"Configuration: Learning rate = {lr}")
        print(f"{'='*70}\n")
        
        set_seed(args.seed)
        
        train_ds = MassachusettsDataset(
            args.data_root, 'train', args.img_size, 200,
            augment=True, seed=args.seed
        )
        val_ds = MassachusettsDataset(
            args.data_root, 'val', args.img_size, 200,
            augment=False, seed=args.seed + 1000
        )
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)
        
        model = build_model(args.backbone, args.pretrained, args.num_classes,
                          args.freeze_encoder).to(device)
        
        sup_criterion = PartialCrossEntropyLoss(ignore_index=-1, focal_gamma=2.0)
        cons_criterion = ConsistencyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        best_acc = 0
        history = {'train_loss': [], 'val_acc': [], 'val_iou': [], 'val_f1': []}
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            sup_loss, pseudo_loss, cons_loss = train_semi_supervised_epoch(
                model, train_loader, optimizer, device, sup_criterion,
                cons_criterion, args.semi_lambda, args.pseudo_thresh,
                args.consistency_weight
            )
            
            val_acc, val_iou, val_f1, _, _ = evaluate_model(
                model, val_loader, device, use_tta=False
            )
            
            scheduler.step(val_acc)
            
            history['train_loss'].append(sup_loss)
            history['val_acc'].append(val_acc)
            history['val_iou'].append(val_iou)
            history['val_f1'].append(val_f1)
            
            print(f"  Loss: {sup_loss:.4f}")
            print(f"  Val - Acc: {val_acc:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = Path(args.out_dir) / 'checkpoints' / f'exp2_best_lr{lr}.pth'
                torch.save(model.state_dict(), ckpt_path)
        
        print("\nFinal evaluation with TTA...")
        acc_tta, iou_tta, f1_tta, _, _ = evaluate_model(
            model, val_loader, device, use_tta=True
        )
        
        result = {
            'config': f'LR={lr}',
            'learning_rate': lr,
            'best_accuracy': best_acc,
            'final_accuracy_tta': acc_tta,
            'final_iou_tta': iou_tta,
            'final_f1_tta': f1_tta,
            'history': history
        }
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Results for LR={lr}:")
        print(f"  Best Acc: {best_acc:.4f}")
        print(f"  With TTA - Acc: {acc_tta:.4f}, IoU: {iou_tta:.4f}, F1: {f1_tta:.4f}")
        print(f"{'='*70}")
    
    return results


# ============================================================
# SINGLE TRAINING MODE
# ============================================================

def train_single_model(args, device):
    """Train a single model with specified configuration"""
    print("\n" + "="*70)
    print("Training Single Model")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Points per image: {args.points_per_image}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Semi-supervised lambda: {args.semi_lambda}\n")
    
    set_seed(args.seed)
    
    train_ds = MassachusettsDataset(
        args.data_root, 'train', args.img_size, args.points_per_image,
        augment=True, seed=args.seed
    )
    val_ds = MassachusettsDataset(
        args.data_root, 'val', args.img_size, args.points_per_image,
        augment=False, seed=args.seed + 1000
    )
    test_ds = MassachusettsDataset(
        args.data_root, 'test', args.img_size, args.points_per_image,
        augment=False, seed=args.seed + 2000
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    
    model = build_model(args.backbone, args.pretrained, args.num_classes,
                      args.freeze_encoder).to(device)
    
    sup_criterion = PartialCrossEntropyLoss(ignore_index=-1, focal_gamma=2.0)
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    best_val_iou = 0
    history = defaultdict(list)
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        sup_loss, pseudo_loss, cons_loss = train_semi_supervised_epoch(
            model, train_loader, optimizer, device, sup_criterion,
            cons_criterion, args.semi_lambda, args.pseudo_thresh,
            args.consistency_weight
        )
        
        val_acc, val_iou, val_f1, _, _ = evaluate_model(
            model, val_loader, device, use_tta=args.tta
        )
        
        scheduler.step(val_iou)
        
        history['train_loss'].append(sup_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)
        
        print(f"  Train Loss: {sup_loss:.4f} (Pseudo: {pseudo_loss:.4f}, Cons: {cons_loss:.4f})")
        print(f"  Val - Acc: {val_acc:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = Path(args.out_dir) / 'checkpoints' / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'args': vars(args)
            }, best_path)
            print(f"  ✓ Saved best model (IoU: {val_iou:.4f})")
    
        # --- produce training curves and a small dataset sample ---
    train_curves_out = Path(args.out_dir) / 'figures' / 'training_curves.png'
    plot_training_history(history, Path(args.out_dir) / 'figures', prefix='training')

    # If you want a dataset sample grid of training images + points:
    sample_viz_path = Path(args.out_dir) / 'figures' / 'dataset_samples.png'
    try:
        # reuse existing visualize_predictions or create a small sample grid:
        visualize_predictions(model, train_ds, device, num_samples=6, save_path=sample_viz_path)
    except Exception as e:
        print(f"Could not create dataset sample visualization: {e}")

    print("\n" + "="*70)
    print("Final Test Evaluation")
    print("="*70)
    
    best_path = Path(args.out_dir) / 'checkpoints' / 'best_model.pth'
    if best_path.exists():
        # Try safe (weights-only) load first (PyTorch 2.6+ default).
        # If it fails (UnpicklingError, AttributeError, RuntimeError), fall back to
        # weights_only=False for backwards compatibility (ONLY for trusted checkpoints).
        try:
            checkpoint = torch.load(best_path)  # weights_only=True by default in recent PyTorch
            print(f"Loaded checkpoint (safe) from {best_path}")
        except (RuntimeError, pickle.UnpicklingError, AttributeError) as e:
            print(f"Safe load failed: {e}\nFalling back to torch.load(..., weights_only=False).")
            # WARNING: this will unpickle arbitrary objects from the file. Only do if you trust file.
            checkpoint = torch.load(best_path, weights_only=False)

        # Now handle either a dict checkpoint or a pure state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # This covers the case you saved state_dict() but wrapped in a dict-like object
            model.load_state_dict(checkpoint)
        else:
            # Try treating the loaded object as a plain state_dict
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"ERROR: Failed to load model weights from checkpoint: {e}")
                raise
    else:
        print(f"⚠️ Warning: checkpoint not found at {best_path}")


    
    test_acc, test_iou, test_f1, preds, targets = evaluate_model(
        model, test_loader, device, use_tta=True
    )
        # Per-class IoU bar chart
    per_class_path = Path(args.out_dir) / 'figures' / 'per_class_iou.png'
    plot_per_class_iou(targets, preds, args.num_classes, per_class_path)

    # Overlay grid (show predictions + overlays)
    overlay_path = Path(args.out_dir) / 'figures' / 'overlay_grid.png'
    save_overlay_grid(model, test_ds, device, overlay_path, num_samples=6, use_tta=True)

    
    print(f"\nTest Results (with TTA):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  IoU: {test_iou:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    
    cm_path = Path(args.out_dir) / 'figures' / 'confusion_matrix.png'
    plot_confusion_matrix(targets, preds, args.num_classes, cm_path)
    
    viz_path = Path(args.out_dir) / 'figures' / 'predictions.png'
    visualize_predictions(model, test_ds, device, num_samples=6, save_path=viz_path)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'best_val_iou': float(best_val_iou),
        'test_accuracy': float(test_acc),
        'test_iou': float(test_iou),
        'test_f1': float(test_f1),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    with open(Path(args.out_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.out_dir}/results.json")


# ============================================================
# INFERENCE MODE
# ============================================================

def run_inference(args, device):
    """Run inference with ensemble of models"""
    print("\n" + "="*70)
    print("Inference Mode")
    print("="*70)
    print(f"\nLoading {len(args.ckpt_paths)} model(s)...")
    
    models = []
    for ckpt_path in args.ckpt_paths:
        model = build_model(args.backbone, pretrained=False, 
                          num_classes=args.num_classes, freeze_encoder=False)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device).eval()
        models.append(model)
        print(f"  ✓ Loaded: {ckpt_path}")
    
    ensemble = EnsembleModel(models) if len(models) > 1 else None
    
    test_ds = MassachusettsDataset(
        args.data_root, 'test', args.img_size, args.points_per_image,
        augment=False, seed=args.seed
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    
    print(f"\nRunning inference (TTA={'Yes' if args.tta else 'No'}, "
          f"Ensemble={'Yes' if len(models) > 1 else 'No'})...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, full_masks, _, _ in tqdm(test_loader, desc="Inference"):
            if ensemble:
                probs = ensemble.predict(images, device, use_tta=args.tta)
            else:
                if args.tta:
                    probs = tta_predict_simple(models[0], images, device)
                else:
                    probs = F.softmax(models[0](images.to(device)), dim=1)
            
            preds = torch.argmax(probs, dim=1).cpu().numpy().flatten()
            targets = full_masks.numpy().flatten()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    iou = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)
    try:
        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
    except:
        f1 = 0.0
    
    print(f"\n{'='*70}")
    print("Inference Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  IoU: {iou:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"{'='*70}")
    
    cm_path = Path(args.out_dir) / 'figures' / 'inference_confusion_matrix.png'
    plot_confusion_matrix(all_targets, all_preds, args.num_classes, cm_path)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_models': len(models),
        'tta_enabled': args.tta,
        'accuracy': float(accuracy),
        'iou': float(iou),
        'f1': float(f1)
    }
    
    with open(Path(args.out_dir) / 'inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.out_dir}/inference_results.json")


# ============================================================
# MAIN FUNCTION
# ============================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Point-Supervised Segmentation for Massachusetts Buildings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'experiments', 'inference'],
                       help='Execution mode')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to prepared dataset root')
    parser.add_argument('--out_dir', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size (square)')
    parser.add_argument('--points_per_image', type=int, default=200,
                       help='Number of point annotations per image')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    
    parser.add_argument('--backbone', type=str, default='resnet34',
                       help='Encoder backbone')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                       help='Do not use pretrained weights')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='Freeze encoder for transfer learning')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    parser.add_argument('--semi_lambda', type=float, default=0.5,
                       help='Weight for pseudo-label loss')
    parser.add_argument('--pseudo_thresh', type=float, default=0.95,
                       help='Confidence threshold for pseudo-labels')
    parser.add_argument('--consistency_weight', type=float, default=0.1,
                       help='Weight for consistency loss')
    
    parser.add_argument('--tta', action='store_true', default=False,
                       help='Use test-time augmentation')
    parser.add_argument('--ckpt_paths', nargs='+', default=[],
                       help='Checkpoint paths for inference/ensemble')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--hist_ref', type=str, default=None,
                       help='Reference image for histogram matching')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("POINT-SUPERVISED SEMANTIC SEGMENTATION")
    print("Massachusetts Buildings Dataset")
    print("="*70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: {args.mode}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.out_dir}")
    
    ensure_dir(args.out_dir)
    ensure_dir(Path(args.out_dir) / 'checkpoints')
    ensure_dir(Path(args.out_dir) / 'figures')
    
    if args.mode == 'train':
        train_single_model(args, device)
    
    elif args.mode == 'experiments':
        print("\n" + "="*70)
        print("RUNNING ALL EXPERIMENTS")
        print("="*70)
        
        exp1_results = experiment_1_point_density(args, device)
        
        exp1_path = Path(args.out_dir) / 'experiment1_results.json'
        with open(exp1_path, 'w') as f:
            json.dump({
                'experiment': 'Point Annotation Density',
                'timestamp': datetime.now().isoformat(),
                'results': exp1_results
            }, f, indent=2)
        print(f"\n✓ Experiment 1 results saved to {exp1_path}")
        
        plot_experiment_results(exp1_results, 'Experiment1_PointDensity',
                              Path(args.out_dir) / 'figures')
        
        exp2_results = experiment_2_learning_rate(args, device)
        
        exp2_path = Path(args.out_dir) / 'experiment2_results.json'
        with open(exp2_path, 'w') as f:
            json.dump({
                'experiment': 'Learning Rate Analysis',
                'timestamp': datetime.now().isoformat(),
                'results': exp2_results
            }, f, indent=2)
        print(f"\n✓ Experiment 2 results saved to {exp2_path}")
        
        plot_experiment_results(exp2_results, 'Experiment2_LearningRate',
                              Path(args.out_dir) / 'figures')
        
        print("\n" + "="*70)
        print("EXPERIMENTS SUMMARY")
        print("="*70)
        
        print("\n1. EXPERIMENT 1: Point Annotation Density")
        print("-" * 70)
        print(f"{'Points':<10} {'Coverage':<12} {'Best Acc':<12} {'TTA Acc':<12} {'IoU':<12}")
        print("-" * 70)
        for r in exp1_results:
            print(f"{r['num_points']:<10} {r['coverage_percent']:<12.3f}% "
                  f"{r['best_accuracy']:<12.4f} {r['final_accuracy_tta']:<12.4f} "
                  f"{r['final_iou_tta']:<12.4f}")
        
        print("\n2. EXPERIMENT 2: Learning Rate Analysis")
        print("-" * 70)
        print(f"{'LR':<12} {'Best Acc':<12} {'TTA Acc':<12} {'IoU':<12}")
        print("-" * 70)
        for r in exp2_results:
            print(f"{r['learning_rate']:<12} {r['best_accuracy']:<12.4f} "
                  f"{r['final_accuracy_tta']:<12.4f} {r['final_iou_tta']:<12.4f}")
        
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        best_exp1 = max(exp1_results, key=lambda x: x['final_accuracy_tta'])
        best_exp2 = max(exp2_results, key=lambda x: x['final_accuracy_tta'])
        
        print(f"\n1. Optimal Point Density: {best_exp1['num_points']} points")
        print(f"   - Achieves {best_exp1['final_accuracy_tta']:.4f} accuracy")
        print(f"   - Coverage: {best_exp1['coverage_percent']:.3f}% of pixels")
        print(f"   - IoU: {best_exp1['final_iou_tta']:.4f}")
        
        print(f"\n2. Optimal Learning Rate: {best_exp2['learning_rate']}")
        print(f"   - Achieves {best_exp2['final_accuracy_tta']:.4f} accuracy")
        print(f"   - IoU: {best_exp2['final_iou_tta']:.4f}")
        
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
    
    elif args.mode == 'inference':
        if not args.ckpt_paths:
            print("ERROR: --ckpt_paths required for inference mode")
            sys.exit(1)
        run_inference(args, device)
    
    print(f"\n✓ All outputs saved to: {args.out_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()










"""
# Point-Supervised Semantic Segmentation - Main Script
# Comprehensive implementation for Massachusetts Buildings dataset
# Features:
#  - Partial Cross-Entropy Loss (point supervision)
#  - Transfer learning with segmentation_models_pytorch (pretrained encoder)
#  - Option to freeze encoder (transfer learning)
#  - Semi-supervised pseudo-labeling + consistency loss
#  - Test-time augmentation (TTA)
#  - Ensemble predictions (average probabilities)
#  - Two experiments: point density & learning rate
#  - Visualization and result saving


import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# segmentation_models_pytorch required
try:
    import segmentation_models_pytorch as smp
except Exception as e:
    print("ERROR: segmentation_models_pytorch not installed. Install with:")
    print("  pip install segmentation-models-pytorch")
    raise e

# optional histogram matching
try:
    from skimage.exposure import match_histograms
    HAS_HIST = True
except Exception:
    HAS_HIST = False

# sklearn metrics
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix, precision_recall_fscore_support

# -------------------------
# Utility helpers
# -------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pil_loader(path):
    return Image.open(str(path)).convert('RGB')

def pil_loader_mask(path):
    im = Image.open(str(path))
    if im.mode != 'L':
        im = im.convert('L')
    return im

# -------------------------
# Partial Cross-Entropy Loss
# -------------------------
class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, focal_gamma=0.0, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_gamma = float(focal_gamma)
        self.reduction = reduction

    def forward(self, predictions, targets, mask=None):
        # predictions: (B,C,H,W) logits
        # targets: (B,H,W) with class indices or ignore_index
        device = predictions.device
        B, C, H, W = predictions.shape

        log_probs = F.log_softmax(predictions, dim=1)  # (B,C,H,W)
        probs = F.softmax(predictions, dim=1)

        if mask is None:
            mask = (targets != self.ignore_index).float()

        mask = mask.to(device)
        targets = targets.to(device)

        lp_flat = log_probs.permute(0,2,3,1).contiguous().view(-1, C)
        p_flat = probs.permute(0,2,3,1).contiguous().view(-1, C)
        t_flat = targets.view(-1)
        m_flat = mask.view(-1)

        valid_mask = (t_flat != self.ignore_index).float()
        t_safe = t_flat.clone()
        t_safe[t_flat == self.ignore_index] = 0

        lp_true = lp_flat.gather(1, t_safe.unsqueeze(1)).squeeze(1)
        p_true = p_flat.gather(1, t_safe.unsqueeze(1)).squeeze(1)

        if self.focal_gamma > 0:
            focal_w = (1 - p_true) ** self.focal_gamma
        else:
            focal_w = 1.0

        loss = -focal_w * lp_true
        masked_loss = loss * m_flat * valid_mask
        num_labeled = (m_flat * valid_mask).sum()

        if self.reduction == 'mean':
            if num_labeled > 0:
                return masked_loss.sum() / num_labeled
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:
            return masked_loss

# -------------------------
# Consistency loss for semi-supervised
# -------------------------
class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred1, pred2, unlabeled_mask):
        # pred*: logits (B,C,H,W)
        p1 = F.softmax(pred1, dim=1)
        p2 = F.softmax(pred2, dim=1)
        mse = (p1 - p2) ** 2
        mse = mse.mean(dim=1)  # (B,H,W)
        unlabeled_mask = unlabeled_mask.float()
        denom = unlabeled_mask.sum().clamp(min=1.0)
        return (mse * unlabeled_mask).sum() / denom

# -------------------------
# Dataset class for Massachusetts
# -------------------------
class MassachusettsDataset(Dataset):
    def __init__(self, root, split='train', img_size=256, num_points=200, augment=False, hist_ref=None, seed=42):
        self.root = Path(root)
        self.split = split
        self.img_size = int(img_size)
        self.num_points = int(num_points)
        self.augment = augment
        self.seed = int(seed)

        self.img_dir = self.root / 'images' / split
        self.mask_dir = self.root / 'masks' / split

        if not self.img_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing dataset folders: {self.img_dir} or {self.mask_dir}")

        self.img_paths = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in ['.png','.jpg','.tif','.tiff','.jpeg']])
        self.mask_paths = sorted([p for p in self.mask_dir.iterdir() if p.suffix.lower() in ['.png','.jpg','.tif','.tiff','.jpeg']])

        # pair by stem if lengths mismatch
        if len(self.img_paths) != len(self.mask_paths):
            mask_map = {p.stem: p for p in self.mask_paths}
            paired_imgs = []
            paired_masks = []
            for imgp in self.img_paths:
                m = mask_map.get(imgp.stem)
                if m:
                    paired_imgs.append(imgp)
                    paired_masks.append(m)
            self.img_paths = paired_imgs
            self.mask_paths = paired_masks

        self.rng = np.random.RandomState(self.seed + (0 if split=='train' else 1000))
        self.hist_ref = None
        if hist_ref and HAS_HIST:
            try:
                ref = pil_loader(hist_ref).resize((self.img_size, self.img_size))
                self.hist_ref = np.array(ref).astype(np.uint8)
            except Exception:
                self.hist_ref = None

        print(f"[Dataset] {split}: {len(self.img_paths)} images, points_per_image={self.num_points}")

    def __len__(self):
        return len(self.img_paths)

    def _hist_match(self, img_np):
        if not HAS_HIST or self.hist_ref is None:
            return img_np
        try:
            matched = match_histograms(img_np, self.hist_ref, channel_axis=2)
            return matched.astype(np.float32)
        except Exception:
            return img_np

    def _sample_points(self, mask_np):
        H,W = mask_np.shape
        point_mask = np.full((H,W), -1, dtype=np.int64)
        point_loc = np.zeros((H,W), dtype=np.float32)

        fg = np.argwhere(mask_np==1)
        bg = np.argwhere(mask_np==0)

        n_fg = min(self.num_points//2, len(fg))
        n_bg = min(self.num_points - n_fg, len(bg))

        if n_fg>0 and len(fg)>0:
            idxs = self.rng.choice(len(fg), n_fg, replace=False)
            for i in idxs:
                y,x = fg[i]
                point_mask[y,x]=1; point_loc[y,x]=1.0

        if n_bg>0 and len(bg)>0:
            idxs = self.rng.choice(len(bg), n_bg, replace=False)
            for i in idxs:
                y,x = bg[i]
                point_mask[y,x]=0; point_loc[y,x]=1.0

        sampled = int(point_loc.sum())
        remaining = self.num_points - sampled
        HxW = H*W
        tries = 0
        while remaining>0 and tries < remaining*5:
            idx = self.rng.randint(0, HxW)
            y,x = divmod(idx, W)
            if point_loc[y,x]==0:
                point_mask[y,x]=int(mask_np[y,x])
                point_loc[y,x]=1.0
                remaining -=1
            tries +=1

        return point_mask, point_loc

    def _load_pair(self, idx):
        img = pil_loader(self.img_paths[idx])
        mask = pil_loader_mask(self.mask_paths[idx])
        img_np = np.array(img).astype(np.float32)/255.0
        mask_np = (np.array(mask) > 127).astype(np.int64)
        if self.hist_ref is not None:
            img_np = self._hist_match(img_np)
        return img_np, mask_np

    def _resize(self, img_np, mask_np):
        if img_np.shape[0] != self.img_size or img_np.shape[1] != self.img_size:
            img_pil = Image.fromarray((img_np*255).astype(np.uint8)).resize((self.img_size,self.img_size), Image.BILINEAR)
            mask_pil = Image.fromarray((mask_np*255).astype(np.uint8)).resize((self.img_size,self.img_size), Image.NEAREST)
            img_np = np.array(img_pil).astype(np.float32)/255.0
            mask_np = (np.array(mask_pil)>127).astype(np.int64)
        return img_np, mask_np

    def _augment(self, img_np, mask_np, point_mask, point_loc):
        if self.rng.rand() > 0.5:
            img_np = np.fliplr(img_np).copy()
            mask_np = np.fliplr(mask_np).copy()
            point_mask = np.fliplr(point_mask).copy()
            point_loc = np.fliplr(point_loc).copy()
        if self.rng.rand() > 0.5:
            img_np = np.flipud(img_np).copy()
            mask_np = np.flipud(mask_np).copy()
            point_mask = np.flipud(point_mask).copy()
            point_loc = np.flipud(point_loc).copy()
        k = self.rng.randint(0,4)
        if k>0:
            img_np = np.rot90(img_np, k).copy()
            mask_np = np.rot90(mask_np, k).copy()
            point_mask = np.rot90(point_mask, k).copy()
            point_loc = np.rot90(point_loc, k).copy()
        return img_np, mask_np, point_mask, point_loc

    def __getitem__(self, idx):
        img_np, mask_np = self._load_pair(idx)
        img_np, mask_np = self._resize(img_np, mask_np)
        point_mask, point_loc = self._sample_points(mask_np)

        if self.augment:
            img_np, mask_np, point_mask, point_loc = self._augment(img_np, mask_np, point_mask, point_loc)

        img_t = torch.from_numpy(img_np.transpose(2,0,1)).float()
        full_mask = torch.from_numpy(mask_np).long()
        point_mask_t = torch.from_numpy(point_mask).long()
        point_loc_t = torch.from_numpy(point_loc).float()
        return img_t, full_mask, point_mask_t, point_loc_t

# -------------------------
# Build model with transfer learning
# -------------------------
def build_model(backbone='resnet34', pretrained=True, num_classes=2, freeze_encoder=False):
    encoder_weights = 'imagenet' if pretrained else None
    model = smp.Unet(encoder_name=backbone, encoder_weights=encoder_weights, in_channels=3, classes=num_classes, activation=None)
    if freeze_encoder and pretrained:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"[Model] Encoder '{backbone}' frozen")
    return model

# -------------------------
# Ensemble wrapper
# -------------------------
class EnsembleModel:
    def __init__(self, models):
        self.models = models
        for m in self.models:
            m.eval()
    def predict(self, images, device, use_tta=False):
        # images: (B,3,H,W) tensor on CPU usually; move to device inside
        all_probs = []
        with torch.no_grad():
            for m in self.models:
                m.to(device)
                m.eval()
                if use_tta:
                    probs = tta_predict_simple(m, images, device)
                else:
                    imgs = images.to(device)
                    probs = F.softmax(m(imgs), dim=1)
                all_probs.append(probs)
        avg = torch.stack(all_probs).mean(dim=0)
        return avg

# -------------------------
# TTA helpers
# -------------------------
class TTAWrapper:
    def __init__(self, model):
        self.model = model
    def predict_with_tta(self, images, device):
        with torch.no_grad():
            self.model.eval()
            imgs = images.to(device)
            preds = []
            # original
            preds.append(F.softmax(self.model(imgs), dim=1))
            # horiz
            p = F.softmax(self.model(torch.flip(imgs, dims=[3])), dim=1)
            preds.append(torch.flip(p, dims=[3]))
            # vert
            p = F.softmax(self.model(torch.flip(imgs, dims=[2])), dim=1)
            preds.append(torch.flip(p, dims=[2]))
            # rot90
            p = F.softmax(self.model(torch.rot90(imgs,1,[2,3])), dim=1)
            preds.append(torch.rot90(p,3,[2,3]))
            # rot180
            p = F.softmax(self.model(torch.rot90(imgs,2,[2,3])), dim=1)
            preds.append(torch.rot90(p,2,[2,3]))
            # average
            avg = torch.stack(preds).mean(dim=0)
            return avg

def tta_predict_simple(model, images, device):
    with torch.no_grad():
        imgs = images.to(device)
        out1 = F.softmax(model(imgs), dim=1)
        out2 = F.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
        out2 = torch.flip(out2, dims=[3])
        return (out1 + out2) / 2.0

# -------------------------
# Evaluation (works for model or ensemble)
# -------------------------
def evaluate_model(model_or_ensemble, dataloader, device, num_classes=2, use_tta=False, out_dir='results'):
    model_is_ensemble = isinstance(model_or_ensemble, EnsembleModel) or isinstance(model_or_ensemble, list)
    if isinstance(model_or_ensemble, EnsembleModel):
        ensemble = model_or_ensemble
        single_model = None
    elif isinstance(model_or_ensemble, list):
        ensemble = EnsembleModel(model_or_ensemble)
        single_model = None
    else:
        ensemble = None
        single_model = model_or_ensemble

    tta_wrapper = None
    if use_tta and single_model is not None:
        tta_wrapper = TTAWrapper(single_model)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, full_masks, _, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            if ensemble is not None:
                probs = ensemble.predict(imgs, device, use_tta=use_tta)
            else:
                if use_tta:
                    probs = tta_wrapper.predict_with_tta(imgs, device)
                else:
                    probs = F.softmax(single_model(imgs.to(device)), dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy().flatten()
            targets = full_masks.numpy().flatten()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = float(accuracy_score(all_targets, all_preds))
    try:
        iou = float(jaccard_score(all_targets, all_preds, average='macro', zero_division=0))
    except Exception:
        iou = 0.0
    try:
        _, _, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        f1 = float(f1)
    except Exception:
        f1 = 0.0

    # Save confusion matrix
    cm_path = Path(out_dir) / 'figures' / 'confusion_matrix.png'
    ensure_dir(Path(out_dir)/'figures')
    plot_confusion_matrix(all_targets, all_preds, num_classes, cm_path)

    results = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'iou': iou,
        'f1': f1,
        'num_samples': int(len(all_targets))
    }
    # Save small summary
    with open(Path(out_dir)/'inference_summary.json','w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Eval] Acc={accuracy:.4f} IoU={iou:.4f} F1={f1:.4f}")
    return accuracy, iou, f1, all_preds, all_targets

# -------------------------
# Plotting helpers
# -------------------------
def plot_confusion_matrix(y_true, y_pred, num_classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f"{i}" for i in range(num_classes)],
                yticklabels=[f"{i}" for i in range(num_classes)])
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] saved: {save_path}")

def visualize_predictions(model, dataset, device, save_path):
    ensure_dir(Path(save_path).parent)
    model.eval()
    n = min(6, len(dataset))
    idxs = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(n,4, figsize=(16,4*n))
    if n==1:
        axes = axes.reshape(1,4)
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            img, full_mask, point_mask, point_loc = dataset[idx]
            out = model(img.unsqueeze(0).to(device))
            pred = torch.argmax(F.softmax(out,dim=1), dim=1).squeeze(0).cpu().numpy()
            img_np = img.permute(1,2,0).numpy()
            axes[i,0].imshow(img_np); axes[i,0].set_title('Image'); axes[i,0].axis('off')
            axes[i,1].imshow(full_mask.numpy(), cmap='gray'); axes[i,1].set_title('GT'); axes[i,1].axis('off')
            axes[i,2].imshow(img_np); yc, xc = np.where(point_loc.numpy()==1); axes[i,2].scatter(xc,yc,c='r',s=10); axes[i,2].set_title('Points'); axes[i,2].axis('off')
            axes[i,3].imshow(pred, cmap='gray'); axes[i,3].set_title('Pred'); axes[i,3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] predictions saved to {save_path}")

# -------------------------
# Training loops
# -------------------------
def train_semi_supervised_epoch(model, dataloader, optimizer, device,
                                sup_criterion, cons_criterion,
                                semi_lambda=0.5, pseudo_thresh=0.95, consistency_weight=0.1):
    model.train()
    total_sup = 0.0
    total_pseudo = 0.0
    total_cons = 0.0
    n = 0
    for imgs, _, point_masks, point_locs in tqdm(dataloader, desc="Train", leave=False):
        imgs = imgs.to(device)
        point_masks = point_masks.to(device)
        point_locs = point_locs.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # logits

        labeled_mask = (point_masks != -1).float()
        sup_loss = sup_criterion(outputs, point_masks, labeled_mask)

        pseudo_loss = torch.tensor(0.0, device=device)
        if semi_lambda > 0:
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                maxp, pseudo_labels = torch.max(probs, dim=1)
            unlabeled_mask = (point_masks == -1)
            reliable_mask = (maxp > pseudo_thresh) & unlabeled_mask
            if reliable_mask.sum() > 0:
                pseudo_loss = sup_criterion(outputs, pseudo_labels, reliable_mask.float())

        cons_loss = torch.tensor(0.0, device=device)
        if cons_criterion is not None and consistency_weight > 0:
            # second forward with dropout/augmentation (model in train keeps dropout)
            outputs2 = model(imgs)
            unlabeled_mask_float = (point_masks == -1).float()
            cons_loss = cons_criterion(outputs, outputs2, unlabeled_mask_float)

        loss = sup_loss + semi_lambda * pseudo_loss + consistency_weight * cons_loss
        loss.backward()
        optimizer.step()

        total_sup += float(sup_loss.item())
        total_pseudo += float(pseudo_loss.item()) if isinstance(pseudo_loss, torch.Tensor) else 0.0
        total_cons += float(cons_loss.item()) if isinstance(cons_loss, torch.Tensor) else 0.0
        n += 1
    if n==0:
        return 0.0, 0.0, 0.0
    return total_sup / n, total_pseudo / n, total_cons / n

# -------------------------
# Experiments
# -------------------------
def experiment_1_point_density(args, device):
    point_counts = [50,100,200,500]
    results = []
    for num_points in point_counts:
        print(f"\n[Exp1] Training with {num_points} points per image")
        set_seed(args.seed)
        train_ds = MassachusettsDataset(args.data_root, 'train', args.img_size, num_points, augment=True, seed=args.seed)
        val_ds = MassachusettsDataset(args.data_root, 'val', args.img_size, num_points, augment=False, seed=args.seed+1)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = build_model(args.backbone, args.pretrained, args.num_classes, args.freeze_encoder).to(device)
        sup_criterion = PartialCrossEntropyLoss(ignore_index=-1, focal_gamma=2.0)
        cons_criterion = ConsistencyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        best_val = 0.0
        history = defaultdict(list)
        for epoch in range(args.epochs):
            sup_loss, pseudo_loss, cons_loss = train_semi_supervised_epoch(model, train_loader, optimizer, device, sup_criterion, cons_criterion, args.semi_lambda, args.pseudo_thresh, args.consistency_weight)
            val_acc, val_iou, val_f1, _, _ = evaluate_model(model, val_loader, device, num_classes=args.num_classes, use_tta=False, out_dir=args.out_dir)
            scheduler.step(val_acc)
            history['train_loss'].append(sup_loss); history['val_acc'].append(val_acc); history['val_iou'].append(val_iou)
            print(f"Epoch {epoch+1}/{args.epochs} sup={sup_loss:.4f} pseudo={pseudo_loss:.4f} cons={cons_loss:.4f} val_acc={val_acc:.4f} val_iou={val_iou:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                ckpt = Path(args.out_dir)/'checkpoints'/f'exp1_best_{num_points}pts.pth'; ensure_dir(ckpt.parent)
                torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, ckpt)
        # final eval with TTA
        acc_tta, iou_tta, f1_tta, _, _ = evaluate_model(model, val_loader, device, num_classes=args.num_classes, use_tta=True, out_dir=args.out_dir)
        results.append({'num_points':num_points, 'best_accuracy':best_val, 'final_accuracy_tta':acc_tta, 'final_iou_tta':iou_tta, 'history': history})
    return results

def experiment_2_learning_rate(args, device):
    lrs = [0.0001, 0.0005, 0.001]
    results = []
    for lr in lrs:
        print(f"\n[Exp2] LR={lr}")
        set_seed(args.seed)
        train_ds = MassachusettsDataset(args.data_root,'train',args.img_size,200,augment=True,seed=args.seed)
        val_ds = MassachusettsDataset(args.data_root,'val',args.img_size,200,augment=False,seed=args.seed+1)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = build_model(args.backbone, args.pretrained, args.num_classes, args.freeze_encoder).to(device)
        sup_criterion = PartialCrossEntropyLoss(ignore_index=-1, focal_gamma=2.0)
        cons_criterion = ConsistencyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        best_val = 0.0
        history = defaultdict(list)
        for epoch in range(args.epochs):
            sup_loss, pseudo_loss, cons_loss = train_semi_supervised_epoch(model, train_loader, optimizer, device, sup_criterion, cons_criterion, args.semi_lambda, args.pseudo_thresh, args.consistency_weight)
            val_acc, val_iou, val_f1, _, _ = evaluate_model(model, val_loader, device, num_classes=args.num_classes, use_tta=False, out_dir=args.out_dir)
            scheduler.step(val_acc)
            history['train_loss'].append(sup_loss); history['val_acc'].append(val_acc); history['val_iou'].append(val_iou)
            print(f"Epoch {epoch+1}/{args.epochs} sup={sup_loss:.4f} val_acc={val_acc:.4f} val_iou={val_iou:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                ckpt = Path(args.out_dir)/'checkpoints'/f'exp2_best_lr{lr}.pth'; ensure_dir(ckpt.parent)
                torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, ckpt)
        acc_tta, iou_tta, f1_tta, _, _ = evaluate_model(model, val_loader, device, num_classes=args.num_classes, use_tta=True, out_dir=args.out_dir)
        results.append({'learning_rate': lr, 'best_accuracy': best_val, 'final_accuracy_tta': acc_tta, 'final_iou_tta': iou_tta, 'history': history})
    return results

# -------------------------
# Single training routine
# -------------------------
def train_single_model(args, device):
    print("[Train] Preparing datasets")
    train_ds = MassachusettsDataset(args.data_root, 'train', args.img_size, args.points_per_image, augment=True, seed=args.seed)
    val_ds = MassachusettsDataset(args.data_root, 'val', args.img_size, args.points_per_image, augment=False, seed=args.seed+1)
    test_ds = MassachusettsDataset(args.data_root, 'test', args.img_size, args.points_per_image, augment=False, seed=args.seed+2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.backbone, args.pretrained, args.num_classes, args.freeze_encoder).to(device)
    sup_criterion = PartialCrossEntropyLoss(ignore_index=-1, focal_gamma=2.0)
    cons_criterion = ConsistencyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_iou = 0.0
    history = defaultdict(list)
    for epoch in range(args.epochs):
        print(f"[Train] Epoch {epoch+1}/{args.epochs}")
        sup_loss, pseudo_loss, cons_loss = train_semi_supervised_epoch(model, train_loader, optimizer, device, sup_criterion, cons_criterion, args.semi_lambda, args.pseudo_thresh, args.consistency_weight)
        val_acc, val_iou, val_f1, _, _ = evaluate_model(model, val_loader, device, num_classes=args.num_classes, use_tta=args.tta, out_dir=args.out_dir)
        scheduler.step(val_iou)
        history['train_loss'].append(sup_loss); history['val_acc'].append(val_acc); history['val_iou'].append(val_iou)
        print(f"  sup={sup_loss:.4f} pseudo={pseudo_loss:.4f} cons={cons_loss:.4f} val_acc={val_acc:.4f} val_iou={val_iou:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = Path(args.out_dir)/'checkpoints'/'best_model.pth'; ensure_dir(ckpt_path.parent)
            torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'val_iou': val_iou, 'args': vars(args)}, ckpt_path)
            print(f"[Train] saved best model -> {ckpt_path}")

    # Load best and evaluate on test
    ckpt_path = Path(args.out_dir)/'checkpoints'/'best_model.pth'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    test_acc, test_iou, test_f1, preds, targets = evaluate_model(model, test_loader, device, num_classes=args.num_classes, use_tta=True, out_dir=args.out_dir)
    # Save predictions visualization
    viz_path = Path(args.out_dir)/'figures'/'predictions.png'
    ensure_dir(viz_path.parent)
    visualize_predictions(model, test_ds, device, str(viz_path))

    results = {'test_acc': test_acc, 'test_iou': test_iou, 'test_f1': test_f1, 'history': history, 'timestamp': datetime.now().isoformat()}
    with open(Path(args.out_dir)/'results.json','w') as f:
        json.dump(results, f, indent=2)
    print("[Train] Finished training. Results saved.")

# -------------------------
# Inference (ensemble)
# -------------------------
def run_inference(args, device):
    print("[Infer] Loading checkpoints")
    models = []
    for p in args.ckpt_paths:
        m = build_model(args.backbone, pretrained=False, num_classes=args.num_classes, freeze_encoder=False)
        ckpt = torch.load(p, map_location='cpu')
        if 'model_state_dict' in ckpt:
            m.load_state_dict(ckpt['model_state_dict'])
        else:
            m.load_state_dict(ckpt)
        m.to(device).eval()
        models.append(m)
    ensemble = EnsembleModel(models) if len(models)>1 else None
    model = models[0] if len(models)==1 else None

    test_ds = MassachusettsDataset(args.data_root,'test',args.img_size,args.points_per_image,augment=False,seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if ensemble:
        acc,iou,f1,_,_ = evaluate_model(ensemble, test_loader, device, num_classes=args.num_classes, use_tta=args.tta, out_dir=args.out_dir)
    else:
        acc,iou,f1,_,_ = evaluate_model(model, test_loader, device, num_classes=args.num_classes, use_tta=args.tta, out_dir=args.out_dir)
    print(f"[Infer] Done. Acc={acc:.4f} IoU={iou:.4f} F1={f1:.4f}")

# -------------------------
# CLI and main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train','experiments','inference'])
    # default data_root: look for common prepared folder, fallback to current path
    default_root = Path.cwd()/'massachusetts_dataset'/'prepared'
    parser.add_argument('--data_root', type=str, default=str(default_root), help='prepared dataset root (images/ masks/)')
    parser.add_argument('--out_dir', type=str, default='results', help='output directory')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--points_per_image', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    # semi params
    parser.add_argument('--semi_lambda', type=float, default=0.5)
    parser.add_argument('--pseudo_thresh', type=float, default=0.95)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    # inference / ensemble
    parser.add_argument('--tta', action='store_true', default=True)
    parser.add_argument('--no_tta', action='store_false', dest='tta')
    parser.add_argument('--ckpt_paths', nargs='*', default=[])
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*60)
    print("Point-Supervised Semantic Segmentation")
    print("Using Massachusetts Buildings dataset")
    print("="*60)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Data root: {args.data_root}")
    print(f"Out dir: {args.out_dir}")

    ensure_dir(args.out_dir)
    ensure_dir(Path(args.out_dir)/'checkpoints')
    ensure_dir(Path(args.out_dir)/'figures')

    if args.mode == 'train':
        train_single_model(args, device)
    elif args.mode == 'experiments':
        exp1 = experiment_1_point_density(args, device)
        with open(Path(args.out_dir)/'experiment1.json','w') as f: json.dump(exp1, f, indent=2)
        exp2 = experiment_2_learning_rate(args, device)
        with open(Path(args.out_dir)/'experiment2.json','w') as f: json.dump(exp2, f, indent=2)
        print("[Main] Experiments done. Results saved.")
    elif args.mode == 'inference':
        if not args.ckpt_paths:
            print("Provide --ckpt_paths for inference.")
            return
        run_inference(args, device)
    else:
        print("Unknown mode")

if __name__=='__main__':
    main()
"""








"""
# main.py

# Point-supervised semantic segmentation training pipeline using a real dataset
# (Massachusetts Buildings prepared dataset), transfer learning (segmentation_models_pytorch),
# Partial Cross-Entropy loss for point supervision, simple pseudo-labeling (semi-supervised),
# histogram matching (optional), and TTA for inference.


import os
import sys
import random
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix
import seaborn as sns

# Optional libraries
try:
    import segmentation_models_pytorch as smp
except Exception as e:
    print("ERROR: segmentation_models_pytorch (smp) not found. Install with:")
    print("  pip install segmentation-models-pytorch")
    sys.exit(1)

try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

# --------------------------
# User / dataset config
# --------------------------
DATA_ROOT = "/home/samama/Desktop/ML/massachusetts_dataset/prepared"  # change if needed
RESULTS_ROOT = "results"
os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(os.path.join(RESULTS_ROOT, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_ROOT, "figures"), exist_ok=True)

# Training config (defaults set to run w/out CLI)
CONFIG = {
    "img_size": 256,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "seed": 42,
    "points_per_image": 200,        # number of point annotations to simulate per image
    "num_classes": 2,               # Massachusetts buildings: binary (background / building)
    "backbone": "resnet34",
    "encoder_weights": "imagenet",  # set to None to disable pretrained
    "freeze_encoder": True,         # freeze encoder if using pretrained backbone
    "semi_lambda": 0.3,             # weight for pseudo-label loss (semi-supervised)
    "pseudo_thresh": 0.9,           # threshold for confident pseudo-labeling
    "tta": True,                    # apply simple TTA at inference
    "hist_ref": None,               # optional path to reference image for histogram matching
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# --------------------------
# Partial Cross Entropy Loss
# --------------------------
class PartialCrossEntropyLoss(nn.Module):
    
    # Partial Cross-Entropy Loss:
    #   L = - sum( log(p_true) * m * valid ) / sum(m * valid)
    # Returns a tensor requiring grad even when no labeled points exist in batch.
    
    def __init__(self, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions, targets, mask=None):
        
        # predictions: (B, C, H, W) logits
        # targets: (B, H, W) long, values in [0..C-1] or ignore_index
        # mask: (B, H, W) float indicating which pixels are annotated (1) or not (0).
        
        device = predictions.device
        dtype = predictions.dtype
        B, C, H, W = predictions.shape

        log_probs = F.log_softmax(predictions, dim=1)  # (B,C,H,W)

        if mask is None:
            mask = (targets != self.ignore_index).float()

        mask = mask.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device)

        # Flatten
        log_probs = log_probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        targets_flat = targets.view(-1)                                    # (B*H*W)
        mask_flat = mask.view(-1)                                          # (B*H*W)

        # valid positions where targets != ignore_index
        valid_mask = (targets_flat != self.ignore_index).float()
        targets_safe = targets_flat.clone()
        targets_safe[targets_flat == self.ignore_index] = 0  # dummy

        # log prob of true class
        logp_true = log_probs.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # (B*H*W)
        masked_logp = logp_true * mask_flat * valid_mask
        num_labeled = (mask_flat * valid_mask).sum()

        if self.reduction == 'mean':
            if num_labeled > 0:
                loss = -masked_logp.sum() / num_labeled
            else:
                # zero scalar but requires grad so backward is safe
                loss = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        elif self.reduction == 'sum':
            loss = -masked_logp.sum()
        else:
            loss = -masked_logp

        return loss

# --------------------------
# Dataset
# --------------------------
class MassachusettsPointsDataset(Dataset):
    
    # Loads full images + full binary masks, but returns:
    #   - image tensor (C,H,W)
    #   - full_mask (H,W) long (0/1)
    #   - point_mask (H,W) long: contains class id at annotated points and -1 elsewhere
    #   - point_loc (H,W) float: 1.0 where annotated, 0 otherwise

    # We simulate point supervision by randomly sampling `points_per_image` positions
    # from the full mask during dataset initialization or on-the-fly.
    
    def __init__(self, images_dir, masks_dir, img_size=256, points_per_image=200,
                 augment=False, seed=0, hist_ref=None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.tif','.tiff')])
        self.mask_paths = sorted([p for p in self.masks_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.tif','.tiff')])

        if len(self.img_paths) != len(self.mask_paths):
            # attempt to pair by filename
            mask_map = {p.stem: p for p in self.mask_paths}
            paired = []
            for p in self.img_paths:
                m = mask_map.get(p.stem)
                if m:
                    paired.append((p,m))
            if not paired:
                raise RuntimeError("Image/mask counts differ and automatic pairing failed.")
            self.img_paths, self.mask_paths = zip(*paired)

        self.img_size = img_size
        self.points_per_image = points_per_image
        self.augment = augment
        self.seed = seed

        # transforms
        self.to_tensor = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # [0,1] float32
        ])
        # For masks, use nearest
        self.mask_resize = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
        ])

        # histogram matching reference
        self.hist_ref = None
        if hist_ref is not None and SKIMAGE_AVAILABLE:
            try:
                ref_img = Image.open(hist_ref).convert("RGB")
                ref_img = ref_img.resize((img_size, img_size))
                self.hist_ref = np.array(ref_img).astype(np.uint8)
            except Exception:
                self.hist_ref = None

        # Pre-seed RNG for deterministic sampling when used
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.img_paths)

    def _sample_points_from_mask(self, mask_np):
        
        # mask_np: (H,W) numpy ints (0/1)
        # returns point_mask (H,W) int64 with -1 for unlabeled, class index where labeled
        #         point_loc (H,W) float32 1.0 where labeled
        
        H, W = mask_np.shape
        point_mask = np.full((H, W), -1, dtype=np.int64)
        point_loc = np.zeros((H, W), dtype=np.float32)

        # sample uniformly over image pixels
        # If the user wants class-aware sampling, adjust here.
        total_pixels = H * W
        N = min(self.points_per_image, total_pixels)
        # Deterministic sampling per call
        idxs = self.rng.choice(total_pixels, size=N, replace=False)
        ys = idxs // W
        xs = idxs % W
        for y, x in zip(ys, xs):
            point_mask[y, x] = int(mask_np[y, x])
            point_loc[y, x] = 1.0

        return point_mask, point_loc

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask 0/255 likely

        # optional histogram matching
        if self.hist_ref is not None and SKIMAGE_AVAILABLE:
            try:
                img_np = np.array(img)
                img_np = match_histograms(img_np, self.hist_ref, multichannel=True)
                img = Image.fromarray(img_np.astype(np.uint8))
            except Exception:
                pass

        img_t = self.to_tensor(img)  # float tensor
        mask_resized = self.mask_resize(mask)
        mask_np = np.array(mask_resized)
        # normalize mask to 0/1 (handle binary mask or 0/255)
        mask_bin = (mask_np > 127).astype(np.int64)

        # sample point annotations
        point_mask_np, point_loc_np = self._sample_points_from_mask(mask_bin)

        # convert to tensors
        full_mask = torch.from_numpy(mask_bin).long()
        point_mask = torch.from_numpy(point_mask_np).long()
        point_loc = torch.from_numpy(point_loc_np).float()

        return img_t, full_mask, point_mask, point_loc

# --------------------------
# Utilities: dataloaders, histogram ref, plotting
# --------------------------
def get_data_dirs(root):
    
    # expects directory structure:
    #   root/images/train, root/images/val, root/images/test
    #   root/masks/train,  root/masks/val,  root/masks/test
    
    root = Path(root)
    images_root = root / "images"
    masks_root = root / "masks"
    if not images_root.exists() or not masks_root.exists():
        raise RuntimeError(f"Expected images/ and masks/ under {root}")
    dirs = {}
    for split in ("train", "val", "test"):
        imgs = images_root / split
        msks = masks_root / split
        if not imgs.exists() or not msks.exists():
            raise RuntimeError(f"Missing split directories: {imgs} or {msks}")
        dirs[split] = (str(imgs), str(msks))
    return dirs

def make_loaders(data_root, config):
    dirs = get_data_dirs(data_root)
    hist_ref = config.get("hist_ref", None)
    train_ds = MassachusettsPointsDataset(dirs["train"][0], dirs["train"][1],
                                          img_size=config["img_size"],
                                          points_per_image=config["points_per_image"],
                                          augment=True, seed=config["seed"], hist_ref=hist_ref)
    val_ds = MassachusettsPointsDataset(dirs["val"][0], dirs["val"][1],
                                        img_size=config["img_size"],
                                        points_per_image=config["points_per_image"],
                                        augment=False, seed=config["seed"]+1, hist_ref=hist_ref)
    test_ds = MassachusettsPointsDataset(dirs["test"][0], dirs["test"][1],
                                         img_size=config["img_size"],
                                         points_per_image=config["points_per_image"],
                                         augment=False, seed=config["seed"]+2, hist_ref=hist_ref)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader

def plot_metrics(history, save_path):
    plt.figure(figsize=(8,4))
    epochs = list(range(1, len(history["train_loss"])+1))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

# --------------------------
# Model / transfer learning
# --------------------------
def build_model(config):
    
    # Build segmentation model using segmentation_models_pytorch (smp).
    # We use a typical encoder-decoder model (Unet with pretrained encoder).
    
    backbone = config["backbone"]
    encoder_weights = config["encoder_weights"]
    num_classes = config["num_classes"]

    model = smp.Unet(encoder_name=backbone,
                     encoder_weights=encoder_weights,
                     in_channels=3,
                     classes=num_classes)

    if encoder_weights is not None and config.get("freeze_encoder", False):
        # freeze encoder parameters
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False

    return model

# --------------------------
# Training & evaluation
# --------------------------
def generate_pseudo_labels(outputs, threshold=0.9):
    
    # outputs: logits (B,C,H,W). We compute softmax and accept pseudo labels
    # where max prob >= threshold. Returns:
    #   pseudo_mask: (B,H,W) long with class idx or -1 for "no pseudo label"
    #   pseudo_conf_mask: (B,H,W) float 1 where pseudo label exists
    
    probs = F.softmax(outputs, dim=1)  # (B,C,H,W)
    maxp, argmaxc = torch.max(probs, dim=1)  # (B,H,W), (B,H,W)
    pseudo_conf_mask = (maxp >= threshold).float()
    pseudo_mask = argmaxc.long()
    # mark unlabeled as ignore by setting to -1 where not confident
    pseudo_mask[ pseudo_conf_mask == 0 ] = -1
    return pseudo_mask, pseudo_conf_mask

def tta_predict(model, images, device):
    
    # Simple TTA: original + horizontal flip. Average softmax probabilities.
    # images: (B,3,H,W) float tensor
    # returns logits averaged (B,C,H,W)
    
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        out1 = model(images)                   # logits
        out1_prob = F.softmax(out1, dim=1)

        # flip
        images_flipped = torch.flip(images, dims=[3])  # horizontal flip
        out2 = model(images_flipped)
        out2_prob = F.softmax(out2, dim=1)
        out2_prob = torch.flip(out2_prob, dims=[3])

        avg_prob = (out1_prob + out2_prob) / 2.0
        avg_logits = torch.log(torch.clamp(avg_prob, 1e-8, 1.0))
    return avg_logits

def train_epoch(model, loader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0.0
    n_batches = 0
    semi_lambda = config.get("semi_lambda", 0.0)
    pseudo_thresh = config.get("pseudo_thresh", 0.9)

    for images, full_masks, point_masks, point_locs in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        full_masks = full_masks.to(device)
        point_masks = point_masks.to(device)    # contains -1 where unlabeled
        point_locs = point_locs.to(device)

        optimizer.zero_grad()
        outputs = model(images)   # logits (B,C,H,W)

        # supervised partial loss (only on point annotations)
        supervised_loss = criterion(outputs, point_masks, mask=point_locs)

        # semi-supervised pseudo-label loss (optional)
        semi_loss = torch.zeros((), device=device, dtype=outputs.dtype, requires_grad=True)
        if semi_lambda > 0.0:
            pseudo_mask, pseudo_conf_mask = generate_pseudo_labels(outputs, threshold=pseudo_thresh)
            # only keep pseudo labels where there was NO manual point annotation
            # create mask of pixels where point_locs==0 but pseudo_conf_mask==1
            no_manual = (point_locs == 0.0).float()
            pseudo_final_mask = pseudo_conf_mask * no_manual  # float
            if pseudo_final_mask.sum() > 0:
                semi_loss = criterion(outputs, pseudo_mask.long(), mask=pseudo_final_mask)
            else:
                semi_loss = torch.zeros((), device=device, dtype=outputs.dtype, requires_grad=True)

        total = supervised_loss + semi_lambda * semi_loss
        # ensure scalar requires grad
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(float(total), device=device, requires_grad=True)

        total.backward()
        optimizer.step()

        total_loss += float(total.item())
        n_batches += 1

    return total_loss / max(1, n_batches)

def validate_epoch(model, loader, criterion, device, config, use_tta=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for images, full_masks, point_masks, point_locs in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            full_masks = full_masks.to(device)
            point_masks = point_masks.to(device)
            point_locs = point_locs.to(device)

            if use_tta and config.get("tta", False):
                logits = tta_predict(model, images, device)
            else:
                logits = model(images)

            loss = criterion(logits, point_masks, mask=point_locs)
            total_loss += float(loss.item())
            n_batches += 1

            preds = torch.argmax(logits, dim=1).cpu().numpy().flatten()
            targets = full_masks.cpu().numpy().flatten()
            preds_all.append(preds)
            targets_all.append(targets)

    if len(preds_all) > 0:
        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)
        try:
            iou = float(jaccard_score(targets_all, preds_all, average="macro"))
        except Exception:
            iou = 0.0
        acc = float(accuracy_score(targets_all, preds_all))
    else:
        iou = 0.0
        acc = 0.0

    return total_loss / max(1, n_batches), acc, iou, preds_all, targets_all

# --------------------------
# Main training runner
# --------------------------
def main():
    print("⚙️ Running with auto-default config (no CLI args provided).")
    device = torch.device(CONFIG["device"])
    print("Device:", device)

    # prepare dataloaders
    print("Preparing datasets and dataloaders...")
    train_loader, val_loader, test_loader = make_loaders(DATA_ROOT, CONFIG)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # build model
    print("Building model (transfer learning backbone: {})".format(CONFIG["backbone"]))
    model = build_model(CONFIG)
    model = model.to(device).float()

    # criterion, optimizer
    criterion = PartialCrossEntropyLoss(ignore_index=-1, reduction="mean")
    # Train only trainable params (encoder may be frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # training
    best_val_iou = -1.0
    history = defaultdict(list)

    print("Starting training...\n")
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, CONFIG)
        val_loss, val_acc, val_iou, _, _ = validate_epoch(model, val_loader, criterion, device, CONFIG, use_tta=True)
        scheduler.step(val_iou)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_iou"].append(val_iou)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val IoU: {val_iou:.4f}")

        # save best
        ckpt_path = os.path.join(RESULTS_ROOT, "checkpoints", f"best_epoch_{epoch}.pth")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(RESULTS_ROOT, "checkpoints", "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": CONFIG,
                "history": dict(history)
            }, best_path)
            print("  Saved best model ->", best_path)

    # save final results/plots
    hist_path = os.path.join(RESULTS_ROOT, "figures", "training_history.png")
    plot_metrics(history, hist_path)
    with open(os.path.join(RESULTS_ROOT, "experiment_results.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "history": history
        }, f, indent=2)

    # final evaluation on test set using best model
    print("\nLoading best model for final evaluation...")
    best_path = os.path.join(RESULTS_ROOT, "checkpoints", "best_model.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        _, test_acc, test_iou, preds, targets = validate_epoch(model, test_loader, criterion, device, CONFIG, use_tta=True)
        print(f"Test Acc: {test_acc:.4f} | Test IoU: {test_iou:.4f}")
        # confusion matrix
        if preds.size > 0:
            cm = confusion_matrix(targets, preds, labels=list(range(CONFIG["num_classes"])))
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=[f"C{i}" for i in range(CONFIG["num_classes"])],
                        yticklabels=[f"C{i}" for i in range(CONFIG["num_classes"])])
            plt.title("Normalized Confusion Matrix (Test)")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_ROOT, "figures", "confusion_matrix_test.png"), dpi=150)
            plt.close()
    else:
        print("Best checkpoint not found; skipping final test evaluation.")

    print("\nDone. Results saved to", RESULTS_ROOT)

if __name__ == "__main__":
    main()
"""