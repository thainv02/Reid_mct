"""
Feature extraction functions for ReID.
"""
import torch
import numpy as np


def extract_feature(model, img, transform, device):
    """
    Extract feature from a single image.
    
    Args:
        model: ReID model
        img: PIL Image
        transform: Image transform
        device: Device ('cuda' or 'cpu')
    
    Returns:
        numpy array: Feature vector
    """
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    cam_label = torch.tensor([0], device=device)
    with torch.no_grad():
        feat = model(img, cam_label=cam_label)
    return feat.cpu().numpy()


def extract_features_batch(model, imgs, transform, device):
    """
    Extract features for multiple images at once (GPU batch processing).
    Much faster than calling extract_feature() multiple times.
    
    Args:
        model: ReID model
        imgs: List of PIL Images
        transform: Image transform
        device: GPU device
    
    Returns:
        numpy array: Features [N, feat_dim]
    """
    if len(imgs) == 0:
        return np.array([])
    
    # Transform all images
    batch = torch.stack([transform(img) for img in imgs]).to(device)
    cam_labels = torch.zeros(len(imgs), dtype=torch.long, device=device)
    
    with torch.no_grad():
        feats = model(batch, cam_label=cam_labels)
    
    return feats.cpu().numpy()
