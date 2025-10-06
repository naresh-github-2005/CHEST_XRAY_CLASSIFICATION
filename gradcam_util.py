# gradcam_util.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from matplotlib import cm

def _find_last_conv_layer(model: nn.Module):
    # Prefer layer4[-1].conv3 for ResNet50, fall back to last Conv2d
    try:
        return model.layer4[-1].conv3
    except Exception:
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                return m
    raise RuntimeError("No Conv2d layer found in model")

def compute_gradcam_overlay(model: nn.Module, input_tensor: torch.Tensor, class_idx: int, resize_hw=(224,224)):
    """
    Compute a Grad-CAM overlay image.
    - model: model in eval mode
    - input_tensor: 1 x C x H x W (already normalized, on the same device as model)
    - class_idx: integer class index to visualize
    - resize_hw: size of overlay (H, W)
    Returns: uint8 HxWx3 RGB overlay image (0..255)
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    model.zero_grad()
    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        # grad_output is a tuple; take the first entry
        gradients = grad_output[0].detach()

    target_layer = _find_last_conv_layer(model)
    fh = target_layer.register_forward_hook(forward_hook)
    # register full backward hook if available, otherwise module.register_backward_hook (deprecated)
    if hasattr(target_layer, "register_full_backward_hook"):
        bh = target_layer.register_full_backward_hook(lambda module, gi, go: backward_hook(module, gi, go))
    else:
        bh = target_layer.register_backward_hook(backward_hook)

    # forward
    outputs = model(input_tensor)          # logits (1, num_classes)
    # select the score for the class of interest
    score = outputs[0, class_idx]
    # backward
    model.zero_grad()
    score.backward(retain_graph=False)

    # activations: (1, C, H', W'), gradients: (1, C, H', W')
    # compute channel-wise weights (global avg pooling of gradients)
    weights = gradients.mean(dim=(2,3), keepdim=True)   # (1, C, 1, 1)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, H', W')
    cam = cam.squeeze().cpu().numpy()  # (H', W')

    # normalize cam to 0..1
    cam -= cam.min()
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    # Unnormalize input_tensor to image (0..1)
    # Assumes ImageNet normalization used: mean/std below
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = input_tensor[0].cpu().numpy()         # C,H,W
    inp = np.transpose(inp, (1,2,0))           # H,W,C
    inp = (inp * std) + mean                   # unnormalize
    inp = np.clip(inp, 0.0, 1.0)

    # Resize cam to input H,W using PIL (no cv2 dependency)
    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_resized = np.array(cam_img.resize((inp.shape[1], inp.shape[0]), resample=Image.BILINEAR)) / 255.0

    # Apply colormap (matplotlib)
    cmap = cm.get_cmap("jet")
    cam_color = cmap(cam_resized)[:,:,:3]   # H,W,3

    # Blend cam with original image
    overlay = 0.5 * cam_color + 0.5 * inp
    overlay = np.clip(overlay, 0.0, 1.0)

    # Cleanup hooks
    try:
        fh.remove()
    except Exception:
        pass
    try:
        bh.remove()
    except Exception:
        pass

    return (overlay * 255).astype("uint8")
