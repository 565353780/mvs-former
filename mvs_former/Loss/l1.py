import torch.nn.functional as F


def simple_loss(outputs, depth_gt_ms, mask_ms):
    depth_est = outputs["depth"]
    depth_gt = depth_gt_ms
    mask = mask_ms
    mask = mask > 0.5

    depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction="mean")

    return depth_loss
