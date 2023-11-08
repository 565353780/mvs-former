import torch


def bimodel_loss(inputs, depth_gt_ms, mask_ms, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for stage_inputs, stage_key in [
        (inputs[k], k) for k in ["stage1", "stage2", "stage3"]
    ]:
        depth0 = stage_inputs["depth0"].to(torch.float32) / depth_interval
        depth1 = stage_inputs["depth1"].to(torch.float32) / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        sigma0 = stage_inputs["sigma0"].to(torch.float32)
        sigma1 = stage_inputs["sigma1"].to(torch.float32)
        pi0 = stage_inputs["pi0"].to(torch.float32)
        pi1 = stage_inputs["pi1"].to(torch.float32)
        dist0 = pi0 * 0.5 * torch.exp(-(torch.abs(depth_gt - depth0) / sigma0)) / sigma0
        dist1 = pi1 * 0.5 * torch.exp(-(torch.abs(depth_gt - depth1) / sigma1)) / sigma1

        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = -torch.log(dist0[mask] + dist1[mask] + 1e-8)
        depth_loss = depth_loss.mean()

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict
