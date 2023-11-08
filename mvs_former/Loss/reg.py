import torch
import torch.nn.functional as F


def reg_loss(inputs, depth_gt_ms, mask_ms, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for stage_inputs, stage_key in [
        (inputs[k], k) for k in ["stage1", "stage2", "stage3"]
    ]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction="mean")

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def reg_loss_stage4(
    inputs,
    depth_gt_ms,
    mask_ms,
    dlossw,
    depth_interval,
    mask_out_range=False,
    inverse_depth=True,
):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for stage_inputs, stage_key in [
        (inputs[k], k) for k in ["stage1", "stage2", "stage3", "stage4"]
    ]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        if mask_out_range:
            depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
            if inverse_depth:
                depth_values = torch.flip(depth_values, dims=[1])
            intervals = (
                torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2
            )  # [b,d-1,h,w]
            intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # [b,d,h,w]
            min_depth_values = (
                depth_values[:, 0]
                - intervals[
                    :,
                    0,
                ]
            )
            max_depth_values = depth_values[:, -1] + intervals[:, -1]
            depth_gt_ = depth_gt_ms[stage_key]
            out_of_range_left = (depth_gt_ < min_depth_values).to(torch.float32)
            out_of_range_right = (depth_gt_ > max_depth_values).to(torch.float32)
            out_of_range_mask = torch.clamp(
                out_of_range_left + out_of_range_right, 0, 1
            )
            in_range_mask = (1 - out_of_range_mask).to(torch.bool)
            mask = mask & in_range_mask

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction="mean")

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def cvx_reg_loss(inputs, depth_gt, mask, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for stage_inputs, stage_key in [
        (inputs[k], k) for k in ["stage1", "stage2", "stage3"]
    ]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt_stage = F.interpolate(
            depth_gt.unsqueeze(1),
            size=(depth_est.shape[1], depth_est.shape[2]),
            mode="nearest",
        ).squeeze(1)
        mask_stage = F.interpolate(
            mask.unsqueeze(1),
            size=(depth_est.shape[1], depth_est.shape[2]),
            mode="nearest",
        ).squeeze(1)
        depth_gt_stage = depth_gt_stage / depth_interval
        mask_stage = mask_stage > 0.5

        depth_loss = F.smooth_l1_loss(
            depth_est[mask_stage], depth_gt_stage[mask_stage], reduction="mean"
        )

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict
