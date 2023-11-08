import torch
import torch.nn.functional as F

from mvs_former.Loss.focal import focal_loss


def ce_loss(inputs, depth_gt_ms, mask_ms, dlossw):
    depth_loss_weights = dlossw

    loss_dict = {}
    for sub_stage_key in inputs:
        if "stage" not in sub_stage_key:
            continue
        stage_inputs = inputs[sub_stage_key]
        stage_key = sub_stage_key.split("_")[0]
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        interval = stage_inputs["interval"]  # float
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        depth_gt_volume = depth_gt.expand_as(depth_values)  # (b, d, h, w)
        # |-|-|-|-|
        #   x x x x
        depth_values_right = depth_values + interval / 2
        out_of_range_left = (depth_gt < depth_values[:, 0:1, :, :]).to(torch.float32)
        out_of_range_right = (depth_gt > depth_values[:, -1:, :, :]).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask
        gt_index_volume = (
            (depth_values_right <= depth_gt_volume)
            .to(torch.float32)
            .sum(dim=1, keepdims=True)
            .to(torch.long)
        )
        gt_index_volume = torch.clamp_max(
            gt_index_volume, max=depth_values.shape[1] - 1
        ).squeeze(1)

        depth_loss = F.cross_entropy(prob_volume_pre, gt_index_volume, reduction="none")
        depth_loss = torch.sum(depth_loss * final_mask) / (torch.sum(final_mask) + 1e-6)

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[sub_stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[sub_stage_key] = depth_loss

    return loss_dict


def ce_loss_stage4(
    inputs, depth_gt_ms, mask_ms, dlossw, focal=False, gamma=0.0, inverse_depth=True
):
    depth_loss_weights = dlossw

    loss_dict = {}
    for stage_inputs, stage_key in [
        (inputs[k], k) for k in ["stage1", "stage2", "stage3", "stage4"]
    ]:
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        depth_gt_volume = depth_gt.expand_as(depth_values)  # (b, d, h, w)
        # inverse depth, depth从大到小变为从小到大
        if inverse_depth:
            depth_values = torch.flip(depth_values, dims=[1])
            prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])
        intervals = (
            torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2
        )  # [b,d-1,h,w]
        intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # [b,d,h,w]
        min_depth_values = (
            depth_values[:, 0:1]
            - intervals[
                :,
                0:1,
            ]
        )
        max_depth_values = depth_values[:, -1:] + intervals[:, -1:]
        depth_values_right = depth_values + intervals
        out_of_range_left = (depth_gt < min_depth_values).to(torch.float32)
        out_of_range_right = (depth_gt > max_depth_values).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask
        gt_index_volume = (
            (depth_values_right <= depth_gt_volume)
            .to(torch.float32)
            .sum(dim=1, keepdims=True)
            .to(torch.long)
        )
        gt_index_volume = torch.clamp_max(
            gt_index_volume, max=depth_values.shape[1] - 1
        ).squeeze(1)

        # mask:[B,H,W], prob:[B,D,H,W], gtd:[B,H,W]
        if focal:
            depth_loss = focal_loss(prob_volume_pre, gt_index_volume, gamma=gamma)
        else:
            final_mask = final_mask.to(torch.bool)
            gt_index_volume = gt_index_volume[final_mask]  # [N,]
            prob_volume_pre = prob_volume_pre.permute(0, 2, 3, 1)[
                final_mask, :
            ]  # [B,H,W,D]->[N,D]
            depth_loss = F.cross_entropy(
                prob_volume_pre, gt_index_volume, reduction="mean"
            )
        # depth_loss = torch.sum(depth_loss * final_mask) / (torch.sum(final_mask) + 1e-6)

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def mixup_ce_loss_stage4(inputs, depth_gt_ms, mask_ms, dlossw, inverse_depth=True):
    depth_loss_weights = dlossw

    loss_dict = {}
    for stage_inputs, stage_key in [
        (inputs[k], k) for k in ["stage1", "stage2", "stage3", "stage4"]
    ]:
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        # inverse depth, depth从大到小变为从小到大
        if inverse_depth:
            depth_values = torch.flip(depth_values, dims=[1])
            prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])

        # 判断out of range
        min_depth_values = depth_values[:, 0:1]
        max_depth_values = depth_values[:, -1:]
        out_of_range_left = (depth_gt < min_depth_values).to(torch.float32)
        out_of_range_right = (depth_gt > max_depth_values).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask  # [b,h,w]

        # 构建GT index,这里获取的label是0~d-2，左右共用此label
        # |○| | |; label=0 and 1
        # | |○| |; label=1 and 2
        # | | |○|; label=2 and 3
        depth_gt_volume = depth_gt.expand_as(depth_values[:, :-1])  # [b,d-1,h,w]
        gt_index_volume = (
            (depth_values[:, 1:] <= depth_gt_volume)
            .to(torch.float32)
            .sum(dim=1, keepdims=True)
            .to(torch.long)
        )  # [b,1,h,w]
        gt_index_volume = torch.clamp_max(
            gt_index_volume, max=depth_values.shape[1] - 2
        ).squeeze(1)  # [b,h,w]

        # 构建mix weights，inverse depth的interval当做线性
        gt_depth_left = torch.gather(
            depth_values[:, :-1], dim=1, index=gt_index_volume.unsqueeze(1)
        )  # [b,1,h,w]
        intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1])  # [b,d-1,h,w]
        intervals = torch.gather(
            intervals, dim=1, index=gt_index_volume.unsqueeze(1)
        )  # [b,1,h,w]
        mix_weights_left = torch.clamp(
            torch.abs(depth_gt - gt_depth_left) / intervals, 0, 1
        ).squeeze(1)  # [b,1,h,w]->[b,h,w]
        mix_weights_right = 1 - mix_weights_left

        # mask:[B,H,W], prob:[B,D,H,W], gtd:[B,H,W]
        # 分别计算左和右loss
        depth_loss_left = F.cross_entropy(
            prob_volume_pre[:, :-1], gt_index_volume, reduction="none"
        )  # [b,h,w]
        depth_loss_left = torch.sum(depth_loss_left * mix_weights_left * final_mask) / (
            torch.sum(final_mask) + 1e-6
        )
        depth_loss_right = F.cross_entropy(
            prob_volume_pre[:, 1:], gt_index_volume, reduction="none"
        )  # [b,h,w]
        depth_loss_right = torch.sum(
            depth_loss_right * mix_weights_right * final_mask
        ) / (torch.sum(final_mask) + 1e-6)
        depth_loss = depth_loss_left + depth_loss_right

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict
