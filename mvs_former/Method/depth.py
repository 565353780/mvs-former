import torch
import torch.nn.functional as F


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def conf_regression(p, n=4):
    ndepths = p.size(1)
    with torch.no_grad():
        if n % 2 == 1:
            prob_volume_sum4 = n * F.avg_pool3d(
                F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2, n // 2]),
                (n, 1, 1),
                stride=1,
                padding=0,
            ).squeeze(1)
        else:
            prob_volume_sum4 = n * F.avg_pool3d(
                F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2 - 1, n // 2]),
                (n, 1, 1),
                stride=1,
                padding=0,
            ).squeeze(1)
        depth_index = depth_regression(
            p.detach(),
            depth_values=torch.arange(ndepths, device=p.device, dtype=torch.float),
        ).long()
        depth_index = depth_index.clamp(min=0, max=ndepths - 1)
        conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1))
    return conf.squeeze(1)


def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (
        torch.arange(
            0, ndepths, device=device, dtype=dtype, requires_grad=False
        ).reshape(1, -1)
        * new_interval.squeeze(1)
    )  # (B, D)
    depth_range_samples = (
        depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
    )  # (B, D, H, W)
    return depth_range_samples


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1.0 / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1.0 / cur_depth[:, -1]
    itv = torch.arange(
        0, ndepths, device=device, dtype=dtype, requires_grad=False
    ).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = (
        inverse_depth_max[:, None, None, None]
        + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv
    )

    return 1.0 / inverse_depth_hypo


def schedule_inverse_range(depth, depth_hypo, ndepths, split_itv, H, W):
    last_depth_itv = 1.0 / depth_hypo[:, 2, :, :] - 1.0 / depth_hypo[:, 1, :, :]
    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
    inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(
        0,
        ndepths,
        device=inverse_min_depth.device,
        dtype=inverse_min_depth.dtype,
        requires_grad=False,
    ).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = (
        inverse_max_depth[:, None, :, :]
        + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv
    )  # B D H W
    inverse_depth_hypo = F.interpolate(
        inverse_depth_hypo.unsqueeze(1),
        [ndepths, H, W],
        mode="trilinear",
        align_corners=True,
    ).squeeze(1)
    return 1.0 / inverse_depth_hypo


def init_inverse_range_eth3d(cur_depth, ndepths, device, dtype, H, W):
    cur_depth = torch.clamp(cur_depth, min=0.01, max=50)

    inverse_depth_min = 1.0 / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1.0 / cur_depth[:, -1]

    itv = torch.arange(
        0, ndepths, device=device, dtype=dtype, requires_grad=False
    ).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = (
        inverse_depth_max[:, None, None, None]
        + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv
    )

    return 1.0 / inverse_depth_hypo


def schedule_inverse_range_eth3d(depth, depth_hypo, ndepths, split_itv, H, W):
    last_depth_itv = 1.0 / depth_hypo[:, 2, :, :] - 1.0 / depth_hypo[:, 1, :, :]
    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
    inverse_max_depth = (
        1 / depth - split_itv * last_depth_itv
    )  # B H W 只有他可能是负数！

    is_neg = (inverse_max_depth < 0.02).float()
    inverse_max_depth = inverse_max_depth - (inverse_max_depth - 0.02) * is_neg
    inverse_min_depth = inverse_min_depth - (inverse_max_depth - 0.02) * is_neg

    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(
        0,
        ndepths,
        device=inverse_min_depth.device,
        dtype=inverse_min_depth.dtype,
        requires_grad=False,
    ).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = (
        inverse_max_depth[:, None, :, :]
        + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv
    )  # B D H W
    inverse_depth_hypo = F.interpolate(
        inverse_depth_hypo.unsqueeze(1),
        [ndepths, H, W],
        mode="trilinear",
        align_corners=True,
    ).squeeze(1)
    return 1.0 / inverse_depth_hypo


def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    # shape, (B, H, W)
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    cur_depth_min = (
        cur_depth - ndepth / 2 * depth_inteval_pixel[:, None, None]
    )  # (B, H, W)
    cur_depth_min = torch.clamp_min(cur_depth_min, 0.01)
    cur_depth_max = cur_depth + ndepth / 2 * depth_inteval_pixel[:, None, None]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (
        torch.arange(
            0,
            ndepth,
            device=cur_depth.device,
            dtype=cur_depth.dtype,
            requires_grad=False,
        ).reshape(1, -1, 1, 1)
        * new_interval.unsqueeze(1)
    )
    depth_range_samples = F.interpolate(
        depth_range_samples.unsqueeze(1),
        [ndepth, H, W],
        mode="trilinear",
        align_corners=True,
    ).squeeze(1)
    return depth_range_samples
