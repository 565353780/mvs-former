import torch


def sinkhorn(gt_depth, hypo_depth, attn_weight, mask, iters, eps=1, continuous=False):
    """
    gt_depth: B H W
    hypo_depth: B D H W
    attn_weight: B D H W
    mask: B H W
    """
    B, D, H, W = attn_weight.shape
    if not continuous:
        D_map = torch.stack(
            [
                torch.arange(-i, D - i, 1, dtype=torch.float32, device=gt_depth.device)
                for i in range(D)
            ],
            dim=1,
        ).abs()
        D_map = D_map[None, None, :, :].repeat(B, H * W, 1, 1)  # B HW D D
        gt_indices = (
            torch.abs(hypo_depth - gt_depth[:, None, :, :])
            .min(1)[1]
            .squeeze(1)
            .reshape(B * H * W, 1)
        )  # BHW, 1
        gt_dist = torch.zeros_like(hypo_depth).permute(0, 2, 3, 1).reshape(B * H * W, D)
        gt_dist.scatter_add_(
            1,
            gt_indices,
            torch.ones(
                [gt_dist.shape[0], 1], dtype=gt_dist.dtype, device=gt_dist.device
            ),
        )
        gt_dist = gt_dist.reshape(B, H * W, D)  # B HW D
    else:
        gt_dist = torch.zeros(
            (B, H * W, D + 1),
            dtype=torch.float32,
            device=gt_depth.device,
            requires_grad=False,
        )  # B HW D+1
        gt_dist[:, :, -1] = 1
        D_map = torch.zeros(
            (B, D, D + 1),
            dtype=torch.float32,
            device=gt_depth.device,
            requires_grad=False,
        )  # B D D+1
        D_map[:, :D, :D] = (
            torch.stack(
                [
                    torch.arange(
                        -i, D - i, 1, dtype=torch.float32, device=gt_depth.device
                    )
                    for i in range(D)
                ],
                dim=1,
            )
            .abs()
            .unsqueeze(0)
        )  # B D D+1
        D_map = D_map[:, None, None, :, :].repeat(1, H, W, 1, 1)  # B H W D D+1
        itv = 1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]  # B H W
        gt_bin_distance_ = (1 / gt_depth - 1 / hypo_depth[:, 0, :, :]) / itv  # B H W
        # FIXME hard code 100
        gt_bin_distance_[~mask] = 10

        gt_bin_distance = torch.stack(
            [(gt_bin_distance_ - i).abs() for i in range(D)], dim=1
        ).permute(0, 2, 3, 1)  # B H W D
        D_map[:, :, :, :, -1] = gt_bin_distance
        D_map = D_map.reshape(B, H * W, D, 1 + D)  # B HW D D+1

    pred_dist = attn_weight.permute(0, 2, 3, 1).reshape(B, H * W, D)  # B HW D

    # map to log space for stability
    log_mu = (gt_dist + 1e-12).log()
    log_nu = (pred_dist + 1e-12).log()  # B HW D or D+1

    u, v = torch.zeros_like(log_nu), torch.zeros_like(log_mu)
    for _ in range(iters):
        # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
        v = log_mu - torch.logsumexp(
            D_map / eps + u.unsqueeze(3), dim=2
        )  # log(sum(exp()))
        u = log_nu - torch.logsumexp(D_map / eps + v.unsqueeze(2), dim=3)

    # convert back from log space, recover probabilities by normalization 2W
    T_map = (D_map / eps + u.unsqueeze(3) + v.unsqueeze(2)).exp()  # B HW D D
    loss = (T_map * D_map).reshape(B * H * W, -1)[mask.reshape(-1)].sum(-1).mean()

    return T_map, loss
