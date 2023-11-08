def wasserstein_loss(
    inputs,
    depth_gt_ms,
    mask_ms,
    dlossw,
    ot_iter=10,
    ot_eps=1,
    ot_continous=False,
    inverse=True,
):
    total_loss = {}
    stage_ot_loss = []
    # range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate(
        [(inputs[k], k) for k in inputs.keys() if "stage" in k]
    ):
        hypo_depth = stage_inputs["depth_values"]
        attn_weight = stage_inputs["prob_volume"]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # # mask range
        # if inverse:
        #     depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
        #     mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0  # B H W
        # else:
        #     depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
        #     mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0  # B H W
        # range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(
            depth_gt,
            hypo_depth,
            attn_weight,
            mask,
            iters=ot_iter,
            eps=ot_eps,
            continuous=ot_continous,
        )[1]

        stage_ot_loss.append(this_stage_ot_loss)
        total_loss[stage_key] = dlossw[stage_idx] * this_stage_ot_loss

    return total_loss  # , range_err_ratio
