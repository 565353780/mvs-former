import gc
import os
import sys
import time

import cv2
import numpy as np
import torch.nn.parallel
from plyfile import PlyData, PlyElement
from torch.utils.data import DataLoader, SequentialSampler

from mvs_former.Config.parser import getParserArgs
from mvs_former.Data.config_parser import ConfigParser
from mvs_former.Data.dict_average_meter import DictAverageMeter
from mvs_former.Dataset.data_loaders import DTULoader
from mvs_former.Dataset.tt import TTDataset
from mvs_former.Method.data_io import save_pfm
from mvs_former.Method.fusion import (
    ave_fusion,
    bin_op_reduce,
    get_pixel_grids,
    get_reproj,
    get_reproj_dynamic,
    idx_cam2world,
    idx_img2cam,
    prob_filter,
    vis_filter,
    vis_filter_dynamic,
)
from mvs_former.Method.gipuma import gipuma_filter
from mvs_former.Method.io import write_cam
from mvs_former.Method.utils import print_args, tensor2float, tensor2numpy, tocuda
from mvs_former.Metric.abs_depth_error import AbsDepthError_metrics
from mvs_former.Metric.thres import Thres_metrics
from mvs_former.Model.mvsformer_model import DINOMVSNet, TwinMVSNet

args = getParserArgs()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)

# args.outdir = args.outdir + f'_{args.max_w}x{args.max_h}'
os.makedirs(args.outdir, exist_ok=True)


# run model to save depth maps and confidence maps
def save_depth(testlist, config):
    # dataset, dataloader

    init_kwags = {
        "data_path": args.testpath,
        "data_list": testlist,
        "mode": "test",
        "num_srcs": args.num_view,
        "num_depths": args.numdepth,
        "interval_scale": Interval_Scale,
        "shuffle": False,
        "batch_size": 1,
        "fix_res": args.fix_res,
        "max_h": args.max_h,
        "max_w": args.max_w,
        "dataset_eval": args.dataset,
        "iterative": False,  # iterative inference
        "refine": not args.no_refinement,
        "use_short_range": args.use_short_range,
        "num_workers": 4,
    }
    test_data_loader = DTULoader(**init_kwags)

    # model
    # build models architecture, then print to console
    if config["arch"]["args"]["vit_args"].get("twin", False):
        model = TwinMVSNet(config["arch"]["args"])
    else:
        model = DINOMVSNet(config["arch"]["args"])

    print("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(str(config.resume))
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, val in state_dict.items():
        new_state_dict[key.replace("module.", "")] = val
    model.load_state_dict(new_state_dict, strict=True)

    # prepare models for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # temp setting
    if (
        hasattr(model, "vit_args")
        and "height" in model.vit_args
        and "width" in model.vit_args
    ):
        model.vit_args["height"] = args.max_h // 2
        model.vit_args["width"] = args.max_w // 2

    times = []

    # get tmp
    if args.tmps is not None:
        tmp = [float(a) for a in args.tmps.split(",")]
    else:
        tmp = args.tmp

    valid_metrics = DictAverageMeter()
    valid_metrics.reset()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_data_loader):
            torch.cuda.synchronize()
            start_time = time.time()
            sample_cuda = tocuda(sample)
            num_stage = 3 if args.no_refinement else 4
            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
            if args.dataset == "dtu":
                depth_gt = None
                if "depth" in sample_cuda.keys():
                    depth_gt = sample_cuda["depth"]["stage{}".format(num_stage)]
                mask = None
                if "mask" in sample_cuda.keys():
                    mask = sample_cuda["mask"]["stage{}".format(num_stage)]
            B, V, _, H, W = imgs.shape
            depth_interval = (
                sample_cuda["depth_values"][:, 1] - sample_cuda["depth_values"][:, 0]
            )
            filenames = sample["filename"]
            # with torch.cuda.amp.autocast():
            outputs = model.forward(
                imgs, cam_params, sample_cuda["depth_values"], tmp=tmp
            )
            torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)
            depth_est_cuda = outputs["refined_depth"]
            outputs = tensor2numpy(outputs)
            del sample_cuda

            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            # imgs = sample["imgs"].numpy()
            print(
                "Iter {}/{}, Time:{} Res:{}".format(
                    batch_idx,
                    len(test_data_loader),
                    end_time - start_time,
                    outputs["refined_depth"][0].shape,
                )
            )

            # save depth maps and confidence maps
            for (
                filename,
                cam,
                img,
                depth_est,
                conf_stage1,
                conf_stage2,
                conf_stage3,
                conf_stage4,
                conf_stage4_,
            ) in zip(
                filenames,
                cams,
                imgs,
                outputs["refined_depth"],
                outputs["stage1"]["photometric_confidence"],
                outputs["stage2"]["photometric_confidence"],
                outputs["stage3"]["photometric_confidence"],
                outputs["photometric_confidence"],
                outputs["stage4"]["photometric_confidence"],
            ):
                img = img[0]  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(
                    args.outdir, filename.format("depth_est", ".pfm")
                )
                confidence_filename = os.path.join(
                    args.outdir, filename.format("confidence", ".npy")
                )
                cam_filename = os.path.join(
                    args.outdir, filename.format("cams", "_cam.txt")
                )
                img_filename = os.path.join(
                    args.outdir, filename.format("images", ".jpg")
                )
                os.makedirs(depth_filename.rsplit("/", 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit("/", 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit("/", 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit("/", 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                h, w = depth_est.shape[0], depth_est.shape[1]
                # save confidence maps
                if args.combine_conf:
                    photometric_confidence = conf_stage4
                    if args.save_all_confs:  # only for visualization
                        all_confidence_filename = os.path.join(
                            args.outdir, filename.format("confidence_all", ".npy")
                        )
                        os.makedirs(
                            all_confidence_filename.rsplit("/", 1)[0], exist_ok=True
                        )
                        all_photometric_confidence = np.stack(
                            [conf_stage1, conf_stage2, conf_stage3, conf_stage4_]
                        ).transpose([1, 2, 0])
                        np.save(all_confidence_filename, all_photometric_confidence)
                else:
                    conf_stage1 = cv2.resize(
                        conf_stage1, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    conf_stage2 = cv2.resize(
                        conf_stage2, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    conf_stage3 = cv2.resize(
                        conf_stage3, (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    photometric_confidence = np.stack(
                        [conf_stage1, conf_stage2, conf_stage3, conf_stage4_]
                    ).transpose([1, 2, 0])
                np.save(confidence_filename, photometric_confidence)
                # save_pfm(confidence_filename, photometric_confidence)
                # save cams, img
                # std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
                # mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
                std = torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(
                    (3, 1, 1)
                )
                mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(
                    (3, 1, 1)
                )
                img = img * std + mean
                img = img.permute(1, 2, 0).cpu().numpy()
                # img = np.transpose(img, (1, 2, 0))
                # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
                write_cam(cam_filename, cam)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                # print(img.shape)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                if args.dataset == "dtu":
                    di = depth_interval[0].item() / 2.65
                    if depth_gt is not None and mask is not None:
                        scalar_outputs = {
                            "abs_depth_error": AbsDepthError_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5
                            ),
                            "thres1mm_error": Thres_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5, di
                            ),
                            "thres2mm_error": Thres_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5, di * 2
                            ),
                            "thres4mm_error": Thres_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5, di * 4
                            ),
                            "thres8mm_error": Thres_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5, di * 8
                            ),
                            "thres14mm_error": Thres_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5, di * 14
                            ),
                            "thres20mm_error": Thres_metrics(
                                depth_est_cuda, depth_gt, mask > 0.5, di * 20
                            ),
                        }
                        scalar_outputs = tensor2float(scalar_outputs)
                        valid_metrics.update(scalar_outputs)

    print("average time: ", sum(times) / len(times))
    if args.dataset == "dtu":
        valid_metrics = valid_metrics.mean()
        with open(os.path.join(args.outdir, "depth_metric.txt"), "w") as w:
            for k in valid_metrics:
                w.write(k + " " + str(valid_metrics[k]) + "\n")
    torch.cuda.empty_cache()
    gc.collect()


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    tt_dataset = TTDataset(pair_folder, scan_folder, n_src_views=10)
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(
        tt_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    views = {}
    prob_threshold = args.prob_threshold
    prob_threshold = [float(p) for p in prob_threshold.split(",")]
    for sample_np in tt_dataloader:
        sample = tocuda(sample_np)
        for ids in range(sample["src_depths"].size(1)):
            if args.combine_conf:
                src_prob_mask = sample["src_confs"][:, ids] > prob_threshold[0]
            else:
                src_prob_mask = prob_filter(
                    sample["src_confs"][:, ids, ...], prob_threshold
                )
            sample["src_depths"][:, ids, ...] *= src_prob_mask.float()

        if args.combine_conf:
            prob_mask = sample["ref_conf"] > prob_threshold[0]
        else:
            prob_mask = prob_filter(sample["ref_conf"], prob_threshold)

        reproj_xyd, in_range = get_reproj(
            *[
                sample[attr]
                for attr in ["ref_depth", "src_depths", "ref_cam", "src_cams"]
            ]
        )
        vis_masks, vis_mask = vis_filter(
            sample["ref_depth"],
            reproj_xyd,
            in_range,
            args.thres_disp,
            0.01,
            args.thres_view,
        )

        ref_depth_ave = ave_fusion(sample["ref_depth"], reproj_xyd, vis_masks)

        mask = bin_op_reduce([prob_mask, vis_mask], torch.min)

        idx_img = get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
        idx_cam = idx_img2cam(idx_img, ref_depth_ave, sample["ref_cam"])
        points = idx_cam2world(idx_cam, sample["ref_cam"])[..., :3, 0].permute(
            0, 3, 1, 2
        )

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(bool)
        # dir_vecs = dir_vecs.cpu().data.numpy()
        ref_img = sample_np["ref_img"].data.numpy()
        for i in range(points_np.shape[0]):
            print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255
            # d_f_list = [dir_vecs[i, k][mask_np[i, 0]] for k in range(3)]
            # d_f = np.stack(d_f_list, -1)
            ref_id = str(sample_np["ref_id"][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print(
                "processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
                    scan_folder,
                    int(ref_id),
                    prob_mask[i].float().mean().item(),
                    vis_mask[i].float().mean().item(),
                    mask[i].float().mean().item(),
                )
            )

    print("Write combined PCD")
    p_all, c_all = [
        np.concatenate([v[k] for v in views.values()], axis=0) for k in range(2)
    ]

    vertexs = np.array(
        [tuple(v) for v in p_all], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    vertex_colors = np.array(
        [tuple(v) for v in c_all],
        dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def dynamic_filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    tt_dataset = TTDataset(pair_folder, scan_folder, n_src_views=10)
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(
        tt_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    views = {}
    prob_threshold = args.prob_threshold
    prob_threshold = [float(p) for p in prob_threshold.split(",")]
    for sample_np in tt_dataloader:
        num_src_views = sample_np["src_depths"].shape[1]
        dy_range = num_src_views + 1  # 10
        sample = tocuda(sample_np)

        if args.combine_conf:
            prob_mask = sample["ref_conf"] > prob_threshold[0]
        else:
            prob_mask = prob_filter(sample["ref_conf"], prob_threshold)

        ref_depth = sample["ref_depth"]  # [n 1 h w ]
        reproj_xyd = get_reproj_dynamic(
            *[
                sample[attr]
                for attr in ["ref_depth", "src_depths", "ref_cam", "src_cams"]
            ]
        )
        # reproj_xyd   nv 3 h w

        # 4 1300
        vis_masks, vis_mask = vis_filter_dynamic(
            sample["ref_depth"],
            reproj_xyd,
            dist_base=args.dist_base,
            rel_diff_base=args.rel_diff_base,
        )

        # mask reproj_depth
        reproj_depth = reproj_xyd[:, :, -1]  # [1 v h w]
        reproj_depth[~vis_mask.squeeze(2)] = 0  # [n v h w ]
        geo_mask_sums = vis_masks.sum(dim=1)  # 0~v
        geo_mask_sum = vis_mask.sum(dim=1)
        depth_est_averaged = (
            torch.sum(reproj_depth, dim=1, keepdim=True) + ref_depth
        ) / (geo_mask_sum + 1)  # [1,1,h,w]
        geo_mask = geo_mask_sum >= dy_range  # all zero
        for i in range(2, dy_range):
            geo_mask = torch.logical_or(geo_mask, geo_mask_sums[:, i - 2] >= i)

        mask = bin_op_reduce([prob_mask, geo_mask], torch.min)
        idx_img = get_pixel_grids(*depth_est_averaged.size()[-2:]).unsqueeze(0)
        idx_cam = idx_img2cam(idx_img, depth_est_averaged, sample["ref_cam"])
        points = idx_cam2world(idx_cam, sample["ref_cam"])[..., :3, 0].permute(
            0, 3, 1, 2
        )

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(bool)

        ref_img = sample_np["ref_img"].data.numpy()
        for i in range(points_np.shape[0]):
            print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255

            ref_id = str(sample_np["ref_id"][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print(
                "processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
                    scan_folder,
                    int(ref_id),
                    prob_mask[i].float().mean().item(),
                    geo_mask[i].float().mean().item(),
                    mask[i].float().mean().item(),
                )
            )

    print("Write combined PCD")
    p_all, c_all = [
        np.concatenate([v[k] for v in views.values()], axis=0) for k in range(2)
    ]

    vertexs = np.array(
        [tuple(v) for v in p_all], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    vertex_colors = np.array(
        [tuple(v) for v in c_all],
        dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def pcd_filter_worker(scan):
    save_name = "{}.ply".format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    if args.filter_method == "pcd":
        filter_depth(
            pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name)
        )
    else:
        dynamic_filter_depth(
            pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name)
        )


def pcd_filter(testlist):
    for scan in testlist:
        pcd_filter_worker(scan)


if __name__ == "__main__":
    config = ConfigParser.from_args(parser, mkdir=False)

    if args.ndepths is not None:
        config["arch"]["args"]["ndepths"] = [int(d) for d in args.ndepths.split(",")]
    if args.depth_interals_ratio is not None:
        config["arch"]["args"]["depth_interals_ratio"] = [
            float(d) for d in args.depth_interals_ratio.split(",")
        ]

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        # for tanks & temples or eth3d or colmap
        testlist = (
            [
                e
                for e in os.listdir(args.testpath)
                if os.path.isdir(os.path.join(args.testpath, e))
            ]
            if not args.testpath_single_scene
            else [os.path.basename(args.testpath_single_scene)]
        )

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist, config)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    if args.filter_method == "pcd" or args.filter_method == "dpcd":
        # support multi-processing, the default number of worker is 4
        pcd_filter(testlist)

    elif args.filter_method == "gipuma":
        prob_threshold = args.prob_threshold
        prob_threshold = [float(p) for p in prob_threshold.split(",")]
        gipuma_filter(
            testlist,
            args.outdir,
            prob_threshold,
            args.disp_threshold,
            args.num_consistent,
            args.fusibile_exe_path,
        )
    else:
        raise NotImplementedError
