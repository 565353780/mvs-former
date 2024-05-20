import gc
import os
import sys
import time

import cv2
import numpy as np
import torch.nn.parallel
from typing import Union

from mvs_former.Config.parser import getParserArgs
from mvs_former.Data.config_parser import ConfigParser
from mvs_former.Data.dict_average_meter import DictAverageMeter
from mvs_former.Dataset.data_loaders import DTULoader
from mvs_former.Method.data_io import save_pfm
from mvs_former.Method.gipuma import gipuma_filter
from mvs_former.Method.io import write_cam
from mvs_former.Method.utils import print_args, tensor2float, tensor2numpy, tocuda
from mvs_former.Metric.abs_depth_error import AbsDepthError_metrics
from mvs_former.Metric.thres import Thres_metrics
from mvs_former.Model.mvsformer_model import DINOMVSNet, TwinMVSNet


class Detector(object):
    def __init__(self, model_file_path=None) -> None:
        self.parser = getParserArgs()
        self.args = self.parser.parse_args()
        print("argv:", sys.argv[1:])
        if self.args.testpath_single_scene:
            self.args.testpath = os.path.dirname(self.args.testpath_single_scene)
        print_args(self.args)

        self.args.prob_threshold = [
            float(p) for p in self.args.prob_threshold.split(",")
        ]
        self.config = ConfigParser.from_args(self.parser, mkdir=False)

        if self.args.ndepths is not None:
            self.config["arch"]["args"]["ndepths"] = [
                int(d) for d in self.args.ndepths.split(",")
            ]
        if self.args.depth_interals_ratio is not None:
            self.config["arch"]["args"]["depth_interals_ratio"] = [
                float(d) for d in self.args.depth_interals_ratio.split(",")
            ]

        print("***********Interval_Scale**********\n", self.args.interval_scale)

        # args.outdir = args.outdir + f'_{args.max_w}x{args.max_h}'
        os.makedirs(self.args.outdir, exist_ok=True)
        return

    def save_depth(self, testlist):
        # dataset, dataloader
        init_kwags = {
            "data_path": self.args.testpath,
            "data_list": testlist,
            "mode": "test",
            "num_srcs": self.args.num_view,
            "num_depths": self.args.numdepth,
            "interval_scale": self.args.interval_scale,
            "shuffle": False,
            "batch_size": 1,
            "fix_res": self.args.fix_res,
            "max_h": self.args.max_h,
            "max_w": self.args.max_w,
            "dataset_eval": self.args.dataset,
            "iterative": False,  # iterative inference
            "refine": not self.args.no_refinement,
            "use_short_range": self.args.use_short_range,
            "num_workers": 4,
        }
        test_data_loader = DTULoader(**init_kwags)

        # model
        # build models architecture, then print to console
        if self.config["arch"]["args"]["vit_args"].get("twin", False):
            model = TwinMVSNet(self.config["arch"]["args"])
        else:
            model = DINOMVSNet(self.config["arch"]["args"])

        print("Loading checkpoint: {} ...".format(self.config.resume))
        checkpoint = torch.load(str(self.config.resume))
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
            model.vit_args["height"] = self.args.max_h // 2
            model.vit_args["width"] = self.args.max_w // 2

        times = []

        # get tmp
        if self.args.tmps is not None:
            tmp = [float(a) for a in self.args.tmps.split(",")]
        else:
            tmp = self.args.tmp

        valid_metrics = DictAverageMeter()
        valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_data_loader):
                torch.cuda.synchronize()
                start_time = time.time()
                sample_cuda = tocuda(sample)
                num_stage = 3 if self.args.no_refinement else 4
                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
                if self.args.dataset == "dtu":
                    depth_gt = None
                    mask = None
                    if "mask" in sample_cuda.keys():
                        mask = sample_cuda["mask"]["stage{}".format(num_stage)]
                B, V, _, H, W = imgs.shape
                depth_interval = (
                    sample_cuda["depth_values"][:, 1]
                    - sample_cuda["depth_values"][:, 0]
                )
                filenames = sample["filename"]

                skip_iter = True
                for filename in filenames:
                    depth_filename = os.path.join(
                        self.args.outdir, filename.format("depth_est", ".pfm")
                    )
                    if not os.path.exists(depth_filename):
                        skip_iter = False
                        break
                    confidence_filename = os.path.join(
                        self.args.outdir, filename.format("confidence", ".npy")
                    )
                    if not os.path.exists(confidence_filename):
                        skip_iter = False
                        break
                    cam_filename = os.path.join(
                        self.args.outdir, filename.format("cams", "_cam.txt")
                    )
                    if not os.path.exists(cam_filename):
                        skip_iter = False
                        break
                    img_filename = os.path.join(
                        self.args.outdir, filename.format("images", ".jpg")
                    )
                    if not os.path.exists(img_filename):
                        skip_iter = False
                        break

                if skip_iter:
                    print(
                        "Iter {}/{}, found result, skipped!".format(
                            batch_idx,
                            len(test_data_loader),
                        )
                    )
                    continue

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
                        self.args.outdir, filename.format("depth_est", ".pfm")
                    )
                    confidence_filename = os.path.join(
                        self.args.outdir, filename.format("confidence", ".npy")
                    )
                    cam_filename = os.path.join(
                        self.args.outdir, filename.format("cams", "_cam.txt")
                    )
                    img_filename = os.path.join(
                        self.args.outdir, filename.format("images", ".jpg")
                    )
                    os.makedirs(depth_filename.rsplit("/", 1)[0], exist_ok=True)
                    os.makedirs(confidence_filename.rsplit("/", 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit("/", 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit("/", 1)[0], exist_ok=True)
                    # save depth maps
                    save_pfm(depth_filename, depth_est)
                    h, w = depth_est.shape[0], depth_est.shape[1]
                    # save confidence maps
                    if self.args.combine_conf:
                        photometric_confidence = conf_stage4
                        if self.args.save_all_confs:  # only for visualization
                            all_confidence_filename = os.path.join(
                                self.args.outdir,
                                filename.format("confidence_all", ".npy"),
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
                    std = torch.tensor(
                        [0.229, 0.224, 0.225], device=img.device
                    ).reshape((3, 1, 1))
                    mean = torch.tensor(
                        [0.485, 0.456, 0.406], device=img.device
                    ).reshape((3, 1, 1))
                    img = img * std + mean
                    img = img.permute(1, 2, 0).cpu().numpy()
                    # img = np.transpose(img, (1, 2, 0))
                    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
                    write_cam(cam_filename, cam)
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    # print(img.shape)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)

                    if self.args.dataset == "dtu":
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

        if len(times) > 0:
            print("average time: ", sum(times) / len(times))
        else:
            print("average time: 0")
        if self.args.dataset == "dtu":
            valid_metrics = valid_metrics.mean()
            with open(os.path.join(self.args.outdir, "depth_metric.txt"), "w") as w:
                for k in valid_metrics:
                    w.write(k + " " + str(valid_metrics[k]) + "\n")
        torch.cuda.empty_cache()
        gc.collect()
        return True

    def detect(self, data):
        return

    def detectImageFolder(self, image_folder_path, run_name: Union[str, None]=None):
        if run_name is not None:
            self.args.outdir += run_name
        image_folder_name = "mvs"
        self.save_depth([image_folder_name])

        gipuma_filter(
            [image_folder_name],
            self.args.outdir,
            self.args.prob_threshold,
            self.args.disp_threshold,
            self.args.num_consistent,
            self.args.fusibile_exe_path,
        )
        return True
