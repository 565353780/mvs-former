import os

import numpy as np
from torch.utils.data import Dataset

from mvs_former.Method.io import (
    read_camera_parameters,
    read_img,
    read_pair_file,
)
from mvs_former.Method.data_io import read_pfm


class TTDataset(Dataset):
    def __init__(self, pair_folder, scan_folder, n_src_views=10):
        super(TTDataset, self).__init__()
        pair_file = os.path.join(pair_folder, "pair.txt")
        self.scan_folder = scan_folder
        self.pair_data = read_pair_file(pair_file)
        self.n_src_views = n_src_views

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        id_ref, id_srcs = self.pair_data[idx]
        id_srcs = id_srcs[: self.n_src_views]

        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(self.scan_folder, "cams/{:0>8}_cam.txt".format(id_ref))
        )
        ref_cam = np.zeros((2, 4, 4), dtype=np.float32)
        ref_cam[0] = ref_extrinsics
        ref_cam[1, :3, :3] = ref_intrinsics
        ref_cam[1, 3, 3] = 1.0
        # load the reference image
        ref_img = read_img(
            os.path.join(self.scan_folder, "images/{:0>8}.jpg".format(id_ref))
        )
        ref_img = ref_img.transpose([2, 0, 1])
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(
            os.path.join(self.scan_folder, "depth_est/{:0>8}.pfm".format(id_ref))
        )[0]
        ref_depth_est = np.array(ref_depth_est, dtype=np.float32)
        # load the photometric mask of the reference view
        # confidence = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(id_ref)))[0]
        conf_path = os.path.join(
            self.scan_folder, "confidence/{:0>8}.npy".format(id_ref)
        )
        if not os.path.exists(conf_path):
            conf_path = os.path.join(
                self.scan_folder, "confidence_v2/{:0>8}.npy".format(id_ref)
            )
        confidence = np.load(conf_path)
        if not args.combine_conf:
            confidence = np.array(confidence, dtype=np.float32).transpose([2, 0, 1])

        src_depths, src_confs, src_cams = [], [], []
        for ids in id_srcs:
            if not os.path.exists(
                os.path.join(self.scan_folder, "cams/{:0>8}_cam.txt".format(ids))
            ):
                continue
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(self.scan_folder, "cams/{:0>8}_cam.txt".format(ids))
            )
            src_proj = np.zeros((2, 4, 4), dtype=np.float32)
            src_proj[0] = src_extrinsics
            src_proj[1, :3, :3] = src_intrinsics
            src_proj[1, 3, 3] = 1.0
            src_cams.append(src_proj)
            # the estimated depth of the source view
            src_depth_est = read_pfm(
                os.path.join(self.scan_folder, "depth_est/{:0>8}.pfm".format(ids))
            )[0]
            src_depths.append(np.array(src_depth_est, dtype=np.float32))
            # src_conf = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(ids)))[0]
            conf_path = os.path.join(
                self.scan_folder, "confidence/{:0>8}.npy".format(ids)
            )
            if not os.path.exists(conf_path):
                conf_path = os.path.join(
                    self.scan_folder, "confidence_v2/{:0>8}.npy".format(ids)
                )
            src_conf = np.load(conf_path)
            if not args.combine_conf:
                src_confs.append(
                    np.array(src_conf, dtype=np.float32).transpose([2, 0, 1])
                )
            else:
                src_confs.append(src_conf)
        src_depths = np.expand_dims(np.stack(src_depths, axis=0), axis=1)
        src_confs = np.stack(src_confs, axis=0)
        src_cams = np.stack(src_cams, axis=0)
        return {
            "ref_depth": np.expand_dims(ref_depth_est, axis=0),
            "ref_cam": ref_cam,
            "ref_conf": confidence,  # np.expand_dims(confidence, axis=0),
            "src_depths": src_depths,
            "src_cams": src_cams,
            "src_confs": src_confs,
            "ref_img": ref_img,
            "ref_id": id_ref,
        }
