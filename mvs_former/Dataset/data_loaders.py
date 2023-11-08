import numpy as np
from torch.utils.data import DataLoader

from mvs_former.Dataset.general_eval import MVSDataset
from mvs_former.Dataset.dtu_mvs import DTUMVSDataset
from mvs_former.Dataset.dtu_mvs_ms import DTUMVSMSDataset
from mvs_former.Dataset.blended_mvs import BlendedMVSDataset
from mvs_former.Dataset.blended_mvs_ms import BlendedMVSMSDataset

np.random.seed(1234)


class DTULoader(DataLoader):
    def __init__(
        self,
        data_path,
        data_list,
        mode,
        num_srcs,
        num_depths,
        interval_scale=1.0,
        shuffle=True,
        seq_size=49,
        batch_size=1,
        fix_res=False,
        max_h=None,
        max_w=None,
        dataset_eval="dtu",
        refine=True,
        num_workers=4,
        crop=False,
        iterative=False,
        augment=False,
        aug_args=None,
        height=None,
        width=None,
        data_set_type="stage4",
        **kwargs,
    ):
        if (mode == "train") or (mode == "val"):
            if data_set_type == "stage4":
                dataset = DTUMVSDataset
            elif data_set_type == "multi_scale":
                dataset = DTUMVSMSDataset
            else:
                raise NotImplementedError
            self.mvs_dataset = dataset(
                data_path,
                data_list,
                mode,
                num_srcs,
                num_depths,
                interval_scale,
                shuffle=shuffle,
                seq_size=seq_size,
                batch_size=batch_size,
                crop=crop,
                augment=augment,
                aug_args=aug_args,
                refine=refine,
                height=height,
                width=width,
                **kwargs,
            )
        else:
            self.mvs_dataset = MVSDataset(
                data_path,
                data_list,
                mode,
                num_srcs,
                num_depths,
                interval_scale,
                shuffle=shuffle,
                seq_size=seq_size,
                batch_size=batch_size,
                max_h=max_h,
                max_w=max_w,
                fix_res=fix_res,
                dataset=dataset_eval,
                refine=refine,
                iterative=iterative,
                **kwargs,
            )
        drop_last = True if mode == "train" else False
        super().__init__(
            self.mvs_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)


class BlendedLoader(DataLoader):
    def __init__(
        self,
        data_path,
        data_list,
        mode,
        num_srcs,
        num_depths,
        interval_scale=1.0,
        shuffle=True,
        seq_size=49,
        batch_size=1,
        fix_res=False,
        max_h=None,
        max_w=None,
        refine=True,
        num_workers=4,
        crop=False,
        iterative=False,
        augment=False,
        aug_args=None,
        height=None,
        width=None,
        data_set_type="stage4",
        **kwargs,
    ):
        if (mode == "train") or (mode == "val"):
            if data_set_type == "stage4":
                dataset = BlendedMVSDataset
            if data_set_type == "multi_scale":
                dataset = BlendedMVSMSDataset
            else:
                raise NotImplementedError
            self.mvs_dataset = dataset(
                data_path,
                data_list,
                mode,
                num_srcs,
                num_depths,
                interval_scale,
                shuffle=shuffle,
                seq_size=seq_size,
                batch_size=batch_size,
                crop=crop,
                augment=augment,
                aug_args=aug_args,
                refine=refine,
                height=height,
                width=width,
                **kwargs,
            )
        else:
            raise NotImplementedError
        drop_last = True if mode == "train" else False
        super().__init__(
            self.mvs_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)
