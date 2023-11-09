import argparse


def getParserArgs():
    parser = argparse.ArgumentParser(description="Predict depth, filter, and fuse")
    parser.add_argument("--model", default="mvsnet", help="select model")
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument(
        "--config", default=None, type=str, help="config file path (default: None)"
    )

    parser.add_argument("--dataset", default="dtu", help="select dataset")
    parser.add_argument("--testpath", help="testing data dir for some scenes")
    parser.add_argument(
        "--testpath_single_scene", help="testing data path for single scene"
    )
    parser.add_argument("--testlist", help="testing scene list")
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1, help="testing batch size")
    parser.add_argument(
        "--numdepth", type=int, default=192, help="the number of depth values"
    )

    parser.add_argument("--resume", default=None, help="load a specific checkpoint")
    parser.add_argument(
        "--outdir", default="/home/wmlce/mount_194/DTU_MVS_outputs", help="output dir"
    )
    parser.add_argument(
        "--display", action="store_true", help="display depth images and masks"
    )

    parser.add_argument(
        "--share_cr",
        action="store_true",
        help="whether share the cost volume regularization",
    )

    parser.add_argument("--ndepths", type=str, default=None, help="ndepths")
    parser.add_argument(
        "--depth_interals_ratio", type=str, default=None, help="depth_interals_ratio"
    )
    parser.add_argument(
        "--cr_base_chs",
        type=str,
        default="8,8,8",
        help="cost regularization base channels",
    )
    parser.add_argument(
        "--grad_method",
        type=str,
        default="detach",
        choices=["detach", "undetach"],
        help="grad method",
    )
    parser.add_argument(
        "--no_refinement", action="store_true", help="depth refinement in last stage"
    )
    parser.add_argument(
        "--full_res", action="store_true", help="full resolution prediction"
    )

    parser.add_argument(
        "--interval_scale", type=float, required=True, help="the depth interval scale"
    )
    parser.add_argument("--num_view", type=int, default=5, help="num of view")
    parser.add_argument("--max_h", type=int, default=864, help="testing max h")
    parser.add_argument("--max_w", type=int, default=1152, help="testing max w")
    parser.add_argument(
        "--fix_res", action="store_true", help="scene all using same res"
    )
    parser.add_argument("--depth_scale", type=float, default=1.0, help="depth scale")
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="temperature of softmax"
    )

    parser.add_argument("--num_worker", type=int, default=4, help="depth_filer worker")
    parser.add_argument(
        "--save_freq", type=int, default=20, help="save freq of local pcd"
    )

    parser.add_argument(
        "--filter_method",
        type=str,
        default="gipuma",
        choices=["gipuma", "pcd", "dpcd"],
        help="filter method",
    )

    # filter
    parser.add_argument(
        "--prob_threshold", type=str, default="0.5,0.5,0.5,0.5", help="prob confidence"
    )
    parser.add_argument(
        "--thres_view", type=int, default=3, help="threshold of num view"
    )
    parser.add_argument(
        "--thres_disp", type=float, default=1.0, help="threshold of disparity"
    )
    parser.add_argument(
        "--downsample", type=float, default=None, help="downsampling point cloud"
    )

    ## dpcd filter
    parser.add_argument(
        "--dist_base", type=float, default=4.0, help="threshold of disparity"
    )
    parser.add_argument(
        "--rel_diff_base", type=float, default=1300.0, help="downsampling point cloud"
    )

    # filter by gimupa
    parser.add_argument("--fusibile_exe_path", type=str, default="./fusibile/fusibile")
    parser.add_argument("--disp_threshold", type=float, default="0.2")
    parser.add_argument("--num_consistent", type=float, default="3")

    # tank templet
    parser.add_argument("--use_short_range", action="store_true")

    # confidence
    parser.add_argument("--combine_conf", action="store_true")
    parser.add_argument("--tmp", default=1.0, type=float)
    parser.add_argument("--tmps", default=None, type=str)
    parser.add_argument("--save_all_confs", action="store_true")

    return parser
