import numpy as np
from PIL import Image


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(
        (4, 4)
    )
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(
        " ".join(lines[7:10]), dtype=np.float32, sep=" "
    ).reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.0
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write("extrinsic\n")
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + " ")
        f.write("\n")
    f.write("\n")

    f.write("intrinsic\n")
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + " ")
        f.write("\n")

    f.write(
        "\n"
        + str(cam[1][3][0])
        + " "
        + str(cam[1][3][1])
        + " "
        + str(cam[1][3][2])
        + " "
        + str(cam[1][3][3])
        + "\n"
    )

    f.close()
