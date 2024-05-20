from mvs_former.Module.detector import Detector


def demo():
    image_folder_path = "/home/chli/chLi/Dataset/MVSFormer/DTU/dtu_testing/dtu/scan1/"
    image_folder_path = (
        "/home/chli/github/NeRF/colmap-manage/output/3vjia_simple/mvs/mvs/"
    )
    run_name = "3vjia_simple"
    run_name = None
    detector = Detector()
    detector.detectImageFolder(image_folder_path, run_name)
    return True
