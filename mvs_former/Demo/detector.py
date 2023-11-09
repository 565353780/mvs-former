from mvs_former.Module.detector import Detector


def demo():
    image_folder_path = "/home/chli/chLi/Dataset/MVSFormer/DTU/dtu_testing/dtu/scan1/"
    detector = Detector()
    detector.detectImageFolder(image_folder_path)
    return True
