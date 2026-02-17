from end_to_end.mnist_ete import mnist_vit_end_to_end, mnist_vlm_end_to_end
from utils.utils import timer_call

import argparse


class ViTRunning:
    def __init__(self):
        pass

    def mnist_vit_end_to_end(self):
        with timer_call():
            mnist_vit_end_to_end()

    def img_1k_vit_end_to_end(self):
        with timer_call():
            mnist_vlm_end_to_end()

    def depth_vit_end_to_end(self):
        with timer_call():
            pass


class VLMRunning:
    def __init__(self):
        pass

    def mnist_vlm_end_to_end(self):
        with timer_call():
            pass

    def img_1k_vlm_end_to_end(self):
        with timer_call():
            pass

    def depth_vlm_end_to_end(self):
        with timer_call():
            pass


if __name__ == "__main__":
    # vit, vlm 선택 옵션
    # mnist, img_1k, depth 선택 옵션

    pass
