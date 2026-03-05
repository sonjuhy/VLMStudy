from end_to_end.mnist_ete import mnist_vit_end_to_end, mnist_vlm_end_to_end
from end_to_end.depth_ete import depth_vlm_end_to_end
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

    def depth_vlm_end_to_end(
        self,
        train_continue: bool = False,
        start_epoch: int = 0,
        end_epoch: int = 100,
    ):
        with timer_call():
            depth_vlm_end_to_end(
                train_continue=train_continue,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            )


if __name__ == "__main__":
    # vit, vlm 선택 옵션
    # mnist, img_1k, depth 선택 옵션

    VLMRunning().depth_vlm_end_to_end(train_continue=True, start_epoch=30, end_epoch=65)
