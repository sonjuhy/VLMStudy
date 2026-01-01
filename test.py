from tests.mnist_vit_test import end_to_end_test


def mnist_video_encoder_test():
    # https://arxiv.org/pdf/2010.11929
    end_to_end_test()


if __name__ == "__main__":
    mnist_video_encoder_test()
