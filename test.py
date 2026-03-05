from train.depth.depth_vit_test import depth_end_to_end_test, depth_valid_test

# from tests.mnist_vit_test import end_to_end_test
# from tests.mnist_vlm_test import train as mnist_vlm_train
# from tests.mnist_vlm_test import valid as mnist_vlm_valid
# from tests.imagenet_1k_test import imagenet_1k_end_to_end_test


# def mnist_video_encoder_test():
#     # https://arxiv.org/pdf/2010.11929
#     end_to_end_test()


# def mnist_vlm_hyper_clova_x_test():
#     mnist_vlm_train(epochs=5)
#     mnist_vlm_valid()


# def imagenet_1k_vit_encoder_test():
#     imagenet_1k_end_to_end_test()


def depth_imagenet_1k_vit_encoder_test():
    depth_end_to_end_test(train_continue=True, start_epoch=0, end_epoch=10)
    # depth_valid_test(pth_file_path="checkpoints/best_vit_depth.pth")


if __name__ == "__main__":
    # mnist_video_encoder_test()
    # mnist_vlm_hyper_clova_x_test()
    # imagenet_1k_vit_encoder_test()
    depth_imagenet_1k_vit_encoder_test()
