from enum import Enum
from dotenv import load_dotenv

import os

load_dotenv(".env")
DATASET_ROOT_PATH = os.getenv("DATASET_ROOT_PATH")


class JSONPathEnum(Enum):
    LLAVA_1_5_MIX665K = os.path.join(
        DATASET_ROOT_PATH, "VLMDatasets", "LlavaJson", "llava_v1_5_mix665k.json"
    )
    LLAVA_1_5_MIX665K_CLEAN = os.path.join(
        DATASET_ROOT_PATH, "VLMDatasets", "LlavaJson", "llava_v1_5_mix665k_clean.json"
    )
    LLAVA_INSTRUCT_150K = os.path.join(
        DATASET_ROOT_PATH, "VLMDatasets", "LlavaJson", "llava_instruct_150k.json"
    )
    BLIP_LAION_CC_SBU_558K = os.path.join(
        DATASET_ROOT_PATH, "VLMDatasets", "LlavaJson", "blip_laion_cc_sbu_558k.json"
    )


class ImagePathEnum(Enum):
    IMAGE_NET_1K = os.path.join(DATASET_ROOT_PATH, "VLMDatasets", "ImageNet1K")
    BLIP_LAION_CC_SUB_558K = os.path.join(
        DATASET_ROOT_PATH, "VLMDatasets", "images", "558_images"
    )
    COCO_SET = os.path.join(DATASET_ROOT_PATH, "VLMDatasets", "images", "coco")
    LLAVA_ALL_IMAGES = os.path.join(DATASET_ROOT_PATH, "VLMDatasets", "LlavaImages")


class CheckPointPathEnum(Enum):
    VIT_IMAGENET_1K_CHECKPOINT = os.path.join(
        "checkpoints", "final_model", "vit_imagenet_1k_checkpoint_epoch_99.pth"
    )
    SOLAR_PROJECTOR_STAGE_2 = os.path.join(
        "checkpoints", "vlm", "stage2", "solor_projector_epoch_1.pth"
    )
    VLM_FINAL_STAGE_3 = os.path.join(
        "checkpoints", "vlm", "stage3", "final_model", "pytorch_model.bin"
    )
