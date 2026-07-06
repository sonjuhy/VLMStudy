"""
checkpoints/final_model/vit_imagenet_1k_checkpoint_epoch_60.pth(epoch 60, val_top1 56.00%)에서
이어서 학습을 재개합니다. 단일 GPU(RTX 3080 Laptop) 환경 기준 batch_size/accumulation_steps로 조정했습니다.

실행 방법 (PYTHONIOENCODING=utf-8 필수 - Windows cp949 콘솔에서 em-dash 등 특수문자 로그 시 크래시 방지):
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scripts/run_imagenet_basic_resume.py
"""

from end_to_end.imagenet_ete import imagenet_vit_end_to_end

if __name__ == "__main__":
    imagenet_vit_end_to_end(
        resume_checkpoint="checkpoints/final_model/vit_imagenet_1k_checkpoint_epoch_60.pth",
        batch_size=32,
        accumulation_steps=16,
    )
