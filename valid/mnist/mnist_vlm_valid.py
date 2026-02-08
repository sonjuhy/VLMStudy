from vision.mnist_vit_model import MNISTViTEncoder, MNISTViTHyperClovaX

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import datasets, transforms
import torch


def valid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

    ##########DataSet###########
    # 1. 이미지 전처리 정의 (학습 시와 동일해야 함)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 2. 데이터셋에서 샘플 하나 추출
    test_ds = datasets.MNIST(root="./datasets/mnist", train=False, download=True)
    sample_img, sample_label = test_ds[0]  # 첫 번째 데이터 (숫자 7)

    # 3. 텐서 변환 및 배치 차원 추가 [1, 1, 28, 28]
    input_tensor = transform(sample_img).unsqueeze(0)

    ##############Inference###################
    # 1. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 학습 시와 동일한 구조의 ViT 선언
    vit = MNISTViTEncoder(
        embedding_size=768,
        img_size=28,
        patch_size=7,
        return_token_type=True,
        num_heads=12,
    ).to(device)

    # VLM 클래스
    vlm_model = MNISTViTHyperClovaX(vit, llm).to(device)

    # 2. 학습된 가중치 로드
    vlm_model.vit.load_state_dict(
        torch.load("vit_768_multi_prompt_mnist.pth", map_location=device)
    )
    vlm_model.projector.load_state_dict(
        torch.load("projector_multi_prompt_768_to_1024.pth", map_location=device)
    )
    vlm_model.eval()

    # 3. 추론용 프롬프트 구성 (답변 직전까지만 입력)
    # prompt = "질문: 이 이미지에 있는 숫자는 무엇인가요?\n답변: 이 숫자는"
    prompt = "질문: 이 이미지에 있는 숫자가 무엇인가요?\n답변: 이 숫자는"
    # prompt = "질문: 이 숫자가 7이 맞니?"
    prompt = "질문: 이 숫자가 5가 맞니?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 4. 생성 (Inference)
    with torch.no_grad():
        target_device_type = "cpu" if device == "cpu" else "cuda"
        with torch.autocast(device_type=target_device_type, dtype=torch.bfloat16):
            # 이미지 특징 추출 및 프로젝션
            visual_tokens = vlm_model.vit(input_tensor.to(device))
            image_embeds = vlm_model.projector(visual_tokens)

            # 텍스트 임베딩 추출
            text_embeds = vlm_model.llm.get_input_embeddings()(inputs["input_ids"])

            # [이미지 임베딩 + 텍스트 임베딩] 결합
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

            output_ids = vlm_model.llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=20,  # " 5입니다." 정도만 생성하면 되므로 짧게 설정
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # 5. 결과 디코딩
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f'프롬프트 내용 : "{prompt}"')
    print(f"--- 추론 결과 ---")
    if sample_label is not None:
        print(f"실제 정답: {sample_label}")
    print(f"모델 답변: {result}")


# --------------------------------------------------
# --- 추론 결과 ---
# 실제 정답: 7
# 모델 답변:  7입니다.


# 학습하지 않은 프롬프트 양식
# 프롬프트 내용 : prompt = "질문: 이미지에 쓰여진 숫자가 뭐니?\n답변: 이 숫자는"
# --- 추론 결과 ---
# 실제 정답: 7
# 모델 답변:  7입니다.
# 학습하지 못한 단어들이지만 HyperCLOVA X의 언어 이해 능력 덕분에 정답 도출

# 프롬프트 내용 : prompt = "질문: 이 숫자가 7이 맞니?\n답변: 이 숫자는"
# --- 추론 결과 ---
# 실제 정답: 7
# 모델 답변:  7입니다.
# 학습하지 못한 패턴이라 고정된 패턴의 답만 제출
