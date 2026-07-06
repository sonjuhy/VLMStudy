"""
VQA 정확도 벤치마크 (VQAv2 / GQA / POPE)

실행 예시:
    # VQAv2
    python -m projector_study.eval.benchmark_runner \\
        --dataset vqav2 \\
        --ann_path /data/vqav2/v2_mscoco_val2014_annotations.json \\
        --ques_path /data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \\
        --img_root /data/vqav2/val2014 \\
        --llm_id Qwen/Qwen2.5-7B-Instruct \\
        --proj_ckpt projector_study/checkpoints/mlp_2024-07-05/best_projector.pth \\
        --projector mlp

    # POPE
    python -m projector_study.eval.benchmark_runner \\
        --dataset pope \\
        --ann_path /data/pope/pope_adversarial.json \\
        --img_root /data/coco/val2014 \\
        --llm_id Qwen/Qwen2.5-7B-Instruct \\
        --projector mlp

    # GQA
    python -m projector_study.eval.benchmark_runner \\
        --dataset gqa \\
        --ann_path /data/gqa/testdev_balanced_questions.json \\
        --img_root /data/gqa/images \\
        --llm_id Qwen/Qwen2.5-7B-Instruct \\
        --projector mlp
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projector_study.models.base_vlm import ProjectorVLM
from projector_study.models.projectors import PROJECTOR_REGISTRY
from vision.vit_model import ViTEncoder


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

class VQAv2Dataset(Dataset):
    """
    VQAv2 val2014.
    ann_path  : v2_mscoco_val2014_annotations.json
    ques_path : v2_OpenEnded_mscoco_val2014_questions.json
    img_root  : val2014/ (COCO 이미지 폴더)
    """
    def __init__(self, ann_path: str, ques_path: str, img_root: str,
                 transform=VIT_TRANSFORM):
        with open(ann_path) as f:
            anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
        with open(ques_path) as f:
            questions = json.load(f)["questions"]

        self.samples = []
        for q in questions:
            qid  = q["question_id"]
            ann  = anns.get(qid)
            if ann is None:
                continue
            # 최다 득표 답변 사용
            answers = [a["answer"] for a in ann["answers"]]
            gt = max(set(answers), key=answers.count)
            img_file = f"COCO_val2014_{q['image_id']:012d}.jpg"
            self.samples.append({
                "question_id": qid,
                "image_path":  os.path.join(img_root, img_file),
                "question":    q["question"],
                "answer":      gt,
            })
        self.transform = transform

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return {
            "image":       img,
            "question":    s["question"],
            "answer":      s["answer"],
            "question_id": s["question_id"],
        }


class GQADataset(Dataset):
    """
    GQA testdev_balanced.
    ann_path : testdev_balanced_questions.json  {qid: {imageId, question, answer, ...}}
    img_root : gqa/images/
    """
    def __init__(self, ann_path: str, img_root: str, transform=VIT_TRANSFORM):
        with open(ann_path) as f:
            data = json.load(f)
        self.samples = [
            {
                "question_id": qid,
                "image_path":  os.path.join(img_root, f"{v['imageId']}.jpg"),
                "question":    v["question"],
                "answer":      v["answer"],
            }
            for qid, v in data.items()
        ]
        self.transform = transform

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return {
            "image":       img,
            "question":    s["question"],
            "answer":      s["answer"],
            "question_id": s["question_id"],
        }


class POPEDataset(Dataset):
    """
    POPE (adversarial / popular / random 중 하나).
    ann_path: JSONL 파일. 각 줄: {"image": "xxx.jpg", "text": "Is there a ...", "label": "yes"/"no"}
    img_root: COCO val2014 폴더
    """
    def __init__(self, ann_path: str, img_root: str, transform=VIT_TRANSFORM):
        self.samples = []
        with open(ann_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                self.samples.append({
                    "image_path": os.path.join(img_root, obj["image"]),
                    "question":   obj["text"],
                    "answer":     obj["label"].strip().lower(),  # "yes" / "no"
                })
        self.transform = transform

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return {"image": img, "question": s["question"], "answer": s["answer"]}


DATASET_MAP = {"vqav2": VQAv2Dataset, "gqa": GQADataset, "pope": POPEDataset}


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_vlm(args: argparse.Namespace, device: torch.device) -> tuple[ProjectorVLM, AutoTokenizer]:
    vit = ViTEncoder(
        img_size=224, patch_size=16, embedding_size=768,
        num_class=1000, num_heads=12, in_channels=3,
    )
    if args.vit_ckpt and os.path.exists(args.vit_ckpt):
        ckpt = torch.load(args.vit_ckpt, map_location="cpu")
        vit.load_state_dict(ckpt.get("model_state_dict", ckpt))

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ProjCls   = PROJECTOR_REGISTRY[args.projector]
    projector = ProjCls(v_dim=768, l_dim=llm.config.hidden_size)

    if args.proj_ckpt and os.path.exists(args.proj_ckpt):
        saved = torch.load(args.proj_ckpt, map_location="cpu")
        projector.load_state_dict(saved.get("state_dict", saved))

    model = ProjectorVLM(vit=vit, projector=projector, llm=llm).to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 프롬프트 구성 및 답변 정규화
# ---------------------------------------------------------------------------

def build_prompt(question: str, dataset: str) -> str:
    if dataset == "pope":
        return (f"<image>\n{question}\nAnswer with yes or no.")
    return (f"<image>\n{question}\nAnswer the question using a single word or phrase.")


def normalize_answer(raw: str) -> str:
    """생성 텍스트에서 답변 부분만 추출, 소문자 정규화."""
    # "Assistant:" / "Answer:" 이후 텍스트만 취함
    for marker in ["assistant:", "answer:", "response:"]:
        if marker in raw.lower():
            raw = raw.lower().split(marker, 1)[-1]
            break
    # 첫 문장/단어만
    raw = re.split(r"[.\n,;]", raw.strip())[0].strip().lower()
    return raw


# ---------------------------------------------------------------------------
# 정확도 / F1 계산
# ---------------------------------------------------------------------------

def compute_vqa_accuracy(preds: list[dict]) -> float:
    """VQAv2 soft accuracy: 정답이 3명 이상 동의하면 1.0, 그 미만은 min(count/3, 1.0)"""
    correct = sum(1 for p in preds if p["pred"] == p["gt"])
    return 100.0 * correct / len(preds)


def compute_pope_f1(preds: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for p in preds:
        pred_yes = p["pred"].startswith("yes")
        gt_yes   = p["gt"] == "yes"
        if pred_yes and gt_yes:     tp += 1
        elif pred_yes and not gt_yes: fp += 1
        elif not pred_yes and gt_yes: fn += 1
        else:                        tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(preds) if preds else 0.0
    return {"accuracy": round(100 * accuracy, 2),
            "precision": round(100 * precision, 2),
            "recall": round(100 * recall, 2),
            "f1": round(100 * f1, 2)}


# ---------------------------------------------------------------------------
# 추론 루프
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: ProjectorVLM,
    tokenizer: AutoTokenizer,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
    max_new_tokens: int = 32,
) -> list[dict]:
    results = []

    for batch in tqdm(loader, desc=f"Inference [{dataset_name}]"):
        images = batch["image"].to(device, dtype=torch.bfloat16)  # (B, 3, 224, 224)
        B = images.size(0)

        for i in range(B):
            question = batch["question"][i]
            gt       = batch["answer"][i]
            prompt   = build_prompt(question, dataset_name)

            tok = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True,
                truncation=True, max_length=256,
            ).to(device)

            out_ids = model.generate(
                images=images[i:i+1],
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            raw  = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            pred = normalize_answer(raw)

            results.append({"pred": pred, "gt": gt.lower()})

    return results


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 데이터셋 구성
    DsCls = DATASET_MAP[args.dataset]
    ds_kwargs = {"ann_path": args.ann_path, "img_root": args.img_root}
    if args.dataset == "vqav2":
        ds_kwargs["ques_path"] = args.ques_path
    dataset = DsCls(**ds_kwargs)

    if args.max_samples and args.max_samples < len(dataset):
        indices = list(range(args.max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    model, tokenizer = load_vlm(args, device)

    results = run_inference(model, tokenizer, loader, device,
                            args.dataset, max_new_tokens=args.max_new_tokens)

    # 지표 계산
    if args.dataset == "pope":
        metrics = compute_pope_f1(results)
        print(f"\n  POPE Results | {args.projector}")
        for k, v in metrics.items():
            print(f"    {k:<12}: {v:.2f}")
    else:
        acc = compute_vqa_accuracy(results)
        metrics = {"accuracy": round(acc, 2)}
        print(f"\n  {args.dataset.upper()} Accuracy | {args.projector}: {acc:.2f}%")

    # 결과 저장
    out_dir = Path("projector_study/results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dataset}_{args.projector}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in metrics.items():
            writer.writerow({"metric": k, "value": v})
        writer.writerow({"metric": "n_samples", "value": len(results)})

    print(f"  결과 저장: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VQA benchmark runner")
    p.add_argument("--dataset",    required=True, choices=["vqav2", "gqa", "pope"])
    p.add_argument("--projector",  required=True, choices=list(PROJECTOR_REGISTRY.keys()))
    p.add_argument("--llm_id",     required=True)
    p.add_argument("--ann_path",   required=True, help="어노테이션 JSON 경로")
    p.add_argument("--img_root",   required=True, help="이미지 루트 폴더")

    # VQAv2 전용
    p.add_argument("--ques_path",  default=None, help="VQAv2 questions JSON")

    p.add_argument("--proj_ckpt",  default=None,
                   help="projector 가중치 .pth (없으면 랜덤 초기화)")
    p.add_argument("--vit_ckpt",
                   default="checkpoints/final_model/best_vit_imagenet_1k.pth")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_samples",    type=int, default=None,
                   help="빠른 검증용 샘플 수 제한 (None=전체)")
    p.add_argument("--gpu",            type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
