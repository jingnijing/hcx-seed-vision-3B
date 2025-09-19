'''텍스트 + 이미지 SFT'''


import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from modeling_hyperclovax import HCXVisionForCausalLM
from processing_hyperclovax import HCXProcessor
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import json
import pandas as pd
from collections.abc import Sequence
from torch.nn.utils.rnn import pad_sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
processor = None

def split_and_save_validation_data(dataset, output_dir):
        """Train 데이터에서 10%를 validation으로 분할하고 정보 저장"""
        
        logger.info("Splitting dataset into train/validation...")
        
        # 감정 라벨 추출 (stratify용)
        emotions = []
        for i in range(len(dataset)):
            sample = dataset[i]
            for msg in sample['chat']:
                if msg['role'] == 'assistant' and isinstance(msg['content'], dict):
                    emotions.append(msg['content'].get('text', ''))
        
        # Train/Validation 분할 (90:10)
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=0.1,
            stratify=emotions,
            random_state=42
        )
        
        # 데이터셋 분할
        train_dataset = dataset.select(train_indices)
        val_dataset = dataset.select(val_indices)
        
        # Validation 데이터 정보 수집
        val_data_info = []
        for idx in val_indices:
            sample = dataset[idx]
            val_info = {
                "original_index": idx,
                "image_path": None,
                "text": None,
                "emotion": None
            }
            
            # chat에서 정보 추출
            for msg in sample['chat']:
                content = msg['content']
                if isinstance(content, dict):
                    if content.get("type") == "image":
                        val_info["image_path"] = content.get("image")
                    elif content.get("type") == "text" and msg["role"] == "user":
                        val_info["text"] = content.get("text")
                    elif content.get("type") == "text" and msg["role"] == "assistant":
                        val_info["emotion"] = content.get("text")
            
            val_data_info.append(val_info)
        
        # JSON 파일로 저장
        val_json_path = f"{output_dir}/validation_data_info.json"
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_count": len(val_indices),
                "train_count": len(train_indices),
                "emotion_distribution": {
                    emotion: sum(1 for d in val_data_info if d['emotion'] == emotion)
                    for emotion in set(emotions)
                },
                "samples": val_data_info
            }, f, ensure_ascii=False, indent=2)
        
        # CSV로도 저장
        val_csv_path = f"{output_dir}/validation_data_list.csv"
        pd.DataFrame(val_data_info).to_csv(val_csv_path, index=False, encoding='utf-8')
        
        logger.info(f"Dataset split complete:")
        logger.info(f"   - Train: {len(train_indices)} samples")
        logger.info(f"   - Validation: {len(val_indices)} samples")
        logger.info(f"   - Validation info saved to: {val_json_path}")
        logger.info(f"   - Validation list saved to: {val_csv_path}")
        
        return train_dataset, val_dataset

    
def custom_collate_fn(batch):
    input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
    labels = [torch.tensor(sample["labels"]) for sample in batch]

    pixel_values_images = []
    for sample in batch:
        image_data = sample["pixel_values_images"]
        
        # image_data가 텐서가 아니면 강제로 tensor로 변환
        if not isinstance(image_data, torch.Tensor):
            image_tensor = torch.tensor(image_data, dtype=torch.float32)
        else:
            image_tensor = image_data
        
        # [C, H, W] 또는 [3, 378, 378] 형식이 아니라면 reshape
        if image_tensor.ndim > 4:
            image_tensor = image_tensor.squeeze()
        print(f"image_tensor shape: {image_tensor.shape}, type: {type(image_tensor)}")
        
        pixel_values_images.append(image_tensor)

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        "pixel_values_images": torch.stack(pixel_values_images),  # → [B, C, H, W]
    }

def debug_masking_for_one_sample(sample, formatting_func, tokenizer, output_path="debug_masking_log.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== 마스킹 디버깅 시작 ===\n")

        processed = formatting_func(sample)

        input_ids = processed["input_ids"]
        labels = processed["labels"]

        f.write(f"\n chat:\n{json.dumps(sample['chat'], ensure_ascii=False, indent=2)}\n")
        f.write(f"\n 전체 input_ids 길이: {len(input_ids)}\n")
        f.write(f" 전체 labels 길이: {len(labels)}\n")

        f.write(" 상위 50개 input_id와 label 비교:\n")
        f.write(f"{'idx':<5} | {'token_id':<10} | {'token_str':<20} | {'label':<6}\n")
        f.write("-" * 50 + "\n")
        
        for i in range(min(50, len(input_ids))):
            token_id = input_ids[i].item() if isinstance(input_ids[i], torch.Tensor) else input_ids[i]
            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            try:
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                if not token_str.strip():
                    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            except:
                token_str = str(token_id)
            label_str = str(label_val) if label_val != -100 else "—"
            f.write(f"{i:<5} | {token_id:<10} | {token_str:<20} | {label_str:<6}\n")

        f.write("\n 마지막 30개 input_id와 label 비교:\n")
        f.write(f"{'idx':<5} | {'token_id':<10} | {'token_str':<20} | {'label':<6}\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(input_ids) - 30, len(input_ids)):
            token_id = input_ids[i].item() if isinstance(input_ids[i], torch.Tensor) else input_ids[i]
            label_val = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            try: 
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                if not token_str.strip():
                    token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            except:
                token_str = str(token_id)
            label_str = str(label_val) if label_val != -100 else "—"
            f.write(f"{i:<5} | {token_id:<10} | {token_str:<20} | {label_str:<6}\n")
            
        masked_text = processor.tokenizer.decode([id for id in labels if id != -100], skip_special_tokens=False)
        print(f"\n 마스킹된 assistant 응답 (디코딩): {masked_text}")
        f.write(f"\n 마스킹된 assistant 응답 (디코딩): {masked_text}\n")
        f.write("\n 마스킹 디버깅 완료. `-100`은 손실 계산에서 제외된 부분입니다.\n")

    
def preprocess(example):
    global processor

    if processor is None:
        raise RuntimeError("processor is not initialized!")

    try:
        if not hasattr(preprocess, 'counter'):
            preprocess.counter = 0
        sample_idx = preprocess.counter
        preprocess.counter += 1

        # 메시지 파싱
        image_path, user_text, assistant_text = None, "", ""
        for msg in example["chat"]:
            content = msg.get("content", {})
            if isinstance(content, dict):
                if content.get("type") == "image":
                    image_path = content.get("image")
                elif content.get("type") == "text":
                    if msg["role"] == "user":
                        user_text = content["text"]
                    elif msg["role"] == "assistant":
                        assistant_text = content["text"]

        # 전체 chat template 텍스트 구성
        conversations = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "제시된 이미지와 텍스트를 함께 고려하여 인물의 감정을 판단하십시오. 응답은 반드시 아래 여섯 가지 중 하나여야 합니다 : 행복, 슬픔, 분노, 공포, 놀람, 중립."
                    }
                ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}]
            }
        ]

        # 1. Chat template 적용
        full_text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)

        # 2. 전체 input_ids 토크나이즈 (모델 입력용 X, 마스킹 위치확인용 토크나이징)
        input_ids = processor.tokenizer(full_text, return_tensors="pt")["input_ids"][0]
        input_ids_list = input_ids.tolist()
        input_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids_list)

        # 3. 마스킹을 위한 시작 위치 찾기
        start_idx = -1
        for i in range(len(input_tokens)):
            if input_tokens[i] == '<|im_start|>' and i + 1 < len(input_tokens) and input_tokens[i+1] == 'assistant':
                start_idx = i + 2  # 응답 시작 위치
                break
        if start_idx == -1:
            raise ValueError("'<|im_start|>assistant' 시퀀스를 토큰에서 찾을 수 없습니다.")

        # 4. 종료 위치 (<|im_end|>) 찾기
        end_idx = -1
        for i in range(start_idx, len(input_tokens)):
            if input_tokens[i] == '<|im_end|>':
                end_idx = i
                break
        if end_idx == -1:
            raise ValueError("'<|im_end|>' 토큰을 찾을 수 없습니다.")

        # 5. 마스킹: assistant 응답 전체 + <|im_end|>까지 포함 / 실질적으로 마스킹을 할 수 있는 labels 생성해둠
        labels = [-100] * len(input_ids)
        labels[start_idx:end_idx + 1] = input_ids_list[start_idx:end_idx + 1]

        # 6. 이미지 전처리
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            
            # ✅ 디버깅 코드 삽입
            print(f"image 타입 확인: {type(image)}")
            print(f"processor.image_processor: {getattr(processor, 'image_processor', None)}")
            print(f"processor.feature_extractor: {getattr(processor, 'feature_extractor', None)}")
            
            inputs = processor(text=full_text, images=[[image]], return_tensors="pt")
            
            # #이미지 텐서만 명시적으로 추출하여 tensor 유지
            # if "pixel_values_images" in inputs:
            #     pixel_tensor = inputs["pixel_values_images"][0][0]  # [[tensor]] -> tensor
            #     inputs["pixel_values_images"] = pixel_tensor

            print("inputs keys :: ", inputs.keys())
        else:
            inputs = processor(text=full_text, return_tensors="pt")

        # squeeze는 text tensor에만 적용
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and k != "pixel_values_images":
                result[k] = v.squeeze(0) if v.dim() > 1 else v
            else:
                result[k] = v

        result["labels"] = torch.tensor(labels)

        # 7. 디버깅 로그 (샘플 1개만)
        if sample_idx == 0:
            print(f"\n샘플 {sample_idx}")
            print(f"   User: {user_text}")
            print(f"   Assistant: {assistant_text}")
            print(f"   마스킹된 토큰 디코딩 (skip_special_tokens=False): {processor.tokenizer.decode([i for i in labels if i != -100], skip_special_tokens=False)}")
            print(f"   input 길이: {len(input_ids)}, 응답 길이: {end_idx - start_idx + 1}")

            # 마스킹된 토큰 문자열 출력
            masked_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids_list[start_idx:end_idx + 1])
            print(f"   마스킹된 토큰 시퀀스 (토큰 문자열): {masked_tokens}")

            # 마스킹 확인
            contains_im_end = '<|im_end|>' in masked_tokens
            print(f" 마스킹된 labels에 <|im_end|> 포함 여부: {contains_im_end}")

            print("\n 마스킹된 구간 토큰 (start_idx ~ end_idx):")
            for i in range(start_idx, end_idx + 1):
                tok_id = input_ids[i].item()
                decoded = processor.tokenizer.decode([tok_id], skip_special_tokens=False)
                label_val = labels[i]
                print(f"{i:<3} | ID: {tok_id:<6} | Token: {decoded!r:<10} | Label: {label_val}")

        return result

    except Exception as e:
        print(f"\n오류: {str(e)}")
        raise e


def main():
    model_path = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
    jsonl_path = "/home/jieun/workspace/clovax/finetune/test_20.jsonl"
    output_dir = "/home/jieun/workspace/clovax/finetune/output/finetuned_hcx_sft_test_masking"

    # Load processor
    global processor
    processor = HCXProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    # image_processor = processor.image_processor
    
    # Fix special token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = HCXVisionForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # 모델과 토치가 어느 GPU를 사용하는지 확인
    if torch.cuda.is_available():
        print("CUDA 사용 가능")
        print("현재 사용 중인 GPU ID:", torch.cuda.current_device())
        print("현재 GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("총 GPU 메모리:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
        print("모델이 할당된 디바이스:", next(model.parameters()).device)
    else:
        print("GPU 사용 불가. CPU에서 실행 중")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 1. 원본 데이터셋 로딩 (Load and format dataset)
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    
    debug_masking_for_one_sample(
        sample=dataset[0],  # 디버깅할 샘플 1개
        formatting_func=preprocess,
        tokenizer=processor.tokenizer,
        output_path="debug_masking_log.txt"
    )
    
    sample = dataset[0]  # 첫 번째 샘플 확인
    
    # 2. Train/Validation split 및 저장
    train_raw, val_raw = split_and_save_validation_data(dataset, output_dir)

    train_dataset = train_raw.map(
        preprocess,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
        desc="이미지 처리 중"
    )
    
    val_dataset = val_raw.map(
        preprocess,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
        desc="이미지 처리 중"
    )

    # 확인
    print(f"처리된 데이터셋 크기: {len(dataset)}")
    print(f"첫 번째 샘플 키: {list(dataset[0].keys())}")
    
    # 샘플 하나 가져오기
    sample = train_dataset[0]

    # print("sample['pixel_values_images'].shape:", sample["pixel_values_images"].shape)
    
    # ======== 텍스트 토큰 확인 ========
    input_ids = sample.get("input_ids", None)

    if isinstance(input_ids, list):
        try:
            input_ids_tensor = torch.tensor(input_ids)
            print("input_ids tensor shape:", input_ids_tensor.shape)
        except Exception as e:
            print("input_ids 리스트지만 tensor 변환 실패:", e)
        print("input_ids 길이:", len(input_ids))
        print("input_ids 앞 10개:", input_ids[:10])
    elif isinstance(input_ids, torch.Tensor):
        print("input_ids shape:", input_ids.shape)
    else:
        print("input_ids: 없음 또는 알 수 없는 타입")

    # ======== 이미지 텐서 확인 ========
    def extract_first_tensor(obj):
        while isinstance(obj, list):
            if len(obj) == 0:
                return None
            obj = obj[0]
        return obj

    pixel_values_images = sample.get("pixel_values_images", [])
    first_image = extract_first_tensor(pixel_values_images)

    if isinstance(first_image, torch.Tensor):
        print("이미지 tensor shape:", first_image.shape)
        print("dtype:", first_image.dtype)
        print("device:", first_image.device)
    else:
        print(f"첫 이미지 타입이 예상과 다름: {type(first_image)}")
    

    def print_nested_structure(obj, level=0):
        indent = "  " * level
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            print(f"{indent}- List (len={len(obj)})")
            if len(obj) > 0:
                print_nested_structure(obj[0], level + 1)
        elif isinstance(obj, torch.Tensor):
            print(f"{indent}- Tensor: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{indent}- Type: {type(obj)}, value={obj}")

    print("pixel_values_images 구조:")
    print_nested_structure(pixel_values_images)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Vision 모델은 2 epoch 권장
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 실질적 배치 = 8
        learning_rate=1e-5,  # 더 낮게 (경량 모델)
        warmup_ratio=0.1,  # warmup 추가
        logging_steps=10,
        save_steps=500,  # 중간 저장
        eval_steps=500,  # 평가 주기
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # 메모리 효율적
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer with validation dataset and early stopping
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # 90% 데이터
        eval_dataset=val_dataset,# 10% 데이터
        args=training_args,
        data_collator=custom_collate_fn,
        peft_config=model.peft_config["default"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,  # 3번 연속 개선 없으면 중단
                early_stopping_threshold=0.001  # 최소 개선 폭
            )
        ]
    )
    
    logger.info("Starting training with validation...")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    batch = custom_collate_fn([train_dataset[0], train_dataset[1]])
    print(batch.keys())  # ['input_ids', 'labels', 'pixel_values_images']
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    print(batch['pixel_values_images'].shape)  # [2, 3, 378, 378]

    
    # 학습 시작
    trainer.train()

    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)  # processor도 저장
    
    logger.info(f"Training complete. Model saved to {output_dir}")
    
    # 최종 validation 성능 출력
    final_metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {final_metrics}")

if __name__ == "__main__":
    main()
