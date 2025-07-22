''' clova x seed 3B/ 기존 코드 '''
# from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device="cuda")
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # LLM Example
# # It is recommended to use the chat template with HyperCLOVAX models.
# # Using the chat template allows you to easily format your input in ChatML style.
# chat = [
#         {"role": "system", "content": "you are helpful assistant!"},
#         {"role": "user", "content": "Hello, how are you?"},
#         {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#         {"role": "user", "content": "I'd like to show off how chat templating works!"},
# ]
# input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True)
# input_ids = input_ids.to(device="cuda")

# # Please adjust parameters like top_p appropriately for your use case.
# output_ids = model.generate(
#         input_ids,
#         max_new_tokens=64,
#         do_sample=True,
#         top_p=0.6,
#         temperature=0.5,
#         repetition_penalty=1.0,
# )
# print("=" * 80)
# print("LLM EXAMPLE")
# print(tokenizer.batch_decode(output_ids)[0])
# print("=" * 80)

# # VLM Example
# # For image and video inputs, you can use url, local_path, base64, or bytes.
# vlm_chat = [
#         {"role": "system", "content": {"type": "text", "text": "System Prompt"}},
#         {"role": "user", "content": {"type": "text", "text": "User Text 1"}},
#         {
#                 "role": "user",
#                 "content": {
#                         "type": "image",
#                         "filename": "tradeoff_sota.png",
#                         "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff_sota.png?raw=true",
#                         "ocr": "List the words in the image in raster order. Even if the word order feels unnatural for reading, the model will handle it as long as it follows raster order.",
#                         "lens_keywords": "Gucci Ophidia, cross bag, Ophidia small, GG, Supreme shoulder bag",
#                         "lens_local_keywords": "[0.07, 0.21, 0.92, 0.90] Gucci Ophidia",
#                 }
#         },
#         {
#                 "role": "user",
#                 "content": {
#                         "type": "image",
#                         "filename": "tradeoff.png",
#                         "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff.png?raw=true",
#                 }
#         },
#         {"role": "assistant", "content": {"type": "text", "text": "Assistant Text 1"}},
#         {"role": "user", "content": {"type": "text", "text": "User Text 2"}},
#         {
#                 "role": "user",
#                 "content": {
#                         "type": "video",
#                         "filename": "rolling-mist-clouds.mp4",
#                         "video": "/home/jieun/clovax_3B/01-01-02-02-02-02-14.mp4",
#                 }
#         },
#         {"role": "user", "content": {"type": "text", "text": "User Text 3"}},
# ]

# new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
# preprocessed = preprocessor(all_images, is_video_list=is_video_list)
# input_ids = tokenizer.apply_chat_template(
#         new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
# )

# output_ids = model.generate(
#         input_ids=input_ids.to(device="cuda"),
#         max_new_tokens=8192,
#         do_sample=True,
#         top_p=0.6,
#         temperature=0.5,
#         repetition_penalty=1.0,
#         **preprocessed,
# )
# print(tokenizer.batch_decode(output_ids)[0])


''' mobile vlm v2와 성능 비교를 위한 테스트 코드 (JSON 파싱, 번역)'''
# import time
# import json
# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# # 모델 로딩
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# device = "cuda"

# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 테스트 데이터 로드 (영어 입력 문장만 존재하는 CSV)
# csv_path = "test_data_100_2.csv"  
# df = pd.read_csv(csv_path)

# results = []

# for idx, row in df.iterrows():
#     sentence = row["영어 입력 문장"]

#     print(f"\n=== 테스트 {idx + 1} ===")

#     chat = [
#         {
#             "role": "system",
#             "content": "너는 영어 문장을 한국어로 정확히 번역하고, 문장에서 느껴지는 감정을 한 단어로 추출해 JSON으로 반환하는 어시스턴트야."
#         },
#         {
#             "role": "user",
#             "content": f"""
#     다음 영어 문장을 한국어로 정확히 번역하고, 감정을 한 단어로 추출해서 JSON으로 반환해줘.

#     영어 문장:
#     \"{sentence}\"
#     반드시 아래 형식으로 작성해. 'emotion_ko'는 감정 하나만, 'observable_signs'는 전체 한국어 번역 결과야.
    
#     출력 예시:
#     {{
#     "emotion_ko": "불안",
#     "observable_signs": "그의 자신감 있는 말투에도 불구하고, 내면 깊숙한 불안은 숨길 수 없었다."
#     }}
#     """
#         }
#     ]

#     input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(device)

#     start = time.time()
#     output_ids = model.generate(
#         input_ids,
#         max_new_tokens=256,
#         do_sample=True,
#         top_p=0.6,
#         temperature=0.5,
#         repetition_penalty=1.0,
#     )
#     elapsed = (time.time() - start) * 1000

#     output_text = tokenizer.batch_decode(output_ids)[0]

#     # "assistant" 태그 이후 JSON만 추출
#     json_candidate = output_text.split("<|im_start|>assistant")[-1]
#     json_start = json_candidate.find("{")
#     json_end = json_candidate.rfind("}") + 1
#     json_str = json_candidate[json_start:json_end]

#     print("출력 결과:\n", json_str)
#     print(f"지연시간: {elapsed:.2f}ms")

#     syntax_valid = False

#     try:
#         parsed_json = json.loads(json_str)
#         syntax_valid = True
#     except Exception as e:
#         print(f"JSON 파싱 실패: {e}")

#     print(f"JSON 구문 유효성: {syntax_valid}")

#     results.append({
#         "ID": row["ID"],
#         "입력 문장": sentence,
#         "모델 출력": json_str,
#         "지연시간(ms)": round(elapsed, 2),
#         "JSON 구문 유효성": syntax_valid
#     })

# results_df = pd.DataFrame(results)
# results_df.to_csv("clova_test_results_5.csv", index=False)
# print("\n전체 테스트 완료, 결과 파일: clova_test_results_5.csv")

''' 복잡한 구조의 JSON 구조로 파싱 '''

# import time
# import json
# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# # 모델 로딩
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# device = "cuda"

# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 테스트 데이터 로드 (영어 입력 문장만 존재하는 CSV)
# csv_path = "test_data_100_2.csv"  
# df = pd.read_csv(csv_path)

# results = []

# for idx, row in df.iterrows():
#     sentence = row["영어 입력 문장"]

#     print(f"\n=== 테스트 {idx + 1} ===")

#     chat = [
#         {
#                 "role": "system",
#                 "content": (
#                 "너는 영어 문장을 한국어로 번역하고, 아래와 같은 복잡한 JSON 구조를 반환하는 어시스턴트야."
#                 )
#         },
#         {
#                 "role": "user",
#                 "content": f"""
#         다음 영어 문장을 한국어로 번역하고, 아래 JSON 구조로 반환해줘:

#         영어 문장:
#         \"{sentence}\"

#         다른 설명 없이 반드시 아래 JSON 구조로 정확히 출력하되, 내용은 실제 번역 결과와 실제 감정을 바탕으로 작성해. 예시 값 복사 금지.

#         출력 예시:
#         {{
#         "emotion_ko": "불안",
#         "observable_signs": {{
#         "summary": "그의 떨리는 손에서 불안이 느껴졌다.",
#         "details": [
#         {{"type": "표정", "description": "불안한 미소", "confidence": 0.92}},
#         {{"type": "몸짓", "description": "손의 떨림", "confidence": 0.85}},
#         {{"type": "목소리", "description": "불안한 떨림", "confidence": 0.88}}
#         ]
#         }},
#         "intensity": 0.85,
#         "additional_info": {{
#         "source_language": "영어",
#         "target_language": "한국어",
#         "translation_quality": "높음",
#         "timestamp": "2025-06-30T12:34:56Z",
#         "reviewed_by": [
#         {{"reviewer_id": "AI-001", "status": "확인 완료"}},
#         {{"reviewer_id": "AI-002", "status": "자동 검증"}}
#         ]
#         }}
#         }}
#         """
#         }
#         ]


#     input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(device)

#     start = time.time()
#     output_ids = model.generate(
#         input_ids,
#         max_new_tokens=256,
#         do_sample=True,
#         top_p=0.6,
#         temperature=0.5,
#         repetition_penalty=1.0,
#     )
#     elapsed = (time.time() - start) * 1000

#     output_text = tokenizer.batch_decode(output_ids)[0]

#     # "assistant" 태그 이후 JSON만 추출
#     json_candidate = output_text.split("<|im_start|>assistant")[-1]
#     json_start = json_candidate.find("{")
#     json_end = json_candidate.rfind("}") + 1
#     json_str = json_candidate[json_start:json_end]

#     print("출력 결과:\n", json_str)
#     print(f"지연시간: {elapsed:.2f}ms")

#     syntax_valid = False

#     try:
#         parsed_json = json.loads(json_str)
#         syntax_valid = True
#     except Exception as e:
#         print(f"JSON 파싱 실패: {e}")

#     print(f"JSON 구문 유효성: {syntax_valid}")

#     results.append({
#         "ID": row["ID"],
#         "입력 문장": sentence,
#         "모델 출력": json_str,
#         "지연시간(ms)": round(elapsed, 2),
#         "JSON 구문 유효성": syntax_valid
#     })

# results_df = pd.DataFrame(results)
# results_df.to_csv("clova_test_results_complex_json_2.csv", index=False)
# print("\n전체 테스트 완료, 결과 파일: clova_test_results_complex_json_2.csv")

''' 번역 없이 JSON 파싱 성능 테스트 '''

# import time
# import json
# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# # 모델 로딩
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# device = "cuda"

# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 테스트 데이터 로드 (영어 입력 문장만 존재하는 CSV)
# csv_path = "test_data_100_2.csv"  
# df = pd.read_csv(csv_path)

# results = []

# for idx, row in df.iterrows():
#     sentence = row["영어 입력 문장"]

#     print(f"\n=== 테스트 {idx + 1} ===")

#     chat = [
#         {
#                 "role": "system",
#                 "content": (
#                 "너는 영어 문장을 분석해서, 감정과 세부 관찰 정보를 포함한 아래와 같은 복잡한 JSON 구조를 반환하는 어시스턴트야."
#                 )
#         },
#         {
#                 "role": "user",
#                 "content": f"""
#         다음 영어 문장을 분석해서, 아래 JSON 구조로 정확히 반환해줘:
#         ** 번역은 하지 말고 영어 원문을 그대로 활용해줘 **
        
#         영어 문장:
#         \"{sentence}\"

#         출력 예시 (내용은 실제로 분석한 결과로 작성, 예시 복사 금지):
#         {{
#         "emotion_en": "anxiety",
#         "observable_signs": {{
#         "summary": "His trembling hands revealed anxiety.",
#         "details": [
#         {{"type": "expression", "description": "nervous smile", "confidence": 0.92}},
#         {{"type": "gesture", "description": "shaking hands", "confidence": 0.85}},
#         {{"type": "voice", "description": "trembling tone", "confidence": 0.88}}
#         ]
#         }},
#         "intensity": 0.85,
#         "additional_info": {{
#         "source_language": "English",
#         "analysis_quality": "high",
#         "timestamp": "2025-06-30T12:34:56Z",
#         "reviewed_by": [
#         {{"reviewer_id": "AI-001", "status": "verified"}},
#         {{"reviewer_id": "AI-002", "status": "auto-checked"}}
#         ]
#         }}
#         }}
#         """
#         }
#         ]


#     input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True).to(device)

#     start = time.time()
#     output_ids = model.generate(
#         input_ids,
#         max_new_tokens=256,
#         do_sample=True,
#         top_p=0.6,
#         temperature=0.5,
#         repetition_penalty=1.0,
#     )
#     elapsed = (time.time() - start) * 1000

#     output_text = tokenizer.batch_decode(output_ids)[0]

#     # "assistant" 태그 이후 JSON만 추출
#     json_candidate = output_text.split("<|im_start|>assistant")[-1]
#     json_start = json_candidate.find("{")
#     json_end = json_candidate.rfind("}") + 1
#     json_str = json_candidate[json_start:json_end]

#     print("출력 결과:\n", json_str)
#     print(f"지연시간: {elapsed:.2f}ms")

#     syntax_valid = False

#     try:
#         parsed_json = json.loads(json_str)
#         syntax_valid = True
#     except Exception as e:
#         print(f"JSON 파싱 실패: {e}")

#     print(f"JSON 구문 유효성: {syntax_valid}")

#     results.append({
#         "ID": row["ID"],
#         "입력 문장": sentence,
#         "모델 출력": json_str,
#         "지연시간(ms)": round(elapsed, 2),
#         "JSON 구문 유효성": syntax_valid
#     })

# results_df = pd.DataFrame(results)
# results_df.to_csv("clova_test_results_complex_json_non_transfer.csv", index=False)
# print("\n전체 테스트 완료, 결과 파일: clova_test_results_complex_json_non_transfer.csv")


''' 감정 이미지 + 텍스트를 이용한 복잡한 JSON 구조 테스트 '''

import time
import json
import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# 모델 로딩
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 테스트 데이터 로드 (영어 입력 문장 + 이미지 경로 포함 CSV)
csv_path = "image_emotion_data.csv"  # 반드시 '영어 입력 문장', '이미지 경로' 컬럼 필요
df = pd.read_csv(csv_path)

results = []

for idx, row in df.iterrows():
    sentence = row["한글 입력 문장"]
    image_path = row["이미지 경로"]

    print(f"\n=== 테스트 {idx + 1} ===")

    chat = [
        {
                "role": "system",
                "content": {
                "type": "text",
                "text": (
                        "너는 텍스트와 이미지를 각각 독립적으로 분석하고, 종합적으로 감정 및 관찰 정보를 JSON 형식으로 정확히 반환하는 한국어 감정 분석 전문가야.\n"
                        "다음 규칙을 반드시 지켜야 한다:\n"
                        "- 텍스트 분석 결과와 이미지 분석 결과를 반드시 분리해서 작성\n"
                        "- 종합 감정(final_emotion)은 두 결과를 바탕으로 최종 판단\n"
                        "- 'emotion_text', 'emotion_image', 'final_emotion' 값은 반드시 '기쁨', '슬픔', '놀람', '분노', '불안', '중립', 'unknown' 중 하나의 한국어 단어만 작성하라\n"
                        "- 'observable_signs.summary'는 한국어 문장으로 작성하라\n"
                        "- 'observable_signs.details'는 반드시 아래 구조의 리스트로 작성하라:\n"
                        "  [\n"
                        "    {\"type\": \"표정\", \"description\": \"눈물이 고인 눈빛\", \"confidence\": 0.87},\n"
                        "    {\"type\": \"제스처\", \"description\": \"고개를 숙인 자세\", \"confidence\": 0.92}\n"
                        "  ]\n"
                        "- 'intensity'는 0.0 ~ 1.0 사이의 실수로 작성하라\n"
                        "- 'additional_info' 필드는 반드시 포함하며 다음 정보를 정확히 포함해야 한다:\n"
                        "  - 'source_language': 원본 문장의 언어 (예: \"영어\")\n"
                        "  - 'target_language': 번역된 언어 (예: \"한국어\")\n"
                        "  - 'translation_quality': \"높음\", \"중간\", \"낮음\" 중 하나\n"
                        "  - 'timestamp': ISO 8601 형식의 날짜/시간 문자열 (예: \"2025-06-30T12:34:56Z\")\n"
                        "  - 'reviewed_by': 다음 구조의 리스트를 포함\n"
                        "    [\n"
                        "      {\"reviewer_id\": \"AI-001\", \"status\": \"확인 완료\"},\n"
                        "      {\"reviewer_id\": \"AI-002\", \"status\": \"자동 검증\"}\n"
                        "    ]\n"
                        "- 반드시 아래 JSON 구조를 엄격히 준수해라. 내용은 반드시 새롭게 분석해 작성하고, 형식을 절대 위반하지 마라.\n"
                        "- 각 필드는 반드시 한국어로 실질적이고 구체적인 내용을 작성해야 한다.\n"
                        "- 빈 값으로 남기거나 의미 없는 출력은 절대 금지다.\n"

                )
                }
        },
        {
                "role": "user",
                "content": {
                "type": "text",
                "text": (
                        f"다음 문장과 이미지를 분석해 감정을 판단해라.\n"
                        f"문장:\n\"{sentence}\""
                )
                }
        },
        {
                "role": "user",
                "content": {
                "type": "image",
                "filename": os.path.basename(image_path),
                "image": image_path
                }
        },
        {
                "role": "user",
                "content": {
                "type": "text",
                "text": (
                "반드시 아래 JSON 구조로만 결과를 출력하라. 내용은 새롭게 분석해 작성하고, 형식을 절대 위반하지 마라.\n"
                "{\n"
                "  \"emotion_ko\": \"\",\n"
                "  \"observable_signs\": {\n"
                "    \"summary\": \"\",\n"
                "    \"details\": [\n"
                "      {\"type\": \"\", \"description\": \"\", \"confidence\": 0.0}\n"
                "    ]\n"
                "  },\n"
                "  \"intensity\": 0.0,\n"
                "  \"additional_info\": {\n"
                "    \"source_language\": \"\",\n"
                "    \"target_language\": \"\",\n"
                "    \"translation_quality\": \"\",\n"
                "    \"timestamp\": \"\",\n"
                "    \"reviewed_by\": [\n"
                "      {\"reviewer_id\": \"\", \"status\": \"\"}\n"
                "    ]\n"
                "  }\n"
                "}\n"
                "위 구조를 정확히 지키되, 모든 필드에 대해 한국어로 실제 분석한 내용을 반드시 작성하라. 빈 값이나 의미 없는 값은 절대 허용되지 않는다."
                )
                }
        }
        ]

    # 멀티모달 전처리
    new_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)
    print(f"\n=== 이미지 로딩 개수: {len(all_images)} ===")
    
    final_prompt = tokenizer.apply_chat_template(new_chat, tokenize=False)
    print("\n=== 최종 모델 입력 프롬프트 ===\n")
    print(final_prompt)
    
    kwargs = {}
    if all_images:
        kwargs = preprocessor(all_images, is_video_list=is_video_list)

    input_ids = tokenizer.apply_chat_template(
        new_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
    ).to(device)

    start = time.time()
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        **kwargs
    )
    elapsed = (time.time() - start) * 1000

    output_text = tokenizer.batch_decode(output_ids)[0]

    # "assistant" 태그 이후 JSON만 추출
    json_candidate = output_text.split("<|im_start|>assistant")[-1]
    json_start = json_candidate.find("{")
    json_end = json_candidate.rfind("}") + 1
    json_str = json_candidate[json_start:json_end]

    print("출력 결과:\n", json_str)
    print(f"지연시간: {elapsed:.2f}ms")

    syntax_valid = False
    try:
        parsed_json = json.loads(json_str)
        syntax_valid = True
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")

    print(f"JSON 구문 유효성: {syntax_valid}")

    results.append({
        "ID": row["ID"],
        "입력 문장": sentence,
        "이미지 경로": image_path,
        "모델 출력": json_str,
        "지연시간(ms)": round(elapsed, 2),
        "JSON 구문 유효성": syntax_valid
    })

# 결과 저장
results_df = pd.DataFrame(results)
results_df.to_csv("clova_test_results_text_image_json_3.csv", index=False)
print("\n전체 테스트 완료, 결과 파일: clova_test_results_text_image_json_3.csv")



''' 지연시간(번역 ~ JSON 파싱까지) 평균 구하기 '''
# import pandas as pd

# # 테스트 결과 CSV 경로
# csv_path = "clova_test_results_text_image_json_2.csv"

# # 결과 파일 로드
# df = pd.read_csv(csv_path)

# # 지연시간 컬럼 평균 계산
# avg_latency = df["지연시간(ms)"].mean()

# print(f"전체 평균 지연시간: {avg_latency:.2f} ms")

