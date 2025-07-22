from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# # 저장된 로컬 모델 경로
# model_path = "/data/clovax/pretrained_model"

# # 로컬에서 로드
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
# preprocessor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
revision="v0.1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision=revision).to(device="cuda")
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

# LLM Example
# It is recommended to use the chat template with HyperCLOVAX models.
# Using the chat template allows you to easily format your input in ChatML style.
chat = [
        {"role": "system", "content": "you are helpful assistant!"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True)
input_ids = input_ids.to(device="cuda")

# Please adjust parameters like top_p appropriately for your use case.
output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.6,
        temperature=0.5,
        repetition_penalty=1.0,
)
print("=" * 80)
print("LLM EXAMPLE")
print(tokenizer.batch_decode(output_ids)[0])
print("=" * 80)

# VLM Example
# For image and video inputs, you can use url, local_path, base64, or bytes.
vlm_chat = [
        {"role": "system", "content": {"type": "text", "text": "System Prompt"}},
        {"role": "user", "content": {"type": "text", "text": "User Text 1"}},
        {
                "role": "user",
                "content": {
                        "type": "image",
                        "filename": "tradeoff_sota.png",
                        "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff_sota.png?raw=true",
                        "ocr": "List the words in the image in raster order. Even if the word order feels unnatural for reading, the model will handle it as long as it follows raster order.",
                        "lens_keywords": "Gucci Ophidia, cross bag, Ophidia small, GG, Supreme shoulder bag",
                        "lens_local_keywords": "[0.07, 0.21, 0.92, 0.90] Gucci Ophidia",
                }
        },
        {
                "role": "user",
                "content": {
                        "type": "image",
                        "filename": "tradeoff.png",
                        "image": "https://github.com/naver-ai/rdnet/blob/main/resources/images/tradeoff.png?raw=true",
                }
        },
        {"role": "assistant", "content": {"type": "text", "text": "Assistant Text 1"}},
        {"role": "user", "content": {"type": "text", "text": "User Text 2"}},
        {
                "role": "user",
                "content": {
                        "type": "video",
                        "filename": "rolling-mist-clouds.mp4",
                        "video": "freenaturestock-rolling-mist-clouds.mp4",
                }
        },
        {"role": "user", "content": {"type": "text", "text": "User Text 3"}},
]

new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
preprocessed = preprocessor(all_images, is_video_list=is_video_list)
input_ids = tokenizer.apply_chat_template(
        new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True,
)

output_ids = model.generate(
        input_ids=input_ids.to(device="cuda"),
        max_new_tokens=8192,
        do_sample=True,
        top_p=0.6,
        temperature=0.5,
        repetition_penalty=1.0,
        **preprocessed,
)
print("=" * 80)
print("VLM EXAMPLE")
print(tokenizer.batch_decode(output_ids)[0])
print("=" * 80)



'''clova x seed 3B 모델을 Gradio로 구현하여 텍스트 채팅'''

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# import gradio as gr

# # 모델 및 토크나이저 로드
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device="cuda")
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model. to(device)

# def chat_interface(user_input: str) -> str:
#     '''
#     사용자 입력을 CLOVA X 채팅 템플릿에 적용하여 모델 응답을 생성합니다.
#     Args:
#     user_input (str): 사용자가 입력한 질문.
#     Returns:
#     str: 모델이 생성한 assistant의 응답 부분.
#     '''
#     # 채팅 템플릿 구성
#     chat = [
#         {"role": "tool_list", "content": ""},
#         {"role": "system", "content": (
#             '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다. \n'
#             '- 오늘은 2025년 04월 24일(목)이다.'
#         )},
#         {"role": "user", "content": user_input}
#     ]
    
#     inputs = tokenizer.apply_chat_template(
#         chat,
#         add_generation_prompt=True, 
#         return_dict=True, 
#         return_tensors="pt"
#     )
#     inputs = {k: v.to(device) for k, v in inputs. items()}
    
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.7,
#         temperature=0.7,
#         repetition_penalty=1.1
#     )
        
#     output = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
#     full_output = "\n".join(output)

#     # "assistant" 태그 이후 텍스트만 정확히 추출
#     assistant_reply = ""
#     if "<|im_start|>assistant" in full_output:
#         after_tag = full_output.split("<|im_start|>assistant", 1)[1]
#         assistant_reply = after_tag.split("<|im_end|>", 1)[0].strip()

#     return assistant_reply


# def respond (message, chat_history):
#     '''
#     대화
#     내역에 사용자의 메시지와 모델의 응답을 추가합니다.
#     Args:
#     mes sage: 사용자의 새 입력 에시지.
#     chat_history: 기존 대화 기록 (듀플(list) 형식: (user, assistant)).
#     Returns:
#     tuple: 빈 문자열(입력장 초기화)과 업데이트된 대화 내역.
#     '''
#     response = chat_interface(message)
#     chat_history.append((message, response))
#     return "", chat_history

# css =  """
# html, body {
#     width: 100%;
#     height: 100%;
#     margin: 0;
#     padding: 0;
# }
# .gradio-container {
#     width: 100% !important;
#     height: 100% !important;
#     margin: 0; 
#     padding: 0;
# """
# # Gradio Blocks 인터페이스 구성 (채팅 UI)
# with gr.Blocks(css=css) as demo:
#     gr.Markdown("## CLOVA X Chat (3B)")
#     chatbot = gr.Chatbot(label="CLOVA X Chat")
#     state = gr.State([]) # 대화 히스토리 저장용
#     with gr.Row():
#         txt= gr.Textbox(placeholder="질문을 입력하세요.", show_label=False)
#     txt. submit(respond, inputs=[txt, state], outputs=[txt, chatbot])
# # 5. 인터페이스 실행 (외부 접속 가능, 포트: 8283)
# demo. launch(server_port=5000, server_name="0.0.0.0")

'''clova x seed 3B 모델을 Gradio로 구현하여 멀티모달 채팅'''

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
# import gradio as gr
# import os

# device = "cuda"

# # 모델 이름 (캐시 기반으로 다운로드 + 로드됨)
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

# # 모델 로딩 (캐시에서 로드되거나 없으면 다운로드됨)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision="main").to(device)
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, revision="main")
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="main")

# model.to(device)

# # 디바이스 상태 확인
# print(f"현재 디바이스: {device}")
# print(f"모델 실제 디바이스 위치: {next(model.parameters()).device}")

# def multimodal_chat(user_input: str, image=None, video_file=None) -> str:
#     chat = [
#         {"role": "tool_list", "content": ""},
#         {"role": "system", "content": (
#             '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n'
#             '- 오늘은 2025년 06월 30일(월)이다.'
#         )}
#     ]

#     if user_input.strip():
#         chat.append({"role": "user", "content": {"type": "text", "text": user_input}})
    
#     if image:
#         chat.append({
#             "role": "user",
#             "content": {"type": "image", "filename": os.path.basename(image), "image": image}
#         })
    
#     if video_file:
#         chat.append({
#             "role": "user",
#             "content": {"type": "video", "filename": os.path.basename(video_file.name), "video": video_file.name}
#         })

#     new_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)

#     kwargs = {}
#     if all_images:
#         kwargs = preprocessor(all_images, is_video_list=is_video_list)

#     input_ids = tokenizer.apply_chat_template(
#         new_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
#     ).to(device)
    
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.7,
#         temperature=0.7,
#         repetition_penalty=1.1,
#         **kwargs  # 이미지·영상 있을 때만 전달
#     )

#     output = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
#     full_output = "\n".join(output)

#     print("=" * 30)
#     print(full_output)  # 모델 전체 원문 출력
#     print("=" * 30)

#     assistant_reply = ""

#     # <|im_start|>assistant 태그 기준 파싱
#     if "<|im_start|>assistant" in full_output:
#         after_tag = full_output.split("<|im_start|>assistant", 1)[1]
#         assistant_reply = after_tag.split("<|im_end|>", 1)[0].strip()

#     # 태그 없으면 전체 출력 반환
#     if not assistant_reply:
#         assistant_reply = full_output.strip()

#     return assistant_reply



# def respond(user_input, image, video, chat_history):
#     '''
#     대화 내역에 사용자 입력, 이미지, 영상, 모델 응답 추가
#     '''
#     response = multimodal_chat(user_input, image, video)
#     chat_history.append({"role": "user", "content": user_input})
#     chat_history.append({"role": "assistant", "content": response})
#     return "", None, None, chat_history

# css = """
# html, body { width: 100%; height: 100%; margin: 0; padding: 0; }
# .gradio-container { width: 100% !important; height: 100% !important; margin: 0; padding: 0; }
# """

# with gr.Blocks(css=css) as demo:
#     gr.Markdown("## CLOVA X Multimodal Chat (3B)")
#     chatbot = gr.Chatbot(label="CLOVA X Chat", type="messages")
#     state = gr.State([])  # 대화 히스토리
#     with gr.Row():
#         txt = gr.Textbox(placeholder="질문을 입력하세요.", show_label=False)
#         img = gr.Image(type="filepath", label="이미지 입력 (선택)")
#         vid = gr.File(file_types=[".mp4", ".avi", ".mov"], label="영상 입력 (선택)")
#     txt.submit(respond, inputs=[txt, img, vid, state], outputs=[txt, img, vid, chatbot])


# demo.launch(server_port=5000, server_name="0.0.0.0")



'''실시간 스트리밍으로 CLOVA X 채팅 인터페이스(Gradio) 구현'''

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TextIteratorStreamer
# import gradio as gr
# import os
# import threading

# # 모델 및 토크나이저 로드
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device="cuda")
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device = "cuda"
# model.to(device)

# # 디바이스 상태 확인
# print(f"현재 디바이스: {device}")
# print(f"모델 실제 디바이스 위치: {next(model.parameters()).device}")

# def multimodal_chat_stream(user_input: str, image=None, video_file=None):
#     chat = [
#         {"role": "tool_list", "content": ""},
#         {"role": "system", "content": (
#             '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n'
#             '- 오늘은 2025년 07월 04일(금)이다.'
#         )}
#     ]

#     if user_input.strip():
#         chat.append({"role": "user", "content": {"type": "text", "text": user_input}})
    
#     if image:
#         chat.append({
#             "role": "user",
#             "content": {"type": "image", "filename": os.path.basename(image), "image": image}
#         })
    
#     if video_file:
#         chat.append({
#             "role": "user",
#             "content": {"type": "video", "filename": os.path.basename(video_file.name), "video": video_file.name}
#         })

#     new_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)

#     kwargs = {}
#     if all_images:
#         kwargs = preprocessor(all_images, is_video_list=is_video_list)

#     input_ids = tokenizer.apply_chat_template(
#         new_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
#     ).to(device)

#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     generation_kwargs = dict(
#         input_ids=input_ids,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.7,
#         temperature=0.7,
#         repetition_penalty=1.1,
#         streamer=streamer,
#         **kwargs
#     )

#     # 백그라운드 스레드로 생성 시작
#     thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()

#     partial_text = ""
#     for new_text in streamer:
#         partial_text += new_text
#         yield partial_text  # Gradio로 실시간 전송


# def respond_stream(user_input, image, video, chat_history):
#     '''
#     스트리밍으로 실시간 응답 추가 + 대화 내역 유지
#     '''
#     history = chat_history.copy() if chat_history else []
#     history.append({"role": "user", "content": user_input})

#     partial_reply = ""

#     for chunk in multimodal_chat_stream(user_input, image, video):
#         partial_reply = chunk
#         updated_history = history + [{"role": "assistant", "content": partial_reply}]
#         yield "", None, None, updated_history  # 실시간 업데이트

#     # 최종 응답 다 나오면 전체 history 갱신
#     history.append({"role": "assistant", "content": partial_reply})
#     yield "", None, None, history


# css = """
# html, body { width: 100%; height: 100%; margin: 0; padding: 0; }
# .gradio-container { width: 100% !important; height: 100% !important; margin: 0; padding: 0; }
# """

# with gr.Blocks(css=css) as demo:
#     gr.Markdown("## CLOVA X Multimodal Chat (3B) - 실시간 스트리밍")
#     chatbot = gr.Chatbot(label="CLOVA X Chat", type="messages")
#     state = gr.State([])  # 대화 히스토리
#     with gr.Row():
#         txt = gr.Textbox(placeholder="질문을 입력하세요.", show_label=False)
#         img = gr.Image(type="filepath", label="이미지 입력 (선택)")
#         vid = gr.File(file_types=[".mp4", ".avi", ".mov"], label="영상 입력 (선택)")
#     txt.submit(respond_stream, inputs=[txt, img, vid, state], outputs=[txt, img, vid, chatbot], show_progress=True)

# demo.launch(server_port=5000, server_name="0.0.0.0")


'''실시간 스트리밍으로 CLOVA X 채팅 인터페이스(Terminal) 구현'''

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TextIteratorStreamer
# import os
# import threading

# # 모델 로드
# model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")
# preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# device = "cuda"
# model.to(device)

# # 디바이스 상태 확인
# print(f"현재 디바이스: {device}")
# print(f"모델 실제 디바이스 위치: {next(model.parameters()).device}")


# def multimodal_chat_stream_terminal(user_input: str, image=None, video_file=None):
#     chat = [
#         {"role": "tool_list", "content": ""},
#         {"role": "system", "content": (
#             '- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n'
#             '- 오늘은 2025년 07월 04일(금)이다.'
#         )}
#     ]

#     if user_input.strip():
#         chat.append({"role": "user", "content": {"type": "text", "text": user_input}})
    
#     if image:
#         chat.append({
#             "role": "user",
#             "content": {"type": "image", "filename": os.path.basename(image), "image": image}
#         })
    
#     if video_file:
#         chat.append({
#             "role": "user",
#             "content": {"type": "video", "filename": os.path.basename(video_file.name), "video": video_file.name}
#         })

#     new_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)

#     kwargs = {}
#     if all_images:
#         kwargs = preprocessor(all_images, is_video_list=is_video_list)

#     input_ids = tokenizer.apply_chat_template(
#         new_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
#     ).to(device)

#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     generation_kwargs = dict(
#         input_ids=input_ids,
#         max_new_tokens=1024,
#         do_sample=True,
#         top_p=0.7,
#         temperature=0.7,
#         repetition_penalty=1.1,
#         streamer=streamer,
#         **kwargs
#     )

#     # 비동기 생성
#     thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
#     thread.start()

#     print("\n[CLOVA X 응답 시작]\n")

#     partial_text = ""
#     for new_text in streamer:
#         print(new_text, end="", flush=True)  # 실시간 터미널 출력
#         partial_text += new_text

#     print("\n\n[CLOVA X 응답 완료]\n")
#     return partial_text


# # 테스트 실행부
# if __name__ == "__main__":
#     while True:
#         user_input = input("나: ")
#         if user_input.strip().lower() in ["exit", "quit"]:
#             break
#         multimodal_chat_stream_terminal(user_input)

