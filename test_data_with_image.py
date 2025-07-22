import os
import pandas as pd

root_dir = '/home/jieun/MobileVLM/data'
emotion_classes = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprised']
data = []

for emotion in emotion_classes:
    folder_path = os.path.join(root_dir, emotion)
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, filename)
            data.append({
                "ID": len(data) + 1,
                "이미지 경로": full_path,
                "감정 클래스": emotion
            })

df = pd.DataFrame(data)
df.to_csv("image_emotion_data.csv", index=False)
print(df.head())
print(f"총 데이터 개수: {len(df)}")
