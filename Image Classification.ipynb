import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image, image_dataset_from_directory

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
import os
import glob

#데이터셋은 dataset_dir 폴더 내에 test/classification1 & test/classification2가 있어야 한다 

# ✅ 데이터셋 경로 설정
dataset_dir = "/"

# 폴더 내 파일 확장자 확인
file_extensions = set()
for root, _, files in os.walk(dataset_dir):
    for file in files:
        file_extensions.add(file.split('.')[-1])  # 확장자 추출

print("🔍 데이터셋에 포함된 파일 확장자 목록:", file_extensions)

# ✅ 변환할 파일 확장자 목록
file_formats = ['.webp', '.jpg', '.avif']

# ✅ 모든 서브폴더에서 이미지 변환
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if any(file.endswith(ext) for ext in file_formats):
            file_path = os.path.join(root, file)
            new_file_path = file_path.rsplit(".", 1)[0] + ".jpeg"  # 확장자 변경

            try:
                # 이미지 열기 및 변환
                img = Image.open(file_path).convert("RGB")
                img.save(new_file_path, "JPEG")  # JPEG 형식으로 저장
                os.remove(file_path)  # 원본 삭제
                print(f"Converted: {file_path} → {new_file_path}")

            except Exception as e:
                print(f"Error converting {file_path}: {e}")

print("✅ 변환 완료! 모든 파일이 .jpg로 변경되었습니다.")

# ✅ 하이퍼파라미터 설정
batch_size = 32
img_size = 256

# ✅ 데이터 불러오기 (훈련, 검증 데이터)
dataset = image_dataset_from_directory(
    dataset_dir,  # Pass the dataset directory path
    image_size=(img_size, img_size),
    batch_size=batch_size,
    validation_split=0.2, # 20%를 검증 데이터로 사용
    subset="both",
    seed=42
)
train_dataset, val_dataset = dataset # 재현성을 위한 랜덤 시드

# ✅ 클래스 이름 확인
class_names = train_dataset.class_names
print(f"클래스 목록: {class_names}")

# ✅ CNN 모델 정의
model = models.Sequential([
    layers.Input(shape=(img_size, img_size, 3)),  # 입력층을 명시적으로 정의
    layers.Rescaling(1./255),  # 정규화 층 (input_shape 제거)
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

# ✅ 모델 컴파일
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",  # 정수 라벨이면 sparse_categorical_crossentropy 사용
              metrics=["accuracy"])

# ✅ 모델 요약 정보 출력
model.summary()

# ✅ 모델 학습
epochs = 15
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# ✅ 학습 결과 시각화 (정확도 그래프)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# ✅ 이미지 분류 테스트 함수
def check_img_and_classification(img_path):
    # 원본 이미지 출력
    print('##### 원본 이미지 #####')
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()
    print(f'img.shape: {img.shape}')

    # 이미지 불러오기 및 전처리
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # 🔥 정규화 추가 (학습 시와 동일)

    # 예측 수행
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # 분류 결과 출력
    print('##### 분류 결과 #####')
    for name, prob in zip(class_names, predictions[0]):
        print(f'{name}: {prob:.2f}')

    print(f"Predicted class: {predicted_class}")

# ✅ 테스트할 이미지 경로 설정
img_path = "/"
check_img_and_classification(img_path)
