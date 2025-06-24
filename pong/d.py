# ======================================================================
# 1) 라이브러리 불러오기
# ======================================================================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, Activation, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ======================================================================
# 2) 데이터셋 경로 및 레이블 파일 로드
# ======================================================================
# (a) 레이블 CSV 경로
LABEL_CSV_PATH = '/mnt/data/labels.csv'  # 필요하다면 경로를 수정하세요

# (b) 이미지 폴더 경로: labels.csv의 'filename' 컬럼에는 이 경로 하위의 파일명만 쓰여 있어야 합니다.
# 예) filename = 'image_00001.jpg' → 실제 파일은 '/mnt/data/images/image_00001.jpg' 여야 함
IMAGE_DIR = '/mnt/data/images'  # 실제 이미지가 있는 폴더로 수정하세요

# 레이블 CSV 읽기
labels_df = pd.read_csv(LABEL_CSV_PATH)
# 예시: labels_df.head()
#       filename        label
# 0   image_00001.jpg    77
# 1   image_00002.jpg    77
# ...

# ======================================================================
# 3) 클래스 정보 확인 및 시각화 (Optional)
# ======================================================================
# 3-1) 전체 이미지(행) 개수, 클래스 수
total_images = len(labels_df)
unique_labels = sorted(labels_df['label'].unique())
num_classes = len(unique_labels)

print(f"총 이미지 수: {total_images}")
print(f"고유 레이블(클래스) 수: {num_classes} (레이블: {unique_labels[:5]} ... {unique_labels[-5:]})")

# 3-2) 클래스별 이미지 개수 분포 확인
plt.figure(figsize=(12, 6))
sns.countplot(x='label', data=labels_df, order=unique_labels)
plt.title('클래스별 이미지 개수 분포')
plt.xlabel('Label (1~{})'.format(num_classes))
plt.ylabel('이미지 개수')
plt.show()

# ======================================================================
# 4) 이미지 → 배열 변환 및 피처/레이블 분리
# ======================================================================
# 이미지 읽어서 (150×150)로 리사이즈 후 리스트에 저장
features = []
labels   = []

# progress 표시를 위해 tqdm을 쓰셔도 좋습니다. 여기서는 간단히 for-loop로 처리합니다.
for idx, row in labels_df.iterrows():
    img_name = row['filename']
    class_id = row['label']

    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        # 이미지가 없는 경우 경고 메시지 출력하고 넘어가기
        print(f"[Warning] 파일을 찾을 수 없음: {img_path}")
        continue

    # (1) 이미지 로드
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[Warning] 이미지 로드 실패: {img_path}")
        continue

    # (2) (150 × 150) 크기 고정
    img_resized = cv2.resize(img, (150, 150))

    # (3) BGR → RGB 변환 (Matplotlib 시각화를 위해)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # (4) 리스트에 저장
    features.append(img_rgb)
    labels.append(class_id)

# NumPy 배열로 변환 및 정규화 (0~1 사이 float32)
X = np.array(features, dtype=np.float32) / 255.0
y = np.array(labels, dtype=np.int32)

print(f"X.shape = {X.shape}")  # e.g. (7323, 150, 150, 3)
print(f"y.shape = {y.shape}")  # e.g. (7323, )

# ======================================================================
# 5) 레이블 One-hot Encoding
# ======================================================================
# 현재 y에는 1~102까지 정수 레이블이 들어 있으므로,
# 0~101로 바꾸기 위해 (label-1) 후 to_categorical 적용
y_zero_based = y - 1  # 0 ~ (num_classes-1) 범위로
y_cat        = to_categorical(y_zero_based, num_classes=num_classes)

print(f"One-hot 변환 후 y_cat.shape = {y_cat.shape}")  # e.g. (7323, 102)

# ======================================================================
# 6) Train / Test 셋 분할
# ======================================================================
# 전체 데이터를 8:2 비율로 train/test로 분할
x_train, x_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_zero_based
)

print(f"x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}")
print(f"x_test.shape  = {x_test.shape},  y_test.shape  = {y_test.shape}")

# ======================================================================
# 7) 모델 설계 (Simple CNN 예제)
#    → 출력 뉴런 개수를 num_classes (=102)로 설정
# ======================================================================
model = Sequential()

# Conv Block 1
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Conv Block 2
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Conv Block 3
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Fully Connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# 출력층: num_classes개
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일
optimizer = Adam(lr=1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 요약
model.summary()

# ======================================================================
# 8) 학습 시 callbacks 및 데이터 증강 설정
# ======================================================================
# (1) Learning Rate Scheduler: val_accuracy가 3 Epoch 이상 증가하지 않으면 LR 0.1 배로 감소
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=3,
    verbose=1
)

# (2) 데이터 증강 (ImageDataGenerator)
datagen = ImageDataGenerator(
    rotation_range=15,         # -15도~+15도 회전
    width_shift_range=0.1,     # 좌우 10% 범위 내 이동
    height_shift_range=0.1,    # 상하 10% 범위 내 이동
    zoom_range=0.1,            # 10% 확대/축소
    horizontal_flip=True,      # 좌우 뒤집기
    fill_mode='nearest'
)
datagen.fit(x_train)

# ======================================================================
# 9) 모델 학습
# ======================================================================
batch_size = 64
epochs     = 50

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch = x_train.shape[0] // batch_size,
    epochs          = epochs,
    validation_data = (x_test, y_test),
    callbacks       = [reduce_lr],
    verbose         = 1
)

# ======================================================================
# 10) 학습 결과 시각화
# ======================================================================
# 10-1) Loss 변화
plt.figure(figsize=(12,5))
plt.plot(history.history['loss'],   label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 10-2) Accuracy 변화
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'],    label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ======================================================================
# 11) 테스트 데이터에 대한 최종 성능 확인
# ======================================================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ======================================================================
# 12) 몇 가지 예측 샘플 시각화 (Correct/Incorrect 예시)
# ======================================================================
# 12-1) 예측 수행
preds = model.predict(x_test)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 12-2) 올바르게 예측된 인덱스 일부, 잘못 예측된 인덱스 일부 추출
correct_idxs   = np.where(pred_classes == true_classes)[0]
incorrect_idxs = np.where(pred_classes != true_classes)[0]

# (예시로 처음 8개씩만 가져와서 시각화)
num_display = 8
correct_sample_idxs   = correct_idxs[:num_display]
incorrect_sample_idxs = incorrect_idxs[:num_display]

# (가정) 레이블을 사람이 읽을 수 있는 이름으로 매핑하고 싶다면,
# labels_map = {1: 'class1_name', 2: 'class2_name', …, 102: 'class102_name'}
# 처럼 dict를 만들어 두고, 아래에서 참조하면 됩니다.
# 여기서는 그냥 정수 레이블을 표시합니다.

# 올바르게 예측된 예시
plt.figure(figsize=(16, 8))
for i, idx in enumerate(correct_sample_idxs):
    ax = plt.subplot(4, 4, i + 1)
    img = x_test[idx]
    true_label = true_classes[idx] + 1      # 다시 1-based로
    pred_label = pred_classes[idx] + 1
    ax.imshow(img)
    ax.set_title(f"True: {true_label}\nPred: {pred_label}")
    ax.axis('off')
plt.suptitle("올바르게 예측된 샘플 예시", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 잘못 예측된 예시
plt.figure(figsize=(16, 8))
for i, idx in enumerate(incorrect_sample_idxs):
    ax = plt.subplot(4, 4, i + 1)
    img = x_test[idx]
    true_label = true_classes[idx] + 1
    pred_label = pred_classes[idx] + 1
    ax.imshow(img)
    ax.set_title(f"True: {true_label}\nPred: {pred_label}")
    ax.axis('off')
plt.suptitle("잘못 예측된 샘플 예시", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ======================================================================
# 13) 임의의 이미지에 대해 예측 함수 정의 (Optional)
# ======================================================================
def process_and_predict_image(img_path, model, img_size=(150, 150)):
    """
    로컬 경로나 URL로부터 이미지를 불러와
    모델에 맞춰 전처리한 뒤 예측 결과를 리턴합니다.
    """
    # 1) 이미지 로드
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

    # 2) (150 × 150)로 리사이즈 후 RGB 변환
    img_resized = cv2.resize(img, img_size)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3) 정규화 및 배치 차원 추가
    img_array = img_rgb.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, 3)

    # 4) 모델 예측
    preds = model.predict(img_array)
    pred_class_zero = np.argmax(preds, axis=1)[0]
    return pred_class_zero + 1  # 1-based 리턴

# 사용 예시 (Local 파일 경로)
# sample_img_path = '/mnt/data/images/image_01234.jpg'
# pred_label = process_and_predict_image(sample_img_path, model)
# print(f"예측 클래스: {pred_label}")

