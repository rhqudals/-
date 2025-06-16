import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import datetime
from deepface import DeepFace
import re
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from xgboost import XGBClassifier

# === Bag of Lies 설정 ===
DATA_DIR = "./data/bag_of_lies"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FEATURES_DIR = "./data/features"
MODEL_PATH = "./model/lie_detector_model.pkl"
LABEL_CSV = os.path.join(DATA_DIR, "Annotations.csv")
LABEL_CSV = "./data/bag_of_lies/Annotations.csv"
RECORDED_DIR = "./data/recorded"

df = pd.read_csv(LABEL_CSV)

def clean_path(p):
    # 중복된 Finalised 제거, 앞/뒤 슬래시 제거
    p = p.strip().lstrip('/').replace('\\', '/')
    while p.startswith('Finalised/Finalised'):
        p = p.replace('Finalised/Finalised', 'Finalised', 1)
    return p

df['video'] = df['video'].apply(clean_path)
df.to_csv(LABEL_CSV, index=False)
print("CSV 경로 정리 완료")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs("./model", exist_ok=True)
os.makedirs(RECORDED_DIR, exist_ok=True)
os.makedirs("./data/frames", exist_ok=True)

print("data 절대경로:", os.path.abspath(DATA_DIR))
print("model 절대경로:", os.path.abspath("./model"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_specific_frame(video_path, frame_number, save_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상 열기 실패: {video_path}")
        return False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"영상 전체 프레임 수: {total_frames}")
    if frame_number >= total_frames:
        print(f"지정한 프레임 번호가 영상 길이보다 큽니다.")
        return False
    # 원하는 프레임 위치로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
        print(f"{frame_number}번째 프레임 저장 완료: {save_path}")
    else:
        print(f"{frame_number}번째 프레임 읽기 실패")
    cap.release()
    return ret

def extract_features_from_video(video_path):
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상 열기 실패: {video_path}")
        return None

    frame_count = 0
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        show_frame = frame.copy()  # 영상 표시용 복사본
        info_text = f"{frame_count}번째 프레임"
        # 프레임 번호 한글로 표시
        show_frame = draw_text_hangul(show_frame, info_text, (10, 30), font_size=32, color=(0,255,0))

        # 원하는 프레임에서 이미지 저장
        if frame_count in [10, 30, 50, 100]:
            save_path = f"./data/frames/frame_{frame_count}.jpg"
            saved = cv2.imwrite(save_path, frame)
            print(f"{save_path} 저장: {saved}")

        if frame_count % 30 == 0:
            print(f"{video_path} - {frame_count}번째 프레임 분석 중...")
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                results.append(result)
                # 얼굴 영역 표시
                region = result.get("region")
                face_conf = result.get("face_confidence", 0)
                h_img, w_img = show_frame.shape[:2]
                # region과 face_confidence 방어 코드 추가
                if (
                    region and
                    face_conf > 0.5 and
                    0 <= region.get("x", 0) < w_img and
                    0 <= region.get("y", 0) < h_img and
                    region.get("w", 0) > 10 and region.get("h", 0) > 10 and
                    region.get("x", 0) + region.get("w", 0) <= w_img and
                    region.get("y", 0) + region.get("h", 0) <= h_img
                ):
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
                    cv2.rectangle(show_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # 감정 결과를 한글로 표시
                emotions = result.get("emotion", {})
                emotion_str = ", ".join([f"{k}:{int(v)}" for k, v in emotions.items()])
                emotion_text = f"감정: {emotion_str}"
                show_frame = draw_text_hangul(show_frame, emotion_text, (10, 70), font_size=24, color=(255,0,0))
            except Exception as e:
                print(f"프레임 분석 실패: {e}")
                cv2.imwrite(f"fail_frame_{frame_count}.jpg", frame)  # 실패한 프레임 저장

        cv2.imshow("특징 추출 중", show_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()    #프레임 저장
    if frame_count == 100:
        saved1 = cv2.imwrite(f"frame_{frame_count}.jpg", frame)
        saved2 = cv2.imwrite("./data/frames/frame_100.jpg", frame)
        print(f"frame_{frame_count}.jpg 저장: {saved1}, ./data/frames/frame_100.jpg 저장: {saved2}")
    if results:
        return results[0]
    else:
        return None

def extract_all_features():
    label_df = pd.read_csv(LABEL_CSV)
    print(f"CSV에 등록된 영상 개수: {len(label_df)}")
    for idx, row in label_df.iterrows():
        video_relpath = str(row["video"]).strip().replace("\\", "/")
        label = str(row["truth"]).strip().lower()
        video_path = os.path.join(VIDEOS_DIR, video_relpath)
        video_path = os.path.normpath(video_path)
        # 저장될 json 파일 경로 생성
        rel_path = os.path.relpath(video_path, VIDEOS_DIR)
        safe_name = re.sub(r'[\\/]', '_', rel_path)
        json_name = os.path.splitext(safe_name)[0] + ".json"
        json_path = os.path.join(FEATURES_DIR, json_name)
        # 이미 json 파일이 있으면 건너뜀
        if os.path.exists(json_path):
            print(f"[{idx}] 이미 추출됨: {json_path}")
            continue
        print(f"[{idx}] 분석 시작: {video_path}")
        if not os.path.exists(video_path):
            print(f"[{idx}] 영상 파일 없음: {video_path}")
            continue
        features = extract_features_from_video(video_path)
        if features:
            print(f"[{idx}] 특징 추출 성공: {video_path}")
            if isinstance(features, list):
                features = features[0]
            features["truth"] = label
            if "emotion" in features:
                for k, v in features["emotion"].items():
                    features["emotion"][k] = float(v)
            with open(json_path, "w") as f:
                json.dump(features, f)
        else:
            print(f"[{idx}] 특징 추출 실패: {video_path}")
    print("특징 추출 완료.")

def load_data():
    X, y = [], []
    for feat_file in glob.glob(os.path.join(FEATURES_DIR, "*.json")):
        with open(feat_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):  # DeepFace 버전에 따라
                data = data[0]
            if 'emotion' in data and 'truth' in data:  # label → truth
                feats = [data['emotion'].get(k, 0) for k in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']]
                label = 1 if str(data['truth']).lower() == "lie" else 0  # label → truth
                X.append(feats)
                y.append(label)
    return np.array(X), np.array(y)

def train_model():
    X, y = load_data()
    if len(X) < 2:
        print("학습 데이터가 2개 이상 필요합니다. 특징 추출을 더 진행하세요.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    print("모델 저장 완료:", MODEL_PATH)

def show_camera():
    cap = cv2.VideoCapture(0)
    print("웹캠을 시작합니다. 'q'를 누르면 종료되고, 영상이 저장됩니다.")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{now}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라를 찾을 수 없습니다.")
            break
        out.write(frame)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"{filename} 파일로 저장되었습니다.")

def run_camera_lie_detector():
    if not os.path.exists(MODEL_PATH):
        print("모델이 없습니다. 먼저 모델을 학습하세요.")
        return
    model = joblib.load(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    print("웹캠을 시작합니다. 'q'를 누르면 종료됩니다.")

    # 날짜 및 임시 파일명으로 VideoWriter 준비
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = os.path.join(RECORDED_DIR, f"lie_detect_temp_{now}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    last_label = "unknown"
    last_region = None  # 마지막 얼굴 위치 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print("DeepFace result:", result)
            while isinstance(result, list):
                if not result:
                    raise ValueError("DeepFace 결과가 비어있음")
                result = result[0]
            # 얼굴 영역 표시
            region = result.get("region")
            print("region:", region)  # 추가
            if region:
                last_region = region
            # 항상 마지막 얼굴 위치에 사각형 그리기
            if last_region:
                x, y, w, h = last_region.get("x", 0), last_region.get("y", 0), last_region.get("w", 0), last_region.get("h", 0)
                # 프레임 크기 체크
                h_img, w_img = frame.shape[:2]
                if 0 <= x < w_img and 0 <= y < h_img and w > 0 and h > 0 and x + w <= w_img and y + h <= h_img:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            emotions = result.get("emotion", {})
            print("emotions:", emotions)
            emotion_features = [emotions.get(k, 0) for k in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']]
            print("emotion_features:", emotion_features)
            pred = model.predict([emotion_features])
            print("pred:", pred)
            last_label = "lie" if pred[0] == 1 else "truth"
            label_text = "거짓말" if pred[0] == 1 else "진실"
        except Exception as e:
            print("분석불가 예외:", e)
            label_text = "분석불가"
            # DeepFace 분석 실패해도 마지막 얼굴 위치에 사각형 그리기
            if last_region:
                x, y, w, h = last_region.get("x", 0), last_region.get("y", 0), last_region.get("w", 0), last_region.get("h", 0)
                # 프레임 크기 체크
                h_img, w_img = frame.shape[:2]
                if 0 <= x < w_img and 0 <= y < h_img and w > 0 and h > 0 and x + w <= w_img and y + h <= h_img:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        frame = draw_text_hangul(frame, f"분석결과: {label_text}", (30, 30), font_size=32, color=(0,0,255))
        out.write(frame)
        cv2.imshow("Lie Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 결과 파일명 생성 및 파일명 변경 (폴더 포함)
    result_filename = os.path.join(RECORDED_DIR, f"liecam_{now}_{last_label}.avi")
    os.rename(temp_filename, result_filename)
    print(f"{result_filename} 파일로 저장되었습니다.")

def draw_text_hangul(img, text, position, font_size=32, color=(0,0,255)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("malgun.ttf", font_size)  # 윈도우 기본 한글 폰트
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main_menu():
    while True:
        print("\n==== 거짓말 탐지 앱 (Bag of Lies) ====")
        print("1. 특징 추출")
        print("2. 모델 학습")
        print("3. 실시간 거짓말 탐지(웹캠)")
        print("4. 웹캠만 보기") 
        print("0. 종료")
        sel = input("선택: ")
        if sel == "1":
            extract_all_features()
        elif sel == "2":
            train_model()
        elif sel == "3":
            run_camera_lie_detector()
        elif sel == "4":
            show_camera() 
        elif sel == "0":
            break
        else:
            print("잘못된 선택입니다.")



if __name__ == "__main__":
    main_menu()