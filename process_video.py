import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_rate=15):
    """
    MP4 영상에서 일정 간격으로 프레임을 추출하여 저장하는 함수.

    Parameters:
    - video_path (str): 입력 동영상 파일 경로.
    - output_folder (str): 프레임 저장 폴더.
    - frame_rate (int): 초당 저장할 프레임 개수 (기본값: 15).
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 비디오 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 영상 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 영상의 FPS
    frame_interval = int(fps / frame_rate)  # 저장할 프레임 간격
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수

    frame_id = 0  # 저장된 프레임 번호
    count = 0  # 전체 프레임 카운터

    print(f"🎬 FPS: {fps}, Frame Interval: {frame_interval}, Total Frames: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 지정된 프레임 간격에 맞춰 저장
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
            print(f"🖼️ 저장됨: {frame_path}")

        count += 1

    cap.release()
    print(f"✅ 총 {frame_id}개의 프레임이 추출되었습니다.")

# 터미널 실행 옵션 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP4 영상을 프레임 단위로 추출하는 프로그램.")
    parser.add_argument("--video", type=str, required=True, help="입력 MP4 파일 경로")
    parser.add_argument("--output", type=str, default="frames/", help="프레임 저장 폴더")
    parser.add_argument("--fps", type=int, default=1, help="초당 저장할 프레임 개수 (기본값: 1)")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps)
