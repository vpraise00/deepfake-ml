import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import sys

# ✅ Dummy 모듈 대체 클래스 (timm.layers 전체를 대체)
class DummyModule:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x  # 입력 그대로 반환

# ✅ "timm.layers"와 하위 모듈까지 모두 Dummy로 대체
sys.modules['timm.layers'] = DummyModule()
sys.modules['timm.layers.norm_act'] = DummyModule()  # ✅ 추가된 부분

# ✅ 모델 불러오기
MODEL_PATH = "model/CelebDF_8epoch.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
except ModuleNotFoundError as e:
    print(f"⚠️ 오류 발생: {e}")
    model = None  # 오류 발생 시 모델이 None으로 초기화

# ✅ 모델이 정상적으로 로드된 경우에만 실행
if model:
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()

    # ✅ 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def load_images_in_batches(image_paths, batch_size=16):
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            images = [transform(Image.open(img_path).convert("RGB")) for img_path in batch_paths]
            yield torch.stack(images).to(device), batch_paths

    def predict(images_tensor):
        with torch.no_grad():
            outputs = model(images_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
        return probabilities

    def process_all_frames(input_folder, output_file, batch_size=16):
        image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg")])
        if not image_files:
            print("❌ 이미지 파일이 없습니다.")
            return

        results = {}
        for batch_images, batch_paths in tqdm(load_images_in_batches(image_files, batch_size), total=len(image_files) // batch_size + 1, desc="🔍 Processing"):
            probs = predict(batch_images)
            for path, prob in zip(batch_paths, probs):
                frame_name = os.path.basename(path)
                results[frame_name] = float(prob)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ 예측 결과 저장 완료: {output_file}")

    # ✅ 실행 예시
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description="딥페이크 프레임 분석기 (CelebDF_8epoch 모델)")
        parser.add_argument("--input", type=str, required=True, help="프레임이 저장된 폴더 경로")
        parser.add_argument("--output", type=str, default="results/deepfake_results_8epoch.json", help="결과 저장 파일")
        parser.add_argument("--batch_size", type=int, default=16, help="배치 크기 (기본값: 16)")

        args = parser.parse_args()
        process_all_frames(args.input, args.output, args.batch_size)
else:
    print("❌ 모델 로드 실패: timm.layers 관련 오류로 인해 모델을 불러올 수 없습니다.")
