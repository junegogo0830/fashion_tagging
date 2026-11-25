"""Test 30장 공유 패키지 — EfficientNet 팀원과 같은 파일 형식으로 만든다.

입력: predict.py 결과 폴더(test30_pred_s42 등)와 공유할 모델 하나
출력 (<out>/):
  probabilities.csv     file_name + 27개 클래스 확률 (학습과 같은 클래스 순서)
  predictions.jsonl     {"test_id": ..., "labels": [...]} 한 줄에 한 이미지
  test_predictions.npz  probs(30x27), file_names, classes
  manifest.json         모델·체크포인트 해시·전처리·threshold·이미지 해시 등 실행 기록
  review.html           이미지별 예측과 비교 이미지(visualizations/) 함께 보기
  visualizations/       predict.py의 비교 이미지(모델별 예측 나란히)
  contact_1.jpg ~ contact_5.jpg  30장 요약 보기 (6장씩)

사용법:
  ~/venv_fashion/bin/python make_share_package.py --pred_dir test30_pred_s42 --name E5 \
      --run common_E5_s42/tinyvit_21m --img_dir ~/data/examples_labeled/images --out share_tinyvit_E5_test30
"""
import argparse, glob, hashlib, html, json, os, shutil, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile, ImageFont

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import CATEGORIES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir", required=True, help="predict.py 출력 폴더")
ap.add_argument("--name", required=True, help="predict.py에서 쓴 모델 이름 (예: E5)")
ap.add_argument("--run", required=True, help="best.pt 와 result.json 이 있는 폴더")
ap.add_argument("--img_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="TinyViT-21M")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


probs = pd.read_csv(os.path.join(args.pred_dir, f"{args.name}_probs.csv"))
labels = pd.read_csv(os.path.join(args.pred_dir, f"{args.name}_labels.csv"))
assert list(probs.columns[1:]) == CATEGORIES, "클래스 순서가 학습과 다릅니다"
info = json.load(open(os.path.join(args.run, "result.json"), encoding="utf-8"))
r = info["result"]
files = probs.file_name.astype(str).tolist()
lab = {f: ([] if pd.isna(s) or s == "" else str(s).split(";")) for f, s in zip(labels.file_name.astype(str), labels["labels"])}

# 1) probabilities.csv (EfficientNet 팀원과 같은 utf-8-sig, 쉼표가 든 클래스명은 따옴표)
probs.to_csv(os.path.join(args.out, "probabilities.csv"), index=False, encoding="utf-8-sig")
# 2) predictions.jsonl
with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as f:
    for fn in files:
        f.write(json.dumps({"test_id": os.path.splitext(fn)[0], "labels": lab[fn]}, ensure_ascii=False) + "\n")
# 3) test_predictions.npz
np.savez_compressed(os.path.join(args.out, "test_predictions.npz"), probs=probs[CATEGORIES].values.astype(np.float32),
                    file_names=np.array(files), classes=np.array(CATEGORIES))
# 4) manifest.json
import timm  # noqa: E402

cfg = timm.create_model(r["checkpoint"], pretrained=False).pretrained_cfg
manifest = dict(
    model=f"{args.title} ({args.name})", checkpoint=os.path.join(args.run, "best.pt"),
    checkpoint_sha256=sha256(os.path.join(args.run, "best.pt")), checkpoint_epoch=r["best_epoch"],
    timm_name=r["checkpoint"], seed=r["seed"],
    model_config=dict(head=r.get("head"), loss=r.get("loss"), num_classes=27, csra_lambda=r.get("csra_lam"), csra_T=r.get("csra_T")),
    input=dict(width=r["img_size"], height=r.get("img_h") or r["img_size"], resize="aspect-preserving letterbox", pad_color=[124, 116, 104]),
    normalization=dict(mean=list(cfg.get("mean", [])), std=list(cfg.get("std", []))),
    threshold="per-class", thresholds=info["thresholds"],
    threshold_source="val 1,158장 전체에서 클래스별 F1 최대값으로 선택. test 기반 튜닝 없음",
    class_order=CATEGORIES, params_M=r["params_M"], gflops=r.get("gflops"),
    note="Test 이미지는 정성 확인·형식 점검용. test3k_000337.png는 파일 끝이 잘려 있어 PIL LOAD_TRUNCATED_IMAGES=True로 읽음",
    images=[dict(file_name=fn, sha256=sha256(os.path.join(args.img_dir, fn))) for fn in files],
)
json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
# 5) visualizations + review.html
vdir = os.path.join(args.out, "visualizations")
os.makedirs(vdir, exist_ok=True)
for p in glob.glob(os.path.join(args.pred_dir, "vis", "*.png")):
    shutil.copy(p, vdir)
arts = "".join(
    f"<article><h3>{html.escape(os.path.splitext(fn)[0])}</h3><p><b>{html.escape(args.name)}:</b> "
    f"{html.escape(', '.join(lab[fn]) or '[]')}</p><img src='visualizations/{html.escape(os.path.splitext(fn)[0])}.png'></article>"
    for fn in files)
open(os.path.join(args.out, "review.html"), "w", encoding="utf-8").write(
    f"<!doctype html><meta charset='utf-8'><title>{html.escape(args.title)} test 30</title>"
    "<style>body{font-family:sans-serif;max-width:1400px;margin:auto}img{width:100%}article{border-bottom:1px solid #bbb;padding:8px 0}</style>"
    f"<h1>{html.escape(args.title)} {html.escape(args.name)}: 30-image predictions</h1>"
    "<p>Per-class thresholds tuned on validation only. Qualitative review; test labels not used for tuning. "
    "Colors in images: green TP, red FP, blue FN, grey below threshold.</p>" + arts)
# 6) contact sheets (6장씩 5장)
try:
    font = ImageFont.load_default(size=15)
except TypeError:
    font = ImageFont.load_default()
CW, CH, TXT = 360, 360, 90
for k in range(0, len(files), 6):
    sheet = Image.new("RGB", (CW * 3, (CH + TXT) * 2), "white")
    d = ImageDraw.Draw(sheet)
    for j, fn in enumerate(files[k:k + 6]):
        im = Image.open(os.path.join(args.img_dir, fn)).convert("RGB")
        im.thumbnail((CW - 10, CH - 10))
        x0, y0 = (j % 3) * CW, (j // 3) * (CH + TXT)
        sheet.paste(im, (x0 + (CW - im.width) // 2, y0 + 5))
        words, lines, cur = (", ".join(lab[fn]) or "[]").split(" "), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 44:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        d.text((x0 + 6, y0 + CH), os.path.splitext(fn)[0], fill=(20, 20, 20), font=font)
        for li, t in enumerate(lines[:4]):
            d.text((x0 + 6, y0 + CH + 18 * (li + 1)), t, fill=(60, 60, 60), font=font)
    sheet.save(os.path.join(args.out, f"contact_{k // 6 + 1}.jpg"), quality=90)
print(f"저장: {args.out}/ ({len(files)}장) — " + ", ".join(sorted(os.listdir(args.out))))
