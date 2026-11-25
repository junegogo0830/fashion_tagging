"""Test 이미지 추론 + 모델별 예측 비교 시각화 (정성 확인·추론 파이프라인 점검용).

사용법 (모델 여러 개를 한 번에 비교 가능):
  python predict.py --img_dir ~/data/test30/images \
      --run common_E0_s42/tinyvit_21m --run common_E2_s42/tinyvit_21m \
      --out_dir test30_pred [--labels_csv ~/data/test30/labels.csv]

- 각 모델은 result.json에 기록된 학습 당시 전처리(해상도·캔버스·정규화)와 저장된 클래스별 threshold를 그대로 쓴다.
  threshold를 다시 고르지 않는다 (Test로 튜닝 금지).
출력:
  <out_dir>/<모델이름>_probs.csv   (이미지 수 x 27, CATEGORIES 순서)
  <out_dir>/<모델이름>_labels.csv  (file_name, 예측 라벨 ';' 구분)
  <out_dir>/vis/<이미지>.png       (원본 + 모델별 예측 나란히; 정답이 있으면 TP/FP/FN 표시)
  <out_dir>/agreement.csv          (모델 2개 이상일 때 이미지별 모델 간 불일치 클래스)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFile, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 끝이 잘린 이미지도 최대한 읽기 (예: test3k_000337.png)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import CATEGORIES, Transform, build_model, read_label_csv  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp")
ap = argparse.ArgumentParser()
ap.add_argument("--img_dir", required=True)
ap.add_argument("--run", action="append", required=True, help="best.pt 와 result.json 이 있는 폴더 (여러 번 지정 가능)")
ap.add_argument("--name", action="append", default=None, help="모델 표시 이름 (--run 순서대로, 생략 시 폴더 이름)")
ap.add_argument("--out_dir", default="test_pred")
ap.add_argument("--labels_csv", default=None, help="정답 라벨이 있으면 TP/FP/FN 표시")
args = ap.parse_args()
os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

files = sorted(f for f in os.listdir(args.img_dir) if f.lower().endswith(EXTS))
print(f"이미지 {len(files)}장, device={device}")
gt = None
if args.labels_csv:
    ids, y = read_label_csv(args.labels_csv)
    lab = {}
    for i, v in zip(ids, y):
        lab[i] = v
        lab[os.path.splitext(i)[0]] = v
    gt = {f: lab.get(f, lab.get(os.path.splitext(f)[0])) for f in files}

names = args.name or [os.path.basename(os.path.dirname(os.path.normpath(r))) or r for r in args.run]
assert len(names) == len(args.run), "--name 개수는 --run 개수와 같아야 합니다"
preds = {}
for name, run in zip(names, args.run):
    info = json.load(open(os.path.join(run, "result.json"), encoding="utf-8"))
    r = info["result"]
    th = np.array([info["thresholds"][c] for c in CATEGORIES])
    model = build_model(r["model"], r["img_size"], pretrained=False, head=r.get("head", "linear"),
                        csra_lam=r.get("csra_lam", 0.1), csra_T=r.get("csra_T", 1.0))
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cpu"))
    model.to(device).eval()
    cfg = model.pretrained_cfg
    h = r.get("img_h") or r["img_size"]
    size = r["img_size"] if h == r["img_size"] else (r["img_size"], h)
    tf = Transform(size, cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225)), train=False)
    P = []
    with torch.no_grad():
        for f in files:
            x = tf(Image.open(os.path.join(args.img_dir, f))).unsqueeze(0).to(device)
            P.append(torch.sigmoid(model(x).float()).cpu().numpy()[0])
    P = np.stack(P)
    pred = P >= th
    pd.DataFrame(P, columns=CATEGORIES).assign(file_name=files)[["file_name"] + CATEGORIES] \
        .to_csv(os.path.join(args.out_dir, f"{name}_probs.csv"), index=False, float_format="%.6f")
    pd.DataFrame(dict(file_name=files, labels=[";".join(c for c, p in zip(CATEGORIES, row) if p) for row in pred])) \
        .to_csv(os.path.join(args.out_dir, f"{name}_labels.csv"), index=False)
    assert P.shape == (len(files), 27), "출력 형식 오류"
    preds[name] = (P, pred, th)
    print(f"{name}: {r['img_size']}x{h}, head={r.get('head')}, 평균 예측 라벨 수 {pred.sum(1).mean():.2f}")
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

# ---- 시각화: 원본 | 모델1 | 모델2 ...
def font(sz):
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()

F, FB = font(15), font(17)
PH, PW, LH = 560, 300, 20
GREEN, RED, BLUE, BLACK, GREY = (20, 130, 60), (200, 40, 40), (40, 90, 200), (20, 20, 20), (120, 120, 120)
for i, f in enumerate(files):
    im = Image.open(os.path.join(args.img_dir, f)).convert("RGB")
    im = im.resize((max(1, round(im.width * PH / im.height)), PH))
    canvas = Image.new("RGB", (im.width + PW * len(preds), PH + 40), "white")
    canvas.paste(im, (0, 40))
    d = ImageDraw.Draw(canvas)
    d.text((8, 10), f, fill=BLACK, font=FB)
    y_true = gt.get(f) if gt else None
    for k, (name, (P, pred, th)) in enumerate(preds.items()):
        x0 = im.width + PW * k + 12
        d.text((x0, 10), name, fill=BLACK, font=FB)
        yy = 48
        order = np.argsort(-P[i])
        for c in order:
            is_pred, is_true = pred[i, c], (y_true is not None and y_true[c] == 1)
            if not (is_pred or is_true or P[i, c] >= 0.5 * th[c]):
                continue
            if y_true is None:
                col, tag = (BLACK, "") if is_pred else (GREY, "  (below th)")
            elif is_pred and is_true:
                col, tag = GREEN, "  TP"
            elif is_pred:
                col, tag = RED, "  FP"
            elif is_true:
                col, tag = BLUE, "  FN"
            else:
                col, tag = GREY, "  (below th)"
            d.text((x0, yy), f"{CATEGORIES[c][:22]}  {P[i, c]:.2f}{tag}", fill=col, font=F)
            yy += LH
            if yy > PH + 20:
                break
    canvas.save(os.path.join(args.out_dir, "vis", os.path.splitext(f)[0] + ".png"))

# ---- 모델 간 일치·불일치
if len(preds) > 1:
    stack = np.stack([p for _, p, _ in preds.values()])  # (M, N, 27)
    rows = []
    for i, f in enumerate(files):
        dis = [CATEGORIES[c] + "(" + ",".join(n for n, (_, p, _) in preds.items() if p[i, c]) + ")"
               for c in range(27) if 0 < stack[:, i, c].sum() < len(preds)]
        rows.append(dict(file_name=f, n_disagree=len(dis), disagree="; ".join(dis)))
    pd.DataFrame(rows).sort_values("n_disagree", ascending=False).to_csv(os.path.join(args.out_dir, "agreement.csv"), index=False)
print(f"저장: {args.out_dir}")
