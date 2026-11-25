"""공통 Validation 평가: 한 실험의 val 예측 확률(val_probs.csv)로 세부 분석을 만든다.

사용법:
  python analyze_val.py --run_dir common_E0_s42/tinyvit_21m \
      --val_csv ~/data/val_zip/MAT4091_fashion_val/val/labels.csv --val_img_dir ~/data/val_zip/MAT4091_fashion_val/val/images

Threshold 보정 절차 (모든 모델 동일):
  - held-out: val을 SPLIT_SEED로 반 나눠 A에서 클래스별 threshold를 고르고 B에서 평가 (모델 비교용 숫자)
  - final: val 전체로 클래스별 threshold를 다시 고름 (Test 추론용; 이 threshold로 낸 val 점수는 낙관적)
출력 (run_dir/analysis/):
  summary.json, per_class.csv, per_image_errors.csv, by_aspect.csv, by_label_count.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import CATEGORIES, SPLIT_SEED, f1_scores, mean_ap, read_label_csv, tune_thresholds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True, help="result.json 과 val_probs.csv 가 있는 폴더")
ap.add_argument("--val_csv", required=True)
ap.add_argument("--val_img_dir", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or os.path.join(args.run_dir, "analysis")
os.makedirs(out, exist_ok=True)

probs_df = pd.read_csv(os.path.join(args.run_dir, "val_probs.csv"))
ids, y_all = read_label_csv(args.val_csv)
lab = dict(zip(ids, y_all))
names = probs_df["file_name"].astype(str).tolist()
P = probs_df[CATEGORIES].values
Y = np.stack([lab[n] for n in names]).astype(np.float32)

# ---- threshold 보정 (모든 모델 같은 절차)
idx = np.random.RandomState(SPLIT_SEED).permutation(len(Y))
a, b = idx[: len(Y) // 2], idx[len(Y) // 2:]
th_a = tune_thresholds(Y[a], P[a])
heldout, _, _, _ = f1_scores(Y[b], P[b] >= th_a)
th = tune_thresholds(Y, P)
pred = P >= th
full, f1c, prec, rec = f1_scores(Y, pred)
mAP, aps = mean_ap(Y, P)
summary = dict(run_dir=args.run_dir, n_val=len(Y), mAP=round(mAP, 4),
               heldout={k: round(v, 4) for k, v in heldout.items()},
               full_val_optimistic={k: round(v, 4) for k, v in full.items()},
               thresholds=dict(zip(CATEGORIES, map(float, th))))
json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---- 클래스별 AP / P / R / F1 (val 전체 threshold 기준)
pd.DataFrame(dict(category=CATEGORIES, n_pos=Y.sum(0).astype(int), AP=aps, threshold=th,
                  precision=prec, recall=rec, f1=f1c, n_fp=(pred & (Y == 0)).sum(0), n_fn=(~pred & (Y == 1)).sum(0))
             ).round(4).to_csv(os.path.join(out, "per_class.csv"), index=False)

# ---- 이미지별 FP / FN (실패 사례 분석용, 오류 많은 순)
rows = []
for i, n in enumerate(names):
    fp = [c for c, p, y in zip(CATEGORIES, pred[i], Y[i]) if p and not y]
    fn = [c for c, p, y in zip(CATEGORIES, pred[i], Y[i]) if y and not p]
    rows.append(dict(file_name=n, n_true=int(Y[i].sum()), n_pred=int(pred[i].sum()), n_fp=len(fp), n_fn=len(fn),
                     fp="; ".join(fp), fn="; ".join(fn)))
err = pd.DataFrame(rows)
err["n_err"] = err.n_fp + err.n_fn
err.sort_values("n_err", ascending=False).to_csv(os.path.join(out, "per_image_errors.csv"), index=False)

# ---- 그룹별 성능: 종횡비, 이미지당 라벨 수
def find(n):
    for e in ["", ".jpg", ".jpeg", ".png", ".webp"]:
        p = os.path.join(args.val_img_dir, n + e)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(n)

ratio = []
for n in names:
    with Image.open(find(n)) as im:
        w, h = im.size
    ratio.append(h / w)
ratio = np.array(ratio)
aspect = np.where(ratio > 1.1, "portrait (h/w>1.1)", np.where(ratio < 0.9, "landscape (h/w<0.9)", "square (0.9~1.1)"))
nlab = Y.sum(1).astype(int)
count = np.array([str(k) if k < 5 else "5+" for k in nlab])

def by(group, fname):
    out_rows = []
    for g in sorted(set(group)):
        m = group == g
        r, _, _, _ = f1_scores(Y[m], pred[m])
        out_rows.append(dict(group=g, n_images=int(m.sum()), **{k: round(v, 4) for k, v in r.items()},
                             fp_per_img=round((pred[m] & (Y[m] == 0)).sum() / m.sum(), 3),
                             fn_per_img=round((~pred[m] & (Y[m] == 1)).sum() / m.sum(), 3)))
    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(out, fname), index=False)
    return df

print(json.dumps({k: v for k, v in summary.items() if k != "thresholds"}, ensure_ascii=False, indent=2))
print("\n== 종횡비별 (val 전체 threshold) ==\n" + by(aspect, "by_aspect.csv").to_string(index=False))
print("\n== 이미지당 라벨 수별 ==\n" + by(count, "by_label_count.csv").to_string(index=False))
print(f"\n저장: {out}")
