"""팀원(ConvNeXt V2 보고서)과 같은 방식으로 TinyViT 결과를 다시 채점한다.

Part A (CPU, val_probs.csv만 사용) — 모든 common_* 실험:
  - Val 전체 @0.5: threshold 0.5 고정 Micro / Macro / Image F1, mAP
  - Holdout: val을 multi-label 층화로 50:50 분할 (분할 seed 5개 x 양방향 = 10회).
             한쪽에서 27개 클래스 공통 threshold 하나를 고르고(Micro F1 최대), 다른 쪽에서 평가한 값의 평균 ± 표준편차
  - 사진 방향별 Micro F1 @0.5: 세로 w/h<0.87, 정사각형에 가까움, 가로 w/h>1.15 (팀원과 같은 경계)
Part B (GPU, --tile_runs 로 지정한 실험만) — 분할 추론:
  - 전체 이미지 + 2x2 조각(각 조각은 원본 가로·세로의 60%, 서로 겹침)
  - 판정 규칙: "전체 이미지 확률 >= 0.5 또는 2x2 조각 최댓값 >= t2" (t2 = 0.999, 0.995, 0.99, 0.95)
  - 평가 대상: val 1,158장 / 가짜 단체 사진 300장(val 2장 또는 4장 붙이기, 라벨은 합집합) / Test 30장(참고용)
  - Test 30장은 확인용으로만 쓰며 t2 선택에 쓰지 않는다.

사용법 (서버):
  CUDA_VISIBLE_DEVICES=3 ~/venv_fashion/bin/python team_eval.py \
      --tile_runs common_E2_s42 common_E5_s42 --test_dir ~/data/examples_labeled
결과: team_eval/summary.csv, team_eval/holdout_detail.csv, team_eval/aspect.csv, team_eval/tiling.csv
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import CATEGORIES, f1_scores, mean_ap, read_label_csv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--val_dir", default=os.path.expanduser("~/data/val_zip/MAT4091_fashion_val/val"))
ap.add_argument("--test_dir", default=os.path.expanduser("~/data/examples_labeled"), help="labels.csv + images (Test 30장)")
ap.add_argument("--tile_runs", nargs="*", default=[], help="분할 추론을 할 실험 폴더 (예: common_E5_s42)")
ap.add_argument("--out", default="team_eval")
ap.add_argument("--n_fake", type=int, default=300)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
EXTS = ("", ".jpg", ".jpeg", ".png", ".webp")


def find(img_dir, name):
    for e in EXTS:
        p = os.path.join(img_dir, name + e)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(os.path.join(img_dir, name))


def metrics(Y, P):
    r, _, _, _ = f1_scores(Y, P)
    return r


# ----------------------------------------------------------------------------- multi-label 층화 50:50 분할
def strat_halves(Y, seed):
    """multi-label 층화 50:50. iterative-stratification 패키지가 있으면 그것을, 없으면 같은 알고리즘의 간이 구현을 쓴다."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        _, b = next(MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(np.zeros(len(Y)), Y))
        fold = np.zeros(len(Y), dtype=int); fold[b] = 1
        return fold
    except ImportError:
        pass
    rng = np.random.RandomState(seed)
    n = len(Y)
    fold = -np.ones(n, dtype=int)
    need = np.array([n / 2.0, n / 2.0])
    need_l = np.tile(Y.sum(0) / 2.0, (2, 1))
    left = np.ones(n, dtype=bool)
    while left.any():
        cnt = Y[left].sum(0)
        if cnt.sum() == 0:  # 라벨 없는 이미지: 남은 자리대로 무작위 배정
            for s in rng.permutation(np.where(left)[0]):
                f = int(np.argmax(need)); fold[s] = f; need[f] -= 1
            break
        l = int(np.argmin(np.where(cnt > 0, cnt, np.inf)))
        for s in rng.permutation(np.where(left & (Y[:, l] == 1))[0]):
            d = need_l[:, l]
            f = int(np.argmax(d)) if d[0] != d[1] else (int(np.argmax(need)) if need[0] != need[1] else rng.randint(2))
            fold[s] = f; need_l[f] -= Y[s]; need[f] -= 1; left[s] = False
    return fold


GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def holdout(Y, Pr):
    rows = []
    for seed in range(5):
        fold = strat_halves(Y, seed)
        for a in (0, 1):
            A, B = fold == a, fold != a
            t = max(GRID, key=lambda t: metrics(Y[A], Pr[A] >= t)["micro_f1"])
            rows.append(dict(split_seed=seed, tune_half=a, threshold=float(t), **metrics(Y[B], Pr[B] >= t)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Part A
val_ids, val_y = read_label_csv(os.path.join(args.val_dir, "labels.csv"))
val_lab = {str(i): v for i, v in zip(val_ids, val_y)}
val_img_dir = os.path.join(args.val_dir, "images")
ratio = {}
for i in val_ids:
    with Image.open(find(val_img_dir, str(i))) as im:
        ratio[str(i)] = im.size[0] / im.size[1]  # w/h


def orient(wh):
    return "portrait (w/h<0.87)" if wh < 0.87 else ("landscape (w/h>1.15)" if wh > 1.15 else "near-square")


summ, det, asp = [], [], []
for d in sorted(glob.glob("common_*_s*/*/val_probs.csv")):
    run = d.split(os.sep)[0]
    m = re.match(r"common_(?:(.+)_)?(E\d[a-z]?)_s(\d+)$", run)
    res = json.load(open(os.path.join(os.path.dirname(d), "result.json"), encoding="utf-8"))["result"]
    df = pd.read_csv(d)
    names = df.file_name.astype(str).tolist()
    Pr = df[CATEGORIES].values
    Y = np.stack([val_lab[n] for n in names]).astype(np.float32)
    at05 = metrics(Y, Pr >= 0.5)
    h = holdout(Y, Pr)
    h.insert(0, "run", run); det.append(h)
    summ.append(dict(run=run, model=res["model"], exp=m.group(2), seed=int(m.group(3)), best_epoch=res["best_epoch"],
                     mAP=round(mean_ap(Y, Pr)[0], 4), **{f"{k}@0.5": round(v, 4) for k, v in at05.items()},
                     holdout_micro=round(h.micro_f1.mean(), 4), holdout_micro_sd=round(h.micro_f1.std(ddof=1), 4),
                     holdout_macro=round(h.macro_f1.mean(), 4), holdout_image=round(h.image_f1.mean(), 4),
                     holdout_t_median=float(h.threshold.median()), params_M=res["params_M"], gflops=res.get("gflops"),
                     latency_ms=res["latency_ms_bs1"]))
    o = np.array([orient(ratio[n]) for n in names])
    for g in sorted(set(o)):
        k = o == g
        asp.append(dict(run=run, group=g, n_images=int(k.sum()), **{f"{a}@0.5": round(b, 4) for a, b in metrics(Y[k], Pr[k] >= 0.5).items()}))
    print(f"[A] {run}: mAP {summ[-1]['mAP']}, micro@0.5 {at05['micro_f1']:.4f}, holdout micro {summ[-1]['holdout_micro']} ± {summ[-1]['holdout_micro_sd']}")

pd.DataFrame(summ).to_csv(os.path.join(args.out, "summary.csv"), index=False)
pd.concat(det).to_csv(os.path.join(args.out, "holdout_detail.csv"), index=False)
pd.DataFrame(asp).to_csv(os.path.join(args.out, "aspect.csv"), index=False)

# ----------------------------------------------------------------------------- Part B: 분할 추론
if args.tile_runs:
    import torch
    from benchmark import Transform, build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 가짜 단체 사진: val 2장(가로로 붙임) 또는 4장(2x2)
    rng = np.random.RandomState(0)
    fake = []
    for k in range(args.n_fake):
        n = 2 if k < args.n_fake // 2 else 4
        pick = [str(val_ids[j]) for j in rng.choice(len(val_ids), n, replace=False)]
        fake.append((f"fake{k:03d}", pick))

    def load_fake(pick):
        ims = [Image.open(find(val_img_dir, p)).convert("RGB") for p in pick]
        ims = [im.resize((max(1, round(im.width * 512 / im.height)), 512)) for im in ims]
        rows = [ims[:2], ims[2:]] if len(ims) == 4 else [ims]
        W = max(sum(i.width for i in r) for r in rows)
        canvas = Image.new("RGB", (W, 512 * len(rows)), (124, 116, 104))
        for ri, r in enumerate(rows):
            x = 0
            for im in r:
                canvas.paste(im, (x, 512 * ri)); x += im.width
        return canvas

    def tiles(img, frac=0.6):
        w, h = img.size
        tw, th = int(w * frac), int(h * frac)
        return [img.crop((x, y, x + tw, y + th)) for x in (0, w - tw) for y in (0, h - th)]

    test_ids, test_y = read_label_csv(os.path.join(args.test_dir, "labels.csv"))
    test_img_dir = os.path.join(args.test_dir, "images")
    sets = {
        "val": [(str(i), lambda i=i: Image.open(find(val_img_dir, str(i))).convert("RGB")) for i in val_ids],
        "fake_group": [(n, lambda p=p: load_fake(p)) for n, p in fake],
        "test30": [(str(i), lambda i=i: Image.open(find(test_img_dir, os.path.splitext(str(i))[0])).convert("RGB")) for i in test_ids],
    }
    Ys = {"val": val_y.astype(np.float32),
          "fake_group": np.stack([np.max([val_lab[p] for p in pick], 0) for _, pick in fake]).astype(np.float32),
          "test30": test_y.astype(np.float32)}

    rows = []
    for run_name in args.tile_runs:
        rd = glob.glob(os.path.join(run_name, "*", "result.json"))[0]
        info = json.load(open(rd, encoding="utf-8")); r = info["result"]
        model = build_model(r["model"], r["img_size"], pretrained=False, head=r.get("head", "linear"),
                            csra_lam=r.get("csra_lam", 0.1), csra_T=r.get("csra_T", 1.0))
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(rd), "best.pt"), map_location="cpu"))
        model.to(device).eval()
        cfg = model.pretrained_cfg
        h_ = r.get("img_h") or r["img_size"]
        size = r["img_size"] if h_ == r["img_size"] else (r["img_size"], h_)
        tf = Transform(size, cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225)), train=False)
        for sname, items in sets.items():
            t0 = time.time()
            full, tmax = [], []
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                for _, loader in items:
                    img = loader()
                    x = torch.stack([tf(im) for im in [img] + tiles(img)]).to(device)
                    p = torch.sigmoid(model(x).float()).cpu().numpy()
                    full.append(p[0]); tmax.append(p[1:].max(0))
            full, tmax = np.stack(full), np.stack(tmax)
            np.savez_compressed(os.path.join(args.out, f"tile_probs_{run_name}_{sname}.npz"), full=full, tile_max=tmax,
                                names=np.array([n for n, _ in items]), classes=np.array(CATEGORIES))
            Y = Ys[sname]
            for t2 in [None, 0.999, 0.995, 0.99, 0.95]:
                P = (full >= 0.5) if t2 is None else ((full >= 0.5) | (tmax >= t2))
                mm = metrics(Y, P)
                rows.append(dict(run=run_name, set=sname, t2="none" if t2 is None else t2,
                                 **{k: round(v, 4) for k, v in mm.items()},
                                 precision=round(float((P & (Y == 1)).sum() / max(P.sum(), 1)), 4),
                                 recall=round(float((P & (Y == 1)).sum() / max(Y.sum(), 1)), 4)))
            print(f"[B] {run_name} {sname}: {len(items)}장, {time.time() - t0:.0f}s")
        del model
    tdf = pd.DataFrame(rows)
    tdf.to_csv(os.path.join(args.out, "tiling.csv"), index=False)
    print(tdf.to_string(index=False))

print("\n== summary ==")
print(pd.DataFrame(summ).sort_values(["exp", "seed"]).to_string(index=False))
print("\n== aspect (seed 42) ==")
a = pd.DataFrame(asp)
print(a[a.run.str.endswith("_s42")].to_string(index=False))
print(f"\n저장: {args.out}/")
