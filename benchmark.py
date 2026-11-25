"""
Fashion Category Tagging — Backbone benchmark (Phase A)

4개 백본(ConvNeXt V2-T, EfficientNetV2-S, EVA-02 S, TinyViT-21M)을
같은 데이터·손실·학습 예산으로 학습하고 같은 지표로 평가한다.

사용 예)
  # 0) 데이터 없이 파라미터 수만 확인 (가중치 다운로드 없음)
  python benchmark.py --count_params

  # 1) 파이프라인 점검 (가짜 이미지 32장, 1 epoch)
  python benchmark.py --smoke --models convnextv2_t

  # 2) 실제 벤치마크
  python benchmark.py --train_csv data/train.csv --train_img_dir data/train \
                      --val_csv data/val.csv --val_img_dir data/val \
                      --models all --epochs 10 --img_size 384

CSV 형식 (둘 중 아무거나 자동 인식)
  (a) 파일명 열 + 27개 카테고리 열(0/1). 열 이름은 아래 CATEGORIES와 정확히 같아야 함
  (b) 파일명 열 + 'labels' 열. 값은 카테고리 이름을 ';'로 구분 (예: "jacket;pants;shoe"), 빈 값 = 라벨 없음
  파일명 열 이름은 --id_col 로 지정 (기본: 자동 탐색 image / file / filename / image_id / id)
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import pandas as pd
except ImportError:  # pandas는 실제 데이터를 쓸 때만 필요
    pd = None

CATEGORIES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
    "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe",
    "bag, wallet", "scarf", "umbrella",
]
NUM_CLASSES = len(CATEGORIES)
# threshold 보정용 val 반 나누기 seed. 학습 seed와 무관하게 고정 → 모든 모델·seed가 같은 절반으로 threshold를 고르고 같은 절반으로 평가
SPLIT_SEED = 0
assert NUM_CLASSES == 27

# timm 모델 이름과 사전학습 체크포인트. 체크포인트를 바꾸면 결과표에 반드시 기록할 것.
MODELS = {
    "convnextv2_t": dict(timm_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", native=384, fixed_size=False),
    "effnetv2_s":   dict(timm_name="tf_efficientnetv2_s.in21k_ft_in1k",       native=384, fixed_size=False),
    "eva02_s":      dict(timm_name="eva02_small_patch14_336.mim_in22k_ft_in1k", native=336, fixed_size=True),
    # TinyViT는 img_size 인자를 받지 않음 -> 해상도별 체크포인트(224/384/512) 중에서 고름
    "tinyvit_21m":  dict(timm_name="tiny_vit_21m_384.dist_in22k_ft_in1k",     native=384, fixed_size=False),
}


# --------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timm_name_for(key, img_size):
    if key == "tinyvit_21m":
        if img_size not in (224, 384, 512):
            raise SystemExit("TinyViT-21M은 --img_size 224, 384, 512 중 하나만 지원합니다.")
        return f"tiny_vit_21m_{img_size}.dist_in22k_ft_in1k"
    return MODELS[key]["timm_name"]


class QueryHead(nn.Module):
    """ML-Decoder / Query2Label 방식의 경량 분류 헤드.

    카테고리마다 학습 가능한 '질문 벡터(query)'를 하나씩 두고, cross-attention으로
    특징 맵에서 그 카테고리와 관련된 위치(손목, 발, 머리 등)를 골라 본 뒤 판단한다.
    전역 평균 풀링(GAP)은 사진 전체를 한 벡터로 뭉개서 작은 소품 정보가 묻히기 쉽다.
    """

    def __init__(self, in_dim, num_classes, dim=256, heads=8, ffn=1024, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.query = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * 0.02)  # 카테고리별 개별 출력층
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, tokens):  # tokens: (B, N, C)
        x = self.proj(tokens)
        q = self.query.unsqueeze(0).expand(x.size(0), -1, -1)
        a, _ = self.attn(q, x, x)
        h = self.norm1(q + a)
        h = self.norm2(h + self.ffn(h))
        return (h * self.weight).sum(-1) + self.bias


class CSRAHead(nn.Module):
    """CSRA: Class-Specific Residual Attention (Zhu & Wu, ICCV 2021).

    카테고리마다 특징 맵의 위치별 점수 s(위치, 클래스)를 만든 뒤
      logit = 평균 풀링 점수 + lam * (softmax(T*s)로 가중합한 점수)
    평균(GAP)은 그대로 두고, 클래스별로 점수가 높은 위치를 잔차로 더해 작은 물체 신호를 살린다.
    파라미터는 Linear 헤드와 거의 같다 (1x1 분류기 하나).
    """

    def __init__(self, in_dim, num_classes, lam=0.1, T=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, num_classes)
        self.lam, self.T = lam, T

    def forward(self, tokens):  # tokens: (B, N, C)
        s = self.fc(self.norm(tokens))  # (B, N, K) 위치별 클래스 점수
        base = s.mean(1)
        att = (torch.softmax(self.T * s.float(), dim=1) * s.float()).sum(1).to(s.dtype)
        return base + self.lam * att


class BackboneWithHead(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.pretrained_cfg = backbone.pretrained_cfg

    def forward(self, x):
        f = self.backbone.forward_features(x)
        if f.ndim == 4:  # CNN 계열 (B, C, H, W) -> (B, H*W, C)
            f = f.flatten(2).transpose(1, 2)
        return self.head(f)


def build_model(key, img_size, pretrained=True, head="linear", csra_lam=0.1, csra_T=1.0):
    import timm
    cfg = MODELS[key]
    kwargs = dict(pretrained=pretrained, num_classes=NUM_CLASSES if head == "linear" else 0)
    # ViT 계열은 입력 크기가 모델에 고정되므로, 다른 해상도를 쓰면 위치 임베딩을 보간한다
    if cfg["fixed_size"]:
        kwargs["img_size"] = img_size
    model = timm.create_model(timm_name_for(key, img_size), **kwargs)
    if head == "query":
        model = BackboneWithHead(model, QueryHead(model.num_features, NUM_CLASSES))
    elif head == "csra":
        model = BackboneWithHead(model, CSRAHead(model.num_features, NUM_CLASSES, csra_lam, csra_T))
    return model


def count_gflops(model, x):
    """입력 1장 기준 연산량(GFLOPs, 곱셈+덧셈 각각 1회로 셈). 측정 실패 시 None."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return round(fc.get_total_flops() / 1e9, 2)
    except Exception as e:  # noqa: BLE001
        print("FLOPs 측정 실패:", e)
        return None


def head_parameters(model):
    """새로 붙인 헤드의 파라미터 (백본보다 큰 학습률을 적용)."""
    if isinstance(model, BackboneWithHead):
        return list(model.head.parameters())
    return list(model.get_classifier().parameters())


class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al., ICCV 2021).

    이미지 한 장에 정답 태그는 3개쯤, 나머지 24개는 '없음'이다. BCE는 쉬운 '없음'까지
    똑같이 학습해서 드문 카테고리의 '있음' 신호가 묻힌다. ASL은
      - 이미 확실히 맞힌 '없음'의 손실을 줄이고 (gamma_neg)
      - 확률이 clip 이하인 '없음'은 아예 무시한다 (라벨 누락 대비).
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits.float())
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = y * torch.log(p.clamp(min=self.eps)) * (1 - p).pow(self.gp)
        loss_neg = (1 - y) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg.pow(self.gn)
        return -(loss_pos + loss_neg).sum(1).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- data
def letterbox(img, size, fill=(124, 116, 104)):
    """비율을 유지하며 캔버스 안에 최대한 크게 넣고 나머지는 패딩 (세로로 긴 패션 사진이 찌그러지지 않게).

    size: 정수면 정사각형 size×size, (W, H) 튜플이면 W×H 캔버스.
    """
    cw, ch = (size, size) if isinstance(size, int) else size
    w, h = img.size
    s = min(cw / w, ch / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (cw, ch), fill)
    canvas.paste(img, ((cw - nw) // 2, (ch - nh) // 2))
    return canvas


class Transform:
    """약한 Augmentation만 사용. 강한 크롭은 신발·모자를 잘라내 라벨을 틀리게 만들 수 있음."""

    def __init__(self, size, mean, std, train):
        self.size, self.train = size, train
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img):
        img = img.convert("RGB")
        if self.train:
            w, h = img.size
            s = random.uniform(0.85, 1.0)  # 가장자리 최대 15%만 잘라냄
            cw, ch = int(w * s), int(h * s)
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = letterbox(img, self.size)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (t - self.mean) / self.std


def read_label_csv(csv_path, id_col=None):
    if pd is None:
        raise SystemExit("pandas가 필요합니다: pip install pandas")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # 제공 CSV 첫 줄의 BOM 제거
    if id_col is None:
        for c in ["image", "file", "filename", "file_name", "image_id", "id", "img"]:
            if c in df.columns:
                id_col = c
                break
    if id_col is None or id_col not in df.columns:
        raise SystemExit(f"파일명 열을 찾지 못했습니다. --id_col 로 지정하세요. 열 목록: {list(df.columns)}")

    if all(c in df.columns for c in CATEGORIES):
        y = df[CATEGORIES].fillna(0).astype(np.float32).values
    elif "labels" in df.columns:
        y = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
        idx = {c: i for i, c in enumerate(CATEGORIES)}
        for r, v in enumerate(df["labels"].fillna("")):
            for name in [s.strip() for s in str(v).split(";") if s.strip()]:
                if name not in idx:
                    raise SystemExit(f"알 수 없는 카테고리 이름: '{name}' ({csv_path}, {r}행)")
                y[r, idx[name]] = 1.0
    else:
        missing = [c for c in CATEGORIES if c not in df.columns]
        raise SystemExit(f"라벨 열을 찾지 못했습니다. 없는 카테고리 열 예시: {missing[:5]}")
    return df[id_col].astype(str).tolist(), y


def keep_existing(ids, labels, img_dir, split):
    """라벨에는 있지만 이미지 파일이 없는 항목을 제외하고 개수를 알린다."""
    present = set(os.listdir(img_dir))
    mask = np.array([any((i + e) in present for e in TagDataset.EXTS) for i in ids])
    if not mask.all():
        print(f"!! {split}: 라벨 {len(ids)}개 중 이미지가 있는 {mask.sum()}개만 사용 ({(~mask).sum()}개 이미지 없음)")
    return [i for i, m in zip(ids, mask) if m], labels[mask]


class TagDataset(Dataset):
    EXTS = ["", ".jpg", ".jpeg", ".png", ".webp"]

    def __init__(self, ids, labels, img_dir, transform):
        self.ids, self.labels, self.img_dir, self.tf = ids, labels, img_dir, transform

    def _path(self, name):
        for ext in self.EXTS:
            p = os.path.join(self.img_dir, name + ext)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(os.path.join(self.img_dir, name))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self._path(self.ids[i]))
        return self.tf(img), torch.from_numpy(self.labels[i])


class SmokeDataset(Dataset):
    """설치·파이프라인 점검용 가짜 데이터."""

    def __init__(self, n, transform, seed=0):
        rng = np.random.RandomState(seed)
        self.y = (rng.rand(n, NUM_CLASSES) < 0.15).astype(np.float32)
        self.y[0] = 0  # 빈 라벨 이미지 포함
        self.sizes = [(int(rng.randint(300, 800)), int(rng.randint(400, 1000))) for _ in range(n)]
        self.tf = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w, h = self.sizes[i]
        img = Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8))
        return self.tf(img), torch.from_numpy(self.y[i])


# --------------------------------------------------------------------------- metrics
def f1_scores(y, pred):
    """y, pred: (N, C) 0/1. 과제의 Micro / Macro / Image-wise F1."""
    y, pred = y.astype(bool), pred.astype(bool)
    tp = (y & pred).sum(0).astype(float)
    fp = (~y & pred).sum(0).astype(float)
    fn = (y & ~pred).sum(0).astype(float)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12)
    denom = 2 * tp + fp + fn
    per_class = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), np.nan)  # 정답·예측 모두 없는 클래스는 제외
    macro = float(np.nanmean(per_class)) if np.any(~np.isnan(per_class)) else 0.0
    inter = (y & pred).sum(1)
    size = y.sum(1) + pred.sum(1)
    img_f1 = np.where(size == 0, 1.0, 2 * inter / np.maximum(size, 1))  # 둘 다 빈 경우 1
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
    return dict(micro_f1=float(micro), macro_f1=macro, image_f1=float(img_f1.mean())), per_class, prec, rec


def average_precision(y, s):
    """클래스 하나의 AP (Threshold 무관 지표)."""
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def mean_ap(y, probs):
    aps = np.array([average_precision(y[:, c], probs[:, c]) for c in range(y.shape[1])])
    return float(np.nanmean(aps)), aps


def tune_thresholds(y, probs, grid=np.arange(0.05, 0.951, 0.025)):
    """클래스별로 F1이 최대가 되는 Threshold. 양성이 없는 클래스는 0.5 유지."""
    th = np.full(y.shape[1], 0.5)
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        best, best_t = -1, 0.5
        for t in grid:
            p = probs[:, c] >= t
            tp = (p & (y[:, c] == 1)).sum()
            f1 = 2 * tp / max(p.sum() + y[:, c].sum(), 1)
            if f1 > best:
                best, best_t = f1, t
        th[c] = best_t
    return th


def evaluate_probs(y, probs, seed=0):
    out = {}
    m, per_class, prec, rec = f1_scores(y, probs >= 0.5)
    out["at_0.5"] = m
    out["mAP"], aps = mean_ap(y, probs)

    # Validation을 반으로 나눠 A에서 Threshold를 고르고 B에서 평가 (같은 데이터로 고르고 평가하면 낙관적)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    th_a = tune_thresholds(y[a], probs[a])
    out["tuned_thresh_heldout"], _, _, _ = f1_scores(y[b], probs[b] >= th_a)

    th_full = tune_thresholds(y, probs)  # 최종 추론용 (Validation 전체로 선택)
    per_class_tbl = [
        dict(category=c, n_pos=int(y[:, i].sum()), AP=None if np.isnan(aps[i]) else round(float(aps[i]), 4),
             precision_0_5=None if np.isnan(prec[i]) else round(float(prec[i]), 4),
             recall_0_5=None if np.isnan(rec[i]) else round(float(rec[i]), 4),
             f1_0_5=None if np.isnan(per_class[i]) else round(float(per_class[i]), 4),
             threshold=round(float(th_full[i]), 3))
        for i, c in enumerate(CATEGORIES)
    ]
    return out, per_class_tbl, th_full


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def run_one(key, args, device):
    set_seed(args.seed)
    cfg = MODELS[key]
    img_size = args.img_size or cfg["native"]
    img_h = args.img_h or img_size  # 입력 높이 (기본: 정사각형). 체크포인트는 img_size(가로) 기준으로 고름
    in_size = img_size if img_h == img_size else (img_size, img_h)  # letterbox용 (W, H)
    print(f"\n===== {key}  ({timm_name_for(key, img_size)}, {img_size}x{img_h}px (WxH), head={args.head}, loss={args.loss}) =====")

    model = build_model(key, img_size, pretrained=not args.smoke, head=args.head, csra_lam=args.csra_lam, csra_T=args.csra_T)
    n_params = count_params(model)
    print(f"parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    if n_params > 50_000_000:
        print("경고: 50M 초과")
    model.to(device)

    dcfg = model.pretrained_cfg
    mean, std = dcfg.get("mean", (0.485, 0.456, 0.406)), dcfg.get("std", (0.229, 0.224, 0.225))
    tf_train, tf_val = Transform(in_size, mean, std, True), Transform(in_size, mean, std, False)

    if args.smoke:
        train_ds, val_ds = SmokeDataset(32, tf_train, 0), SmokeDataset(16, tf_val, 1)
    else:
        tr_ids, tr_y = read_label_csv(args.train_csv, args.id_col)
        va_ids, va_y = read_label_csv(args.val_csv, args.id_col)
        tr_ids, tr_y = keep_existing(tr_ids, tr_y, args.train_img_dir, "train")
        va_ids, va_y = keep_existing(va_ids, va_y, args.val_img_dir, "val")
        if args.limit:
            tr_ids, tr_y, va_ids, va_y = tr_ids[: args.limit], tr_y[: args.limit], va_ids[: args.limit], va_y[: args.limit]
        train_ds = TagDataset(tr_ids, tr_y, args.train_img_dir, tf_train)
        val_ds = TagDataset(va_ids, va_y, args.val_img_dir, tf_val)

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    # 백본은 작은 LR, 새로 붙인 27-class 헤드는 10배 LR
    head_ids = {id(p) for p in head_parameters(model)}
    groups = [
        dict(params=[p for p in model.parameters() if id(p) not in head_ids], lr=args.lr),
        dict(params=[p for p in model.parameters() if id(p) in head_ids], lr=args.lr * 10),
    ]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    accum = max(1, args.accum_steps)  # 유효 batch = batch_size × accum_steps
    steps = max(1, args.epochs * math.ceil(len(train_dl) / accum))  # optimizer step 수
    warm = min(int(0.05 * steps), 500)
    base = [g["lr"] for g in opt.param_groups]
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    crit = AsymmetricLoss() if args.loss == "asl" else nn.BCEWithLogitsLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best, step, t0 = [], None, 0, time.time()
    out_dir = os.path.join(args.out_dir, key)
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            f = (step + 1) / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss / accum).backward()
            loss_sum += loss.item() * len(x)
            n += len(x)
            if (bi + 1) % accum == 0 or bi + 1 == len(train_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

        y_val, p_val = predict(model, val_dl, device, amp)
        m, _, _ = evaluate_probs(y_val, p_val, SPLIT_SEED)
        rec = dict(epoch=ep + 1, train_loss=loss_sum / max(n, 1), mAP=m["mAP"], **{f"{k}@0.5": v for k, v in m["at_0.5"].items()})
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, ensure_ascii=False))
        if best is None or m["mAP"] > best["mAP"]:  # 모델 선택은 Threshold 무관 지표(mAP)로
            best = dict(epoch=ep + 1, mAP=m["mAP"])
            if not args.smoke:
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            best_probs = (y_val, p_val)

    train_time = time.time() - t0
    y_val, p_val = best_probs
    metrics, per_class_tbl, th = evaluate_probs(y_val, p_val, SPLIT_SEED)

    # 추론 속도 (배치 1, 이미지 1장 기준)
    model.eval()
    dummy = torch.randn(1, 3, img_h, img_size, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        for _ in range(3):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1, reps = time.time(), 20
        for _ in range(reps):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.time() - t1) / reps * 1000
    gflops = count_gflops(model, torch.randn(1, 3, img_h, img_size, device=device))

    result = dict(
        model=key, checkpoint=timm_name_for(key, img_size), head=args.head, csra_lam=args.csra_lam, csra_T=args.csra_T, loss=args.loss, img_size=img_size, img_h=img_h, params=n_params,
        params_M=round(n_params / 1e6, 2), seed=args.seed, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, accum_steps=accum, effective_batch=args.batch_size * accum, best_epoch=best["epoch"], mAP=round(metrics["mAP"], 4),
        **{f"{k}@0.5": round(v, 4) for k, v in metrics["at_0.5"].items()},
        **{f"{k}@tuned": round(v, 4) for k, v in metrics["tuned_thresh_heldout"].items()},
        train_time_min=round(train_time / 60, 2),
        peak_vram_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
        latency_ms_bs1=round(latency_ms, 2), gflops=gflops, smoke=args.smoke,
    )
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(result=result, history=history, per_class=per_class_tbl,
                       thresholds=dict(zip(CATEGORIES, map(float, th)))), f, ensure_ascii=False, indent=2)
    if pd is not None:
        pd.DataFrame(per_class_tbl).to_csv(os.path.join(out_dir, "per_class.csv"), index=False)
        if not args.smoke:  # best epoch의 val 예측 확률 (N x 27) — 공통 평가·앙상블용
            pd.DataFrame(p_val, columns=CATEGORIES).assign(file_name=val_ds.ids)[["file_name"] + CATEGORIES] \
                .to_csv(os.path.join(out_dir, "val_probs.csv"), index=False, float_format="%.6f")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="all 또는 쉼표 구분: " + ",".join(MODELS))
    ap.add_argument("--train_csv"); ap.add_argument("--train_img_dir")
    ap.add_argument("--val_csv"); ap.add_argument("--val_img_dir")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--img_size", type=int, default=384, help="입력 가로(정사각형이면 한 변). 0이면 모델별 공식 해상도 사용")
    ap.add_argument("--img_h", type=int, default=0, help="입력 세로. 0이면 img_size와 같음(정사각형). 예: --img_size 512 --img_h 768 (세로 사진용 캔버스)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation (유효 batch = batch_size × accum_steps)")
    ap.add_argument("--lr", type=float, default=1e-4, help="백본 LR (헤드는 10배)")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head", choices=["linear", "query", "csra"], default="linear", help="linear: GAP+Linear, query: ML-Decoder식 쿼리 헤드, csra: CSRA")
    ap.add_argument("--csra_lam", type=float, default=0.1, help="CSRA 잔차 attention 가중치 (논문 기본 0.1)")
    ap.add_argument("--csra_T", type=float, default=1.0, help="CSRA softmax 온도 (단일 헤드 기본 1)")
    ap.add_argument("--loss", choices=["bce", "asl"], default="bce")
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검용: train/val 앞 N장만 사용")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="가짜 데이터로 파이프라인만 점검 (사전학습 가중치 미사용)")
    ap.add_argument("--count_params", action="store_true", help="파라미터 수만 출력하고 종료")
    args = ap.parse_args()

    keys = list(MODELS) if args.models == "all" else [k.strip() for k in args.models.split(",")]
    for k in keys:
        if k not in MODELS:
            raise SystemExit(f"알 수 없는 모델: {k}. 가능: {list(MODELS)}")

    if args.count_params:
        print(f"{'model':14s} {'checkpoint':45s} {'img':>4s} {'params':>14s}")
        for k in keys:
            size = args.img_size or MODELS[k]["native"]
            n = count_params(build_model(k, size, pretrained=False, head=args.head))
            flag = "  (50M 초과!)" if n > 50e6 else ""
            print(f"{k:14s} {timm_name_for(k, size):45s} {size:4d} {n:14,d}{flag}")
        return

    if args.smoke:
        args.epochs, args.batch_size, args.workers = 1, 8, 0
        args.img_size = args.img_size or 224
    elif not all([args.train_csv, args.train_img_dir, args.val_csv, args.val_img_dir]):
        raise SystemExit("--train_csv --train_img_dir --val_csv --val_img_dir 가 필요합니다 (또는 --smoke / --count_params)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for k in keys:
        results.append(run_one(k, args, device))
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    cols = ["model", "head", "loss", "params_M", "img_size", "img_h", "effective_batch", "best_epoch", "mAP", "micro_f1@0.5", "macro_f1@0.5", "image_f1@0.5",
            "micro_f1@tuned", "macro_f1@tuned", "image_f1@tuned", "train_time_min", "peak_vram_GB", "latency_ms_bs1", "gflops"]
    if pd is not None:
        df = pd.DataFrame(results)[cols].sort_values("mAP", ascending=False)
        df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
