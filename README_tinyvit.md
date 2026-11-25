# TinyViT-21M 최종 모델 (E5)

MAT4091 패션 태깅: 패션 사진 1장에서 27개 카테고리의 존재 여부를 예측하는 multi-label 분류 모델입니다.

## 모델

| 항목 | 값 |
| --- | --- |
| 백본 | TinyViT-21M (`tiny_vit_21m_512.dist_in22k_ft_in1k`, timm) |
| 입력 | 가로 512 × 세로 768, 비율 유지 letterbox (패딩색 124,116,104) |
| 헤드 | CSRA (1 head, λ 0.1, T 1) |
| 손실 함수 | BCEWithLogits |
| 학습 | AdamW (wd 0.05), LR 백본 1e-4 / 헤드 1e-3, warmup + cosine, 20 epoch, effective batch 32, seed 42 |
| 파라미터 | 20.71M (제한 50M) |
| 가중치 | `best.pt` — Releases에서 내려받기 (epoch 18) |
| threshold | `result.json`의 `thresholds` (클래스별, val에서만 선택). 팀 비교용은 0.5 공통 |

## 성능 (val 1,158장)

| 지표 | seed 42 | seed 42·123·2026 평균 |
| --- | ---: | ---: |
| mAP | 0.839 | 0.830 |
| Micro F1 @0.5 | 0.873 | 0.874 |
| Macro F1 @0.5 | 0.757 | 0.759 |
| Holdout Micro F1 (공통 threshold) | 0.872 | 0.873 |

## 설치

```bash
pip install torch timm pandas pillow
```

## 추론

`best.pt`와 `result.json`을 한 폴더(예: `weights/`)에 두고 실행합니다.

```bash
python predict.py --img_dir <이미지 폴더> --run weights --name E5 --out_dir pred
```

- 출력: `pred/E5_probs.csv` (이미지 수 × 27 확률), `pred/E5_labels.csv` (최종 라벨), `pred/vis/` (시각화)
- 정답 라벨이 있으면 `--labels_csv <labels.csv>`를 붙이면 TP/FP/FN이 표시됩니다.
- 27개 클래스 순서는 `benchmark.py`의 `CATEGORIES`와 같습니다.

## 학습 재현

```bash
# 팀 공통 조건 E0~E5 (GPU 2장)
bash run_common.sh "0 1" 42 "E5"
python compare_common.py
```

## 파일

| 파일 | 내용 |
| --- | --- |
| `benchmark.py` | 학습·평가 (CSRA 헤드, 세로 캔버스 `--img_h` 포함) |
| `predict.py` | 추론 + 시각화 |
| `run_common.sh`, `compare_common.py` | 팀 공통 조건 실험 실행·결과표 |
| `analyze_val.py` | 클래스별·이미지별·방향별·라벨 수별 평가 |
| `team_eval.py` | threshold 0.5 / holdout / 분할 추론 평가 |
| `make_share_package.py` | Test 예측 공유 파일 생성 |
| `result.json` | E5 설정, 학습 기록, 클래스별 threshold |
