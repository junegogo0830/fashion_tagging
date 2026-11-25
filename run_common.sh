#!/usr/bin/env bash
# 팀 공통 조건 실험. GPU 2장만 사용: 실험을 두 줄(queue)로 나눠 GPU마다 하나씩 차례로 돌린다.
#   공통: letterbox, AdamW, warmup+cosine, LR 1e-4/1e-3, WD 0.05, 최대 20 epoch(best val mAP 시점 저장),
#         effective batch 32 (batch 32 그대로; 메모리 부족 시 BS=16 ACC=2 로 바꿔 실행하고 기록)
#   E0: 384x384, BCE, GAP+Linear   E1: 512x512   E2: 512x768(WxH)   E3: CSRA 헤드   E4: ASL
#   E5: 최종 = E2 + CSRA (512x768, BCE, CSRA)   (참고용 E5a: E2+CSRA   E5b: E2+ASL   E5c: E2+CSRA+ASL)
#
# 사용법: bash run_common.sh "<GPU 2개>" <seed> ["<실험 목록>"]
#   bash run_common.sh "0 1" 42                 # E0~E4 전부, GPU 0·1
#   bash run_common.sh "2 3" 123 "E0 E2"        # 특정 실험만
#   bash run_common.sh "3 3" 123 "E0 E2 E5"     # GPU 한 장(3번)에서 하나씩 차례로
#   MODEL=convnextv2_t bash run_common.sh "0 1" 42   # 다른 백본 (tinyvit_21m / convnextv2_t / effnetv2_s)
set -e
cd ~/backbone_bench
PY=$HOME/venv_fashion/bin/python
TRAIN=~/data/train_zip/train
VAL=~/data/val_zip/MAT4091_fashion_val/val
read -ra GPUS <<< "${1:-0 1}"
[ ${#GPUS[@]} -eq 2 ] || { echo "GPU는 정확히 2개를 지정하세요. 예: bash run_common.sh \"0 1\" 42"; exit 1; }
SEED=${2:-42}
read -ra EXPS <<< "${3:-E0 E1 E2 E3 E4}"
MODEL=${MODEL:-tinyvit_21m}
BS=${BS:-32}; ACC=${ACC:-1}
W=$(( $(nproc) / 4 )); [ $W -gt 8 ] && W=8; [ $W -lt 2 ] && W=2   # 다른 팀원도 CPU를 쓰므로 GPU당 최대 8
DATA="--train_csv $TRAIN/labels.csv --train_img_dir $TRAIN/images --val_csv $VAL/labels.csv --val_img_dir $VAL/images"
BASE="--models $MODEL --workers $W --batch_size $BS --accum_steps $ACC --epochs 20 --seed $SEED"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

args_for() {
  case $1 in
    E0) echo "--img_size 384 --loss bce --head linear" ;;
    E1) echo "--img_size 512 --loss bce --head linear" ;;
    E2) echo "--img_size 512 --img_h 768 --loss bce --head linear" ;;
    E3) echo "--img_size 384 --loss bce --head csra" ;;
    E4) echo "--img_size 384 --loss asl --head linear" ;;
    E5) echo "--img_size 512 --img_h 768 --loss bce --head csra" ;;   # 최종: E2(512x768) + E3(CSRA)
    E5a) echo "--img_size 512 --img_h 768 --loss bce --head csra" ;;
    E5b) echo "--img_size 512 --img_h 768 --loss asl --head linear" ;;
    E5c) echo "--img_size 512 --img_h 768 --loss asl --head csra" ;;
    *) echo "알 수 없는 실험: $1" >&2; exit 1 ;;
  esac
}
# 실험별 상대 소요 시간 (TinyViT A5000 실측 기준, 시간). 두 GPU의 총 시간이 비슷해지도록 나눈다.
cost_for() { case $1 in E2|E5*) echo 51 ;; E1) echo 33 ;; *) echo 18 ;; esac; }

# 오래 걸리는 것부터, 지금 더 한가한 GPU에 배정
SORTED=$(for e in "${EXPS[@]}"; do echo "$(cost_for $e) $e"; done | sort -rn | awk '{print $2}')
QA=(); QB=(); TA=0; TB=0
for e in $SORTED; do
  if [ "${GPUS[0]}" = "${GPUS[1]}" ]; then QA+=("$e"); continue; fi   # 같은 GPU 두 번 = 한 장에서 전부 차례로
  if [ $TA -le $TB ]; then QA+=("$e"); TA=$((TA + $(cost_for $e))); else QB+=("$e"); TB=$((TB + $(cost_for $e))); fi
done

if [[ " ${EXPS[*]} " == *" E3 "* ]] && [ ! -f logs/check_csra_${MODEL}.ok ]; then
  echo "== check: CSRA head, 200 images, 1 epoch (GPU ${GPUS[0]}) =="
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY benchmark.py $DATA --models $MODEL --workers $W --batch_size 8 --head csra --limit 200 --epochs 1 --out_dir ft_check_csra_${MODEL} > logs/check_csra_${MODEL}.log 2>&1 \
    || { echo "CHECK FAILED"; tail -30 logs/check_csra_${MODEL}.log; exit 1; }
  grep -E "parameters" logs/check_csra_${MODEL}.log | head -1; touch logs/check_csra_${MODEL}.ok
fi

run_queue() {  # $1=GPU, 나머지=실험들. 하나가 실패해도 다음 실험은 계속
  local g=$1; shift
  for e in "$@"; do
    local out=common_${e}_s${SEED}
    [ "$MODEL" != "tinyvit_21m" ] && out=common_${MODEL}_${e}_s${SEED}
    echo "[$(date '+%m-%d %H:%M')] start $e (GPU $g)" >> logs/queue_gpu${g}.log
    if CUDA_VISIBLE_DEVICES=$g $PY benchmark.py $DATA $BASE $(args_for $e) --out_dir $out > logs/${out}.log 2>&1; then
      echo "[$(date '+%m-%d %H:%M')] done  $e" >> logs/queue_gpu${g}.log
    else
      echo "[$(date '+%m-%d %H:%M')] FAIL  $e  (logs/${out}.log)" >> logs/queue_gpu${g}.log
    fi
  done
}

nohup bash -c "$(declare -f run_queue args_for); PY='$PY' DATA='$DATA' BASE='$BASE' SEED='$SEED' MODEL='$MODEL'; run_queue ${GPUS[0]} ${QA[*]}" > /dev/null 2>&1 &
nohup bash -c "$(declare -f run_queue args_for); PY='$PY' DATA='$DATA' BASE='$BASE' SEED='$SEED' MODEL='$MODEL'; run_queue ${GPUS[1]} ${QB[*]}" > /dev/null 2>&1 &
echo "model $MODEL, seed $SEED, batch ${BS}x${ACC}"
echo "GPU ${GPUS[0]}: ${QA[*]}   (차례로 실행)"
echo "GPU ${GPUS[1]}: ${QB[*]}   (차례로 실행)"
echo "진행: cat logs/queue_gpu*.log ; tail -n 1 logs/common_*.log    결과: \$HOME/venv_fashion/bin/python compare_common.py"
