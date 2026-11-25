"""팀 공통 조건 실험 결과표 (common_[백본_]<실험>_s<seed> 폴더). 백본별로 E0 대비 같은 seed끼리 차이도 출력."""
import glob, json, os, re
from collections import defaultdict
import numpy as np

M = ["mAP", "micro_f1@tuned", "macro_f1@tuned", "image_f1@tuned"]
DESC = {"E0": "Baseline 384 BCE Linear", "E1": "512x512", "E2": "512x768", "E3": "CSRA", "E4": "ASL", "E5": "512x768 + CSRA", "E5a": "E2 + CSRA", "E5b": "E2 + ASL", "E5c": "E2 + CSRA + ASL"}
def report(R):
    print(f"{'exp':4s} {'설정':24s}" + "".join(f"{k:>17s}" for k in M) + f"{'params_M':>9s}{'GFLOPs':>8s}{'ms':>6s}  seeds/best_ep")
    for e in sorted(R):
        rs = [R[e][s] for s in sorted(R[e])]
        cells = ""
        for k in M:
            v = np.array([r[k] for r in rs])
            cells += f"{v.mean():9.4f} ±{(v.std(ddof=1) if len(v) > 1 else 0):6.4f}"
        r0 = rs[0]
        print(f"{e:4s} {DESC.get(e, ''):24s}{cells}{r0['params_M']:9.2f}{str(r0.get('gflops')):>8s}{r0['latency_ms_bs1']:6.1f}  "
              + ",".join(f"{s}/{R[e][s]['best_epoch']}" for s in sorted(R[e])))

    if "E0" in R:
        print("\n== E0 대비 같은 seed끼리 차이 ==")
        for e in sorted(R):
            if e == "E0":
                continue
            for k in M:
                d = [(s, R[e][s][k] - R["E0"][s][k]) for s in sorted(R[e]) if s in R["E0"]]
                if d:
                    tag = "  (모두 +)" if len(d) > 1 and min(x for _, x in d) > 0 else "  (모두 -)" if len(d) > 1 and max(x for _, x in d) < 0 else ""
                    print(f"{e} {k:16s} " + "  ".join(f"s{s}:{x:+.4f}" for s, x in d) + tag)
    if "E1" in R and "E2" in R:
        print("\n== E2 vs E1 (종횡비 효과만 분리: 가로 512 동일, 세로 캔버스만 다름) ==")
        for k in M:
            d = [(s, R["E2"][s][k] - R["E1"][s][k]) for s in sorted(R["E2"]) if s in R["E1"]]
            if d:
                print(f"   {k:16s} " + "  ".join(f"s{s}:{x:+.4f}" for s, x in d))
    for e5 in [e for e in sorted(R) if e.startswith("E5") and "E2" in R]:
        print(f"\n== {e5} vs E2 (512x768 위에 더한 효과) ==")
        for k in M:
            d = [(s, R[e5][s][k] - R["E2"][s][k]) for s in sorted(R[e5]) if s in R["E2"]]
            if d:
                print(f"   {k:16s} " + "  ".join(f"s{s}:{x:+.4f}" for s, x in d))


ALL = defaultdict(lambda: defaultdict(dict))  # 백본 -> 실험 -> seed -> result
for f in glob.glob("common_*_s*/*/result.json"):
    m = re.match(r"common_(?:.+_)?(E\d[a-z]?)_s(\d+)$", f.split(os.sep)[0])
    if m:
        r = json.load(open(f, encoding="utf-8"))["result"]
        ALL[r["model"]][m.group(1)][int(m.group(2))] = r

for model, R in sorted(ALL.items()):
    print(f"\n######## {model} ########")
    report(R)
