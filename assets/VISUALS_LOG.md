# Visuals Log

자동 생성된 시각화 파일 목록과 업데이트 기록.

`python tools/generate_all_visuals.py` 실행 시 모든 파일이 재생성됩니다.

## Generated Files

| File | Description | Type |
|------|-------------|------|
| `collapse_animation.gif` | WFC 붕괴 과정 step-by-step 애니메이션 | Success |
| `score_distribution.png` | Cost Model 3요소 분해 (roofline/affinity/cache) | Analysis |
| `propagation_sweep.png` | threshold별 전파 제거율 그래프 | Analysis |
| `copier_problem.png` | 복사기 문제: 64KB(실패) vs 12KB(해결) 비교 | Failure→Fix |
| `backtracking.png` | 모순 발생 → 백트래킹 시나리오 4단계 | Failure |
| `sram_comparison.png` | SRAM 64KB/16KB/12KB 결과 비교 | Comparison |

## How to Update

```bash
# 코드 변경 후 시각화 재생성
python tools/generate_all_visuals.py

# 개별 GIF만 재생성
python tools/visualize_collapse.py
```

## History

| Date | Time | Version | Notes |
|------|------|---------|-------|
| 2026-03-21 10:16:02 | 1158ms | v2.7.1 | All 6 visuals regenerated |
| 2026-03-17 21:09:24 | 1342ms | v2.7.1 | All 6 visuals regenerated |
| 2026-03-17 20:28:59 | 1223ms | v2.7.1 | All 6 visuals regenerated |
| 2026-03-14 17:29:20 | 1298ms | v2.1 | All 6 visuals regenerated |
