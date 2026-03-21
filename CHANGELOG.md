# CHANGELOG

| 날짜 | 변경 | 상세 |
|------|------|------|
| 2026-03-21 | v2.9: 연구 마무리 — cost model ↔ GPU 실행 시간 상관관계 검증 (평균 Spearman ρ = +0.52, Softmax ρ = 1.0), 문서 전면 재감사 완료, naive exhaustive baseline 한계 및 미래 과제 정리, HANDOFF 연구 결론 작성 | [work_log](docs/daily_logs/2026-03-21/work_log.md) |
| 2026-03-20 | v2.8: 전면 감사 + working memory hard constraint 도입 — `check_working_memory()` 추가로 WFC vs DP 품질 98.3%→100% 달성, Softmax 32x32 SRAM 물리적 제거, README/Architecture Guide 전면 업데이트, Safety Check #5b DP 검증 추가 | [work_log](docs/daily_logs/2026-03-20/work_log.md) |
| 2026-03-17 | v2.7.1: README benchmark 재감사, full exact DP 비교 추가, heuristic/truncated baseline 문구 정정, benchmark fairness/measurement stability 규칙 추가, `.venv` GPU 재검증 및 isolated autotuner rerun 감사 반영 | [work_log](docs/daily_logs/2026-03-17/work_log.md) |
| 2026-03-16 | v2.7: Phase 4 — RTX 3060 Triton 커널 실행 검증, autotuner 대비 90% 성능 달성 | [work_log](docs/daily_logs/2026-03-16/work_log.md) |
| 2026-03-15 | v2.6: Phase 1~3 완료 — DAG 전파, adaptive threshold, Conv2D/Depthwise, A100/H100 스케일링, codegen, visualize | [work_log](docs/daily_logs/2026-03-15/work_log.md) |
| 2026-03-14 | v2.1: 전환 비용 통합, stress_gpu 스펙, Safety Checks 5/5 통과 | [work_log](docs/daily_logs/2026-03-14/work_log.md) |
| 2026-03-14 | v2.0: PoC 구현, Cost Model v2, 백트래킹 버그 수정, 직사각 타일 | [work_log](docs/daily_logs/2026-03-14/work_log.md) |
