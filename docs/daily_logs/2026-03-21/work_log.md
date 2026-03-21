# Work Log — 2026-03-21

## 목표
- 연구 프로토타입 마무리: cost model 상관관계를 핵심 feasibility 근거로 정리
- Naive Exhaustive baseline 한계 및 미래 과제 명확화
- 문서 최종 업데이트 및 연구 결론 작성

## 완료 작업

### 1. 문서 최종 업데이트

| 파일 | 변경 내용 |
|------|-----------|
| docs/result.md | 연구 결과 보고서 신규 작성 (EN) — 성공/실패/한계/결론 구조 |
| docs/result.ko.md | 연구 결과 보고서 신규 작성 (KO) |
| README.md | 퍼블릭 발행 톤으로 전면 개편 — 설명투 제거, R&D 보고서 형식 |
| README.ko.md | 동일 톤으로 전면 개편 |
| docs/architecture_guide.md | §8 "cost model correlation unverified" → 부분적 확립 (ρ = +0.52) 반영 |
| docs/architecture_guide.ko.md | §8 동일 내용 한국어 반영 |
| HANDOFF.md | v2.8 → v2.9, 연구 결론 섹션 추가, 완료 작업 테이블로 정리, 미래 과제 구조화 |
| CHANGELOG.md | v2.9 엔트리 추가 |
| assets/ | 시각화 전체 재생성 (working memory hard constraint 반영) |

### 2. 연구 결론 정리

**달성:**
- WFC ↔ Exact DP 100% 일치 (stress_gpu, toy_gpu)
- Cost model ↔ GPU 실행 시간 양의 상관관계 (평균 ρ = +0.52)
  - Softmax: ρ = 1.0 (완벽한 순위 일치)
  - MatMul 대형: ρ = +0.72
  - MatMul 소형: 측정 노이즈로 역전
- Hard constraint (SRAM, alignment, working memory) 물리적 정확성 검증
- 55억 조합 → 수 ms 해결

**미달성:**
- 기존 방법 대비 명확한 우위 없음
- MatMul roofline score 포화 → workload 변별 불가
- 가상 스펙 기반 (12KB SRAM은 현존 GPU에 없음)

### 3. Naive Exhaustive Baseline 한계 정리

README에 이미 명시된 내용:
- ~3500x speedup은 최적화되지 않은 Python 순회 대비 → 공정하지 않음
- Exact DP가 ~7-10ms에 같은 답을 줌 → naive exhaustive 자체가 불필요
- 공정한 baseline 강화 방향: branch-and-bound, beam search, 벡터화 DP, C++/CUDA 구현

## 기술적 결정

- **v2.9로 버전 부여**: cost model 상관관계 검증과 문서 최종 정리를 포함한 연구 마무리 버전
- **HANDOFF를 "연구 결론" 형식으로 전환**: 더 이상 "다음 작업" 목록이 아니라, 연구에서 달성한 것과 못한 것을 솔직하게 기록

## 다음 단계

연구 프로토타입으로서 마무리 완료. 이어갈 경우의 우선순위:
1. Cost model 교정 (GPU 프로파일링 데이터 기반 피팅)
2. Workload 감응형 roofline 개선
3. 최적화된 탐색 baseline (branch-and-bound, beam search)
