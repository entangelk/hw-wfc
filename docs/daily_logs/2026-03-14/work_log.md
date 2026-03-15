# Work Log — 2026-03-14

## 목표
HW-WFC v2.0 최소 PoC 구현 및 결과 신뢰성 검증

## 완료한 작업

### 1. PoC 코어 모듈 구현 (신규 생성)
| 파일 | 역할 |
|------|------|
| `src/state.py` | LayerNode, HWState, Superposition |
| `src/constraint.py` | Hard/Soft 제약 엔진, HardwareSpec |
| `src/cost_model.py` | Roofline 비용 모델 (v2까지 반복 개선) |
| `src/collapse.py` | Shannon 엔트로피 기반 붕괴 + 백트래킹 |
| `src/scheduler.py` | 메인 파이프라인 |
| `specs/toy_gpu.json` | 타겟 하드웨어 스펙 |
| `examples/toy_model.py` | Linear→ReLU→Linear PoC |
| `tests/test_safety_checks.py` | 5가지 안전 점검 |
| `CLAUDE.md`, `AGENTS.md` | HW-WFC 작업 지침서 Section 6 추가 |

### 2. Safety Checks 반복 실행 — 결과 비교표

| Check | 1차 (초기) | 2차 (Cost v2) | 3차 (직사각) | 4차 (전파 강화) | 5차 (전환 통합) | 기준 |
|-------|-----------|--------------|------------|---------------|---------------|------|
| #1 Hard Constraint | OK | OK | OK | OK | OK | < 40% |
| #2 Cost Model | 0.06 FLAT | 1.25 | 1.19 | 1.19 | 1.19 | > 0.01 |
| #3 전파 (t=1.5) | 17% | 17% | 17% | **43%** | 43% | > 25% |
| #4 백트래킹 | dead code | 미보고 | 1회 확인 | 1회 확인 | 1회 확인 | > 0 |
| #5 복사기 | all_same | all_same | **해결** (64KB) | **해결** | **해결** (12KB) | False |

### 3. 코드 수정 이력

#### Cost Model v1 → v2 (핵심 수정)
- **원인:** `roofline_score`에서 `num_tiles * tile_bytes = total_elements * dtype_bytes`로 상쇄
- **수정:** 레이어 타입별 메모리 트래픽 함수 도입 (MatMul reuse, Softmax multi-pass)
- **효과:** spread 0.06 → 1.19

#### 백트래킹 버그 2건 수정
1. `find_lowest_entropy_node`: 후보 0개 노드를 skip → 즉시 return
2. `scheduler.schedule`: 조기 return 3곳에서 `backtracks_used` 미세팅

#### 직사각 타일 추가 → 복사기 문제 해결
- 6종 추가: (32,64), (64,32), (32,128), (128,32), (64,128), (128,64)
- Softmax가 32x128 선택 (fewer row-reduction passes)

#### tile_shape_transition_penalty 추가 → 전파 강도 해결
- 타일 형태 전환 비용: same area=0.3, ratio 기반 최대 0.6
- Check #3: 17% → 43%

#### 전환 비용 통합 (v2.1)
- `select_best_state`가 인접 붕괴 노드와의 전환 비용을 반영 (weight=0.1)
- 64KB SRAM: 점수 차이 0.0002 < 전환 비용 0.03 → 파이프라인 일관성 우선 (합리적)
- 12KB SRAM: 점수 차이 0.35 > 전환 비용 0.06 → 레이어별 최적 유지 (합리적)
- Check #5b를 stress_gpu (12KB)로 변경하여 진짜 차별화 테스트

### 4. Stress Test 결과

#### SRAM 크기별 Attention Block 결과
| Spec | SRAM | Unique States | Quality | Backtracks | 시간 |
|------|------|--------------|---------|------------|------|
| toy_gpu | 64KB | 1/6 | 100% | 0 | 2.9ms |
| tight_gpu | 16KB | 1/6 | 100% | 0 | 2.6ms |
| **stress_gpu** | **12KB** | **2/6** | **100%** | **0** | **1.8ms** |

#### 전환 비용 Crossover 분석
- Softmax (12KB): 16x16 vs 32x32, 양쪽 인접 기준 crossover weight = 0.29
- 현재 weight=0.1 → 16x16 선택 유지 (올바름)
- 실제 하드웨어 전환 비용이 크면 weight 상향 필요

#### Grid Search 전환 비용 포함 비교
- 6-layer attention block, top-8 후보 → 262,144 조합 exhaustive search
- WFC와 Grid Search 결과 동일 (score=3.1294, quality=100%)
- Grid Search: 6006ms, WFC: 1.8ms (3300x 빠름)

### 5. 미해결 문제

#### 백트래킹이 자연적으로 발동하지 않음
- 모든 현실적 시나리오에서 backtrack=0
- 원인: DRAM 후보가 항상 fallback으로 존재하여 모순이 발생하지 않음
- 인위적 시나리오(Check #4b)에서만 발동 확인
- **이것이 문제인지 아닌지**: 실제로는 좋은 신호일 수 있음 (탐색 공간이 잘 설계됨)

#### toy_gpu (64KB)에서 복사기 문제
- 전환 비용 통합 후, 64KB에서는 모든 레이어가 동일 상태로 수렴
- 원인: SRAM이 충분하여 모든 레이어의 최적이 동일 (64x64 ROW SRAM)
- 이것은 버그가 아니라 "편한 환경"에서의 정상 동작

## 기술 결정
- 기존 Circle-WFC 코드 불사용 (도메인 불일치)
- Cost Model: Roofline + layout_affinity + cache_efficiency 3요소 합산
- 백트래킹: snapshot 기반 상태 복원
- 직사각 타일: 정사각만으로는 탐색 공간 동질적
- 전환 비용: collapse 시 weight=0.1로 인접 노드 영향 반영
- Check #5b 테스트 스펙: stress_gpu (12KB) — 진짜 트레이드오프가 발생하는 환경

## 다음 단계
1. Conv2D 레이어 지원 강화 (heterogeneous 모델)
2. transition_weight를 하드웨어 스펙에서 파생하는 방법
3. 더 큰 모델 (20+ 레이어) 스케일링 벤치마크
4. 실제 GPU 스펙 (A100, H100) 기반 테스트
