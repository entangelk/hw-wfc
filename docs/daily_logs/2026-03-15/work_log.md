# Work Log — 2026-03-15

## 목표
Phase 1 Task 1: 전환 비용 가중치(transition_weight) 자동 파생 구현

## 완료한 작업

### 1. `derive_transition_weight()` 구현

| 파일 | 변경 내용 |
|------|----------|
| `src/constraint.py` | `derive_transition_weight(spec, dtype_bytes)` 함수 추가 |
| `src/collapse.py` | `CollapseEngine`에 `transition_weight` 속성 추가, `collapse_node`에서 auto-derive 사용 |
| `src/scheduler.py` | `HWWFCScheduler.__init__`에 `transition_weight` 파라미터 추가 (None=자동) |

**공식**: `weight = clamp(reference_tile_bytes / sram_bytes, 0.01, 0.5)`
- reference_tile = 32×32 float32 = 4096 bytes
- SRAM이 작을수록 전환 비용이 상대적으로 커짐 (물리적 직관)

### 2. 스펙별 자동 파생 결과

| 스펙 | SRAM | 수동 weight | 자동 weight |
|------|------|------------|------------|
| toy_gpu | 64KB | 0.1 | 0.0625 |
| tight_gpu | 16KB | 0.1 | 0.2500 |
| stress_gpu | 12KB | 0.1 | 0.3333 |

### 3. 검증 결과

**기존 시나리오**: toy_gpu / stress_gpu / tight_gpu 모두 기존 수동 0.1과 동일한 최종 선택.
- 이유: 기존 시나리오에서는 base score 차이가 transition penalty보다 훨씬 커서 weight 변화가 결과에 영향을 주지 않음.

**Close-call 시나리오** (8KB SRAM, alignment=32B):
- A가 COL_MAJOR로 붕괴 → B의 ROW vs COL 선택
- B의 ROW_MAJOR base=0.3085, COL_MAJOR base=0.1585 (차이=0.15)
- layout 전환 penalty = 1.0

| weight | 선택 | 이유 |
|--------|------|------|
| 0.01 | ROW_MAJOR | penalty*0.01=0.01 < 0.15 base차이 |
| 0.1 | ROW_MAJOR | penalty*0.1=0.10 < 0.15 base차이 |
| 0.5 (auto) | **COL_MAJOR** | penalty*0.5=0.50 > 0.15 base차이 → 전환 비용 회피 |

→ tight SRAM에서 auto-derive가 올바르게 전환 비용을 중시함

### 4. Safety Check 전후 비교

| Check | v2.1 (수동 0.1) | v2.2 (auto-derive) | 변화 |
|-------|----------------|-------------------|------|
| #1 Hard Constraint | OK | OK | 변화 없음 |
| #2 Cost Model | spread=1.19 | spread=1.19 | 변화 없음 |
| #3 전파 (t=1.5) | 43% | 43% | 변화 없음 |
| #4 백트래킹 | 1회 발동 | 1회 발동 | 변화 없음 |
| #5 복사기 | 해결 | 해결 | 변화 없음 |

→ 기존 동작 보존하면서 범용성 확보

## 기술 결정

### 소형 AI 모델 필요성 검토

**결론: 현 단계(Phase 1)에서는 불필요. Phase 2에서 model_zoo 모듈로 대응.**

| 단계 | 필요한 것 | 실제 모델 필요? | 대안 |
|------|----------|---------------|------|
| Phase 1 (코어 알고리즘) | 알고리즘 검증용 레이어 구성 | X | 수동 LayerNode 정의 (현재 방식) |
| Phase 2 (대규모 스케일링) | GPT-2/BERT의 정확한 dims | △ | `model_zoo.py` — 아키텍처 정의만 (weights 불필요) |
| Phase 3 (코드 생성) | 실제 추론 벤치마크 | O | Tiny model + Triton kernel 실행 |

**이유:**
1. HW-WFC는 모델을 실행하지 않음 — 그래프 구조(layer_type, dims)만 필요
2. PyTorch 의존성 추가는 프로젝트 복잡도를 불필요하게 높임
3. GPT-2/BERT의 레이어 구성은 공개 정보이므로 `model_zoo.py`로 하드코딩 가능
4. Phase 3에서 실제 벤치마크가 필요할 때 tiny model (e.g., 2-layer Transformer) 도입

---

### 5. 2D 그래프 전파 (DAG 지원) — Phase 1 Task 2

| 파일 | 변경 내용 |
|------|----------|
| `src/state.py` | `LayerNode`에 `neighbors: list[str]` 필드 추가 |
| `src/collapse.py` | `select_best_state`의 인접 노드 탐색을 neighbors 기반으로 교체 |
| `src/scheduler.py` | `ensure_neighbors()` 헬퍼 추가, `_propagate_from`을 BFS 기반으로 완전 교체 |
| `examples/resnet_block.py` | ResNet Residual Block DAG 예제 (skip connection 포함) |

**설계 결정:**
- neighbors 미설정 시 리스트 순서로 1D 체인 자동 추론 → 하위 호환 100%
- BFS 전파 + visited set으로 cycle 방지
- 붕괴된 노드를 통과하여 그 너머의 미붕괴 노드에도 전파 계속
- 자동 붕괴(후보 1개) 시 해당 노드의 이웃도 큐에 추가하여 연쇄 전파

**검증 결과:**

| 항목 | 결과 |
|------|------|
| 기존 Safety Check 5건 | 전부 통과 (이전과 동일) |
| 기존 attention_block.py | 동일 결과, quality 100% |
| ResNet DAG 예제 | Conv1 → skip → Add 직접 전파 확인 |
| DAG vs Linear 비교 | 동일 최종 선택, DAG에서 전파 경로가 더 짧음 |

**관찰:**
- stress_gpu(12KB)에서 ResNet block의 모든 레이어가 32x32 BLOCK_TILED 선택 → Conv2D의 layout_affinity가 BLOCK_TILED 선호하므로 정상
- DAG 전파의 진짜 차이는 이질적 연산이 branch에 있을 때 나타남 (예: branch에 Softmax가 있는 Inception 구조)
- 현재 시나리오에서는 같은 결과이지만, 전파 순서/경로가 다름: DAG는 Conv1에서 Add로 즉시 전파, Linear는 순차 7홉

---

### 6. 적응적 penalty threshold — Phase 1 Task 3

| 파일 | 변경 내용 |
|------|----------|
| `src/scheduler.py` | `penalty_threshold="auto"` 지원, `_adapt_threshold()` 메서드 추가 |
| `src/scheduler.py` | `ScheduleResult.final_threshold` 필드 추가 |
| `src/scheduler.py` | `_propagate_from`이 `(removed, target)` 튜플 반환하도록 변경 |

**설계:**
- 첫 전파 후 제거율을 측정하여 이후 전파의 threshold를 1회 보정
- 목표 제거율 범위: 20~60%
- 제거율 < 20% → factor = 0.6 + 2.0×rate (rate=0% → ×0.6, rate=10% → ×0.8)
- 제거율 > 60% → factor = 1.0 + 2.0×(rate-0.6) (rate=80% → ×1.4, rate=100% → ×1.8)
- 고정값도 여전히 지원 (`penalty_threshold=1.5`)

**검증 결과:**

| 시나리오 | 제거율 | 보정 | 최종 threshold |
|----------|--------|------|----------------|
| toy_gpu attention (기본) | 35% | 유지 | 1.50 |
| stress_gpu attention | 48% | 유지 | 1.50 |
| ROW_MAJOR only 후보 (전파 불가) | 0% | 강화 | 0.90 |
| 기존 Safety Check 5건 | — | — | 전부 통과 |

**관찰:**
- 기본 threshold=1.5는 이미 대부분의 시나리오에서 20~60% 범위 안
- 적응적 보정이 진짜 가치를 발휘하는 경우: 동질적 후보군(같은 layout만)처럼 전파가 무효인 상황
- 이후 Phase 2에서 대규모 모델(72+ layers)에 적용 시 threshold 자동 선택의 편의성 확인 예정

---

### 7. Phase 2: 실제 GPU 스펙 + 대규모 모델 스케일링 — Task 4 & 5

| 파일 | 변경 내용 |
|------|----------|
| `specs/a100.json` | NVIDIA A100: SRAM 192KB, HBM2e 2039 GB/s, 312 TFLOPS |
| `specs/h100.json` | NVIDIA H100: SRAM 256KB, HBM3 3352 GB/s, 990 TFLOPS, alignment 256B |
| `src/model_zoo.py` | GPT-2 Small (72L), BERT-Base (84L) 레이어 정의 + `generate_large_candidates()` |
| `examples/scaling_benchmark.py` | A100/H100 × GPT-2/BERT 스케일링 벤치마크 |

**벤치마크 결과:**

| Model | GPU | Layers | States | Time | Unique | Score |
|-------|-----|--------|--------|------|--------|-------|
| GPT-2 Small | A100 | 72 | 5,616 | 530ms | 2/72 | 67.85 |
| BERT-Base | A100 | 84 | 6,552 | 700ms | 2/84 | 83.12 |
| GPT-2 Small | H100 | 72 | 5,616 | 571ms | 1/72 | 61.93 |
| BERT-Base | H100 | 84 | 6,552 | 632ms | 1/84 | 77.64 |

Grid Search 비교: 78^72 = 1.70×10^136 조합 vs WFC 530ms

**핵심 발견:**
1. **A100에서 Softmax 차별화 확인**: 128x128 대신 64x128 선택 — SRAM 192KB에서 working memory 4x(Softmax) = 128×128×4bytes×4 = 262KB > 192KB → 강제 축소
2. **H100은 전부 128x128**: SRAM 256KB가 충분히 넉넉 → unique=1은 물리적으로 정상 (복사기 문제 아님)
3. **성능**: 500~700ms는 "몇 ms"보다 느림 — 원인은 엔트로피 계산(score_all_candidates)이 collapse loop에서 72~84회 호출. 후보 수(78개) × 레이어 수가 O(n²) 수준
4. **H100 alignment=256B**: 8x8, 16x16, 32x32(tile_n=32, row_bytes=128 < 256)도 제거 → hard constraint가 더 공격적

**관찰 — 성능 최적화 기회:**
- 엔트로피 계산 캐싱: 변경되지 않은 노드의 엔트로피를 재계산하지 않음
- score_all_candidates를 한 번만 계산하고 결과 재사용
- 현재 단계에서는 정확성이 우선이므로 최적화는 Phase 3 이후

---

### 8. Conv2D im2col + Depthwise Conv 지원 — Phase 2 Task 6

| 파일 | 변경 내용 |
|------|----------|
| `src/state.py` | `LayerType.DEPTHWISE_CONV` 추가, `flops()` 대응 |
| `src/cost_model.py` | `_conv2d_memory_traffic()` im2col 기반 구현, `_depthwise_conv_memory_traffic()` 신규 |
| `src/cost_model.py` | `layout_affinity` DEPTHWISE_CONV (BLOCK_TILED 0.20 강선호), `_working_memory_multiplier` 2.5x |
| `examples/mobilenet_block.py` | MobileNetV2 Inverted Residual Block (skip connection DAG) |

**Conv2D im2col 모델:**
- 입력을 [H×W, C_in×KH×KW] 행렬로 변환 후 커널 [C_out, C_in×KH×KW]과 MatMul
- 타일링이 im2col 행렬에 적용되어 Linear과는 다른 traffic 패턴

**Depthwise Conv 모델:**
- 채널별 독립 연산 → 데이터 재사용 제한적
- halo (커널 오버랩) 포함 입력 traffic
- BLOCK_TILED 0.20 강선호 (공간 지역성)

**벤치마크 결과:**

| Layer | stress_gpu(12KB) | A100(192KB) |
|-------|------------------|-------------|
| PW_Expand (Conv2D 1x1) | 32x32 BLOCK_TILED (0.60) | 128x128 BLOCK_TILED (0.90) |
| DW_Conv (Depthwise 3x3) | 32x32 BLOCK_TILED (0.52) | 128x128 BLOCK_TILED (0.53) |
| PW_Project (Conv2D 1x1) | 32x32 BLOCK_TILED (0.72) | 128x128 BLOCK_TILED (0.97) |

- DW_Conv score가 PW보다 낮음 → FLOPs 대비 traffic이 높아서 roofline score 낮음 (물리적으로 정상)
- 같은 BLOCK_TILED이지만 이는 Conv 계열 모두 공간 지역성이 중요하기 때문

---

## 최종 재점검 결과

| 항목 | 결과 |
|------|------|
| Safety Check #1~#5 | 전부 통과 (v2.2 대비 변화 없음) |
| toy_model.py | OK, quality 100% |
| attention_block.py | OK, quality 100% |
| resnet_block.py | OK, DAG vs Linear 동일 결과 |
| mobilenet_block.py | OK, stress_gpu/A100 모두 정상 |
| scaling_benchmark.py | OK, GPT-2 517ms / BERT 638ms (A100) |

---

## 시각화 아이디어 메모 (Phase 3 Task 8용)

### 1. Collapse 과정 애니메이션
- 각 step에서 어떤 노드가 붕괴되었는지 시각적으로 표시
- 노드 색상: 미붕괴(회색) → 붕괴중(주황) → 확정(초록)
- DAG 그래프 구조에서 전파 화살표 애니메이션
- **구현**: matplotlib + networkx 또는 Graphviz로 프레임 생성 → GIF

### 2. 엔트로피 히트맵
- X축: 레이어, Y축: collapse step
- 색상: 엔트로피 값 (높음=빨강, 낮음=파랑, 붕괴=검정)
- 전파로 인해 엔트로피가 급락하는 순간이 시각적으로 보여야 함
- **데이터**: collapse loop에서 매 step 엔트로피 스냅샷 저장 필요

### 3. 점수 분해 바 차트 (Score Breakdown)
- 각 레이어의 최종 선택에 대해: roofline / affinity / cache / transition 기여도
- 스택 바 차트로 어떤 요소가 결정적이었는지 한눈에 파악
- **데이터**: `debug_score_breakdown.py` 확장

### 4. GPU 스펙별 비교 대시보드
- 슬라이더: SRAM 크기 (8KB~256KB), threshold (0.5~3.0)
- 실시간으로 스케줄 결과 변화 표시
- Working memory 초과 영역 시각화 (SRAM 용량 대비 타일 크기 scatter)
- **구현**: Streamlit 또는 Jupyter widget

### 5. 탐색 공간 축소 퍼널
- 초기 후보 수 → Hard Constraint → 전파 → 붕괴 각 단계별 남은 후보 수
- 깔때기(funnel) 차트로 각 단계의 기여도 표시
- 대규모 모델(72~84 layers)에서 특히 임팩트 있음

### 6. DAG 그래프 + 선택 결과 오버레이
- ResNet/MobileNet의 DAG 구조를 그래프로 시각화
- 각 노드에 선택된 tile/layout을 label로 표시
- 전환 비용이 높은 엣지를 빨간색으로 강조
- **구현**: Graphviz dot 또는 mermaid

---

## Phase 3 완료 작업

### Task 7: Triton 커널 코드 생성기 (`src/codegen.py`)

| 파일 | 변경 내용 |
|------|----------|
| `src/codegen.py` (신규) | KernelConfig, 커널 스켈레톤 생성, 런치 설정 코드 생성 |

**주요 구현**:
- `KernelConfig` dataclass: tile_m/n/k, num_warps, num_stages 자동 추정
- `generate_matmul_kernel()`: Linear/Conv2D → Triton MatMul 스켈레톤
- `generate_softmax_kernel()`: Row-parallel 3-pass softmax
- `generate_elementwise_kernel()`: ReLU/LayerNorm element-wise 커널
- `generate_launch_config()`: grid 계산 + 런치 코드 생성
- `generate_full_pipeline()`: 전체 파이프라인 코드 한번에 생성

**검증**: stress_gpu × 4-layer attention 시나리오에서 정상 코드 생성 확인

### Task 8: 시각화 모듈 (`src/visualize.py`)

| 파일 | 변경 내용 |
|------|----------|
| `src/visualize.py` (신규) | 4종 시각화 함수 (matplotlib 기반) |

**주요 구현**:
- `plot_funnel()`: 탐색 공간 축소 퍼널 차트 (Init → Hard → Propagation → Final)
- `plot_score_breakdown()`: 레이어별 점수 분해 스택 바 차트 (Roofline/Affinity/Cache)
- `plot_dag()`: DAG 그래프 + 선택 결과 오버레이 (타입별 색상, 타일/레이아웃 label)
- `plot_spec_comparison()`: GPU 스펙별 비교 차트 (Unique States + Total Score)

**검증**: stress_gpu × 4-layer attention 시나리오에서 PNG 4건 정상 출력

### Safety Check 재점검 결과 (Phase 3 완료 후)

| Check | 상태 | 비고 |
|-------|------|------|
| #1 Hard Constraint | OK | 8x8, 16x16 alignment 제거 (12/66) |
| #2 Cost Model 변별력 | OK | spread=1.19, 54개 완전 구분 |
| #3 전파 강도 | OK | t=1.5에서 43% 제거 |
| #4 백트래킹 | OK | 4b 시나리오에서 1회 발동 |
| #5 복사기 문제 | OK | Softmax=16x16 ≠ MatMul=32x32 |

Phase 3 작업은 코어 알고리즘에 영향을 주지 않으므로 Safety Check 결과 변동 없음.

### 전체 예제 재검증

| 예제 | 상태 | 비고 |
|------|------|------|
| toy_model.py | OK | 3-layer, grid search 비교 정상 |
| attention_block.py | OK | 6-layer, unique=1 (toy_gpu SRAM 충분) |
| resnet_block.py | OK | DAG skip connection 정상 |
| mobilenet_block.py | OK | Depthwise Conv 정상 |
| scaling_benchmark.py | OK | GPT-2/BERT × A100/H100, 500~700ms |

## Next Steps
- Phase 1~3 전부 완료
- 향후 가능한 작업: 성능 최적화 (엔트로피 캐싱), 인터랙티브 대시보드 (Streamlit), 실제 Triton 커널 실행 검증
