<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-6B7280?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-111111?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# HW-WFC

**Hardware Wave Function Collapse** — 제약 기반 붕괴 탐색을 AI 하드웨어 컴파일러 자동 스케줄링에 적용

GPU 하드웨어에서 AI 모델 추론 시 각 레이어의 타일 크기, 메모리 레이아웃, 연산 위치를 결정하는 Wave Function Collapse (WFC) 알고리즘 연구 프로토타입.

**상태**: 연구 완료 (v2.9). 알고리즘은 정확하며 cost model이 실제 GPU 성능과 방향성 있는 상관관계를 보이나, 기존 방법 대비 우위는 입증되지 않음. 전체 분석은 [연구 결과](docs/result.ko.md) 참고.

**문서**: [연구 결과](docs/result.ko.md) | [아키텍처 가이드](docs/architecture_guide.ko.md) | [컨셉](docs/concept.ko.md)

---

## 알고리즘

![HW-WFC Collapse Animation](assets/collapse_animation.gif)

1. **Superposition** — 각 레이어가 모든 가능한 HW 상태(tile size × layout × location)를 보유
2. **Hard Constraints** — 물리적으로 불가능한 후보 제거 (SRAM 용량, working memory 초과, alignment 위반)
3. **Bottleneck-First Collapse** — 최대 FLOPs 레이어 우선 붕괴
4. **Constraint Propagation** — 전환 비용 기반 후보 제거가 인접 노드로 전파
5. **Entropy-Ordered Collapse** — 나머지 레이어를 Shannon 엔트로피 순으로 붕괴
6. **Backtracking** — 모순 시 snapshot 기반 상태 복원

---

## 결과 요약

6-layer Transformer Attention Block, `stress_gpu` (12KB SRAM) 기준.

| 지표 | 값 |
|------|-----|
| WFC vs Exact DP (Viterbi) | **100% 일치** — 동일한 상태 선택 |
| WFC 시간 | ~2-3ms |
| Exact DP 시간 | ~7-10ms |
| 탐색 공간 | ~20억 조합 |
| Cost model ↔ GPU 상관관계 | **평균 Spearman ρ = +0.52** (Softmax: 1.0, MatMul 대형: +0.72) |

### 성공한 것

- WFC가 제약된 탐색 공간을 올바르게 탐색하며, 이 벤치마크에서 exact DP 최적해와 일치한다.
- Cost model이 실제 Triton 커널 실행 시간과 양의 순위 상관관계를 보인다 — Softmax는 완벽 일치 (ρ = 1.0).
- Hard constraint(SRAM, alignment, working memory)가 가장 의미 있는 스케줄링 차별화를 이끈다.

### 실패한 것

- **기존 방법 대비 우위 없음.** WFC는 exact DP, Triton autotuner, 또는 잘 최적화된 baseline을 이기지 못한다.
- **MatMul에서 cost model 포화.** Roofline score가 전부 1.0 → 부차적 `cache_efficiency` 컴포넌트만으로 변별.
- **가상 스펙 의존성.** 핵심 시나리오(12KB SRAM)는 실제 GPU에 존재하지 않음. A100(192KB)에서는 문제가 trivial.
- **Naive exhaustive speedup (~3500x)은 허수.** Baseline이 최적화되지 않은 Python. Exact DP는 같은 문제를 ~7-10ms에 해결.

전체 분석: **[연구 결과](docs/result.ko.md)**

---

## WFC vs Exact DP (12KB SRAM)

```
레이어      WFC heuristic       Exact DP
QKV_Proj    32x32 ROW SRAM      32x32 ROW SRAM
QK_MatMul   32x32 ROW SRAM      32x32 ROW SRAM
Softmax     16x16 ROW SRAM      16x16 ROW SRAM   ← working memory 제약
AV_MatMul   32x32 ROW SRAM      32x32 ROW SRAM
Out_Proj    32x32 ROW SRAM      32x32 ROW SRAM
LayerNorm   32x32 ROW SRAM      32x32 ROW SRAM
```

Softmax working memory: 4x × 4KB = 16KB > 12KB SRAM → `32x32` hard constraint에 의해 제거. 두 알고리즘 모두 `16x16` 선택.

---

## Cost Model ↔ GPU 상관관계

Cost model 점수와 실제 Triton 실행 시간의 Spearman 순위 상관계수 (RTX 3060, CUDA 12.8).

| 레이어 | 워크로드 | ρ | 유의성 |
|--------|----------|---|--------|
| Softmax | Medium–Large | **+1.00** | *** |
| MatMul | Large (2048²) | **+0.72** | * |
| MatMul | GPT-2 FFN | +0.66 | ns |
| MatMul | Small (256x512) | -0.69 | ns |

평균 ρ = +0.52. 방향성은 맞지만 정밀하지 않음. Roofline 컴포넌트가 MatMul에서 포화되어 `cache_efficiency`만이 유일한 변별 요소.

재현: `python examples/cost_model_correlation.py`

---

## 시각화

`python tools/generate_all_visuals.py`로 재생성.

| | |
|---|---|
| ![Score Distribution](assets/score_distribution.png) | ![Propagation Sweep](assets/propagation_sweep.png) |
| Cost model 점수 분해 (Roofline + Affinity + Cache) | Threshold별 후보 제거율 (t=1.5 → 43%) |
| ![Copier Problem](assets/copier_problem.png) | ![SRAM Comparison](assets/sram_comparison.png) |
| 64KB: 전부 동일. 12KB: working memory가 차별화 유발 | SRAM 크기별 스케줄링 결과 |

---

## 아키텍처

```
LayerNode ──▶ HardConstraint ──▶ CostModel ──▶ CollapseEngine ──▶ Propagation ──▶ Scheduler
(candidates)  (SRAM, align,     (roofline +    (entropy,         (layout +       (bottleneck
 HWState[]     working memory)   affinity +      backtrack)        tile_shape +     first,
                                 cache)                            location)        bidirect)
```

### Cost Model

3요소 합산:
- **Roofline Score** — 타일링에 의한 데이터 재사용률 (MatMul 대형 SRAM에서 1.0으로 포화)
- **Layout Affinity** — 레이어 타입별 레이아웃 선호도
- **Cache Efficiency** — SRAM 활용률 가우시안 함수 (working memory multiplier 적용)

### 하드웨어 스펙

| 스펙 | SRAM | 역할 |
|------|------|------|
| `stress_gpu.json` | 12KB | 주요 벤치마크 — 차별화 유발 |
| `toy_gpu.json` | 64KB | 기준선 — 복사기 문제 노출 |
| `a100.json` / `h100.json` | 192–256KB | 스케일링 검증 |
| `rtx3060.json` | 100KB | GPU 실행 검증 |

---

## Safety Checks (5/5 통과)

| # | 검증 | 기준 | 결과 |
|---|------|------|------|
| 1 | Hard Constraint | false positive/negative 없음 | 통과 |
| 2 | Cost Model 변별력 | 점수 spread > 0.01 | 1.19 |
| 3 | 전파 | t=1.5에서 제거율 > 25% | 43% |
| 4 | 백트래킹 | 모순 시 발동 | 검증됨 |
| 5 | 복사기 문제 없음 | 이질적 레이어 차별화 | WFC = DP |

실행: `python tests/test_safety_checks.py`

---

## 한계

- **기존 방법 대비 입증된 우위 없음.** Exact DP와 autotuner를 이기지 못함.
- **Cost model 정밀도**: 평균 ρ = +0.52. 방향성은 맞지만 프로덕션 스케줄링에는 부족.
- **가상 스펙 의존성**: 핵심 결과는 인위적으로 제약된 SRAM(12KB)을 요구. 실제 GPU(192–256KB)에서는 문제가 trivial.
- **Roofline 포화**: MatMul 점수가 수렴하여 workload별 타일 추천이 제한됨.
- **단일 커널 범위**: Multi-kernel fusion, operator 스케줄링, cross-layer 메모리 계획 미고려.

## 향후 방향

주요 병목은 cost model 정밀도. 순위 상관관계를 GPU 프로파일링 데이터 교정으로 ρ = +0.52에서 ρ ≥ +0.8로 개선할 수 있다면, 하드웨어 실행 없이 경쟁력 있는 타일 설정을 예측할 수 있다 — 크로스 컴파일, design-space 탐색, autotuner 초기 스케줄 생성에 활용 가능.

이 교정에는 다양한 워크로드에 걸친 실제 GPU 프로파일링 데이터가 필요하며, 이는 소프트웨어 전용 실험의 범위를 넘어선다.

---

## 빠른 시작

```bash
python examples/attention_block.py                # Attention Block 스케줄링
python examples/attention_exhaustive_benchmark.py  # 핵심 결과 재현
python tests/test_safety_checks.py                 # Safety Checks
python examples/cost_model_correlation.py          # Cost model ↔ GPU 상관관계
python examples/triton_verify.py                   # Triton 커널 correctness (GPU 필요)
```

## 프로젝트 구조

```
hw-wfc/
├── src/                    # 코어 알고리즘
│   ├── state.py            # HWState, LayerNode, superposition
│   ├── constraint.py       # Hard/Soft 제약, 전파
│   ├── cost_model.py       # Roofline + affinity + cache
│   ├── collapse.py         # 엔트로피 기반 붕괴 + 백트래킹
│   └── scheduler.py        # 메인 파이프라인
├── specs/                  # 하드웨어 스펙 (JSON)
├── examples/               # 벤치마크 및 실험
├── tests/                  # Safety Checks
├── docs/
│   ├── result.md / .ko.md              # 연구 결과 (EN/KO)
│   ├── concept.md / .ko.md             # 알고리즘 컨셉 (EN/KO)
│   ├── architecture_guide.md / .ko.md  # 설계 의도 해설 (EN/KO)
│   └── daily_logs/                     # 작업 일지
└── HANDOFF.md              # 프로젝트 상태 및 미래 과제
```

## 요구사항

- Python 3.10+
- scipy (상관관계 분석)
- Pillow (시각화)
- PyTorch + Triton (GPU 검증, `pip install torch triton`)

## 라이선스

연구용 프로토타입. 프로덕션 사용 불가.
