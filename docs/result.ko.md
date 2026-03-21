<p align="center">
  <a href="./result.md"><img src="https://img.shields.io/badge/Language-EN-6B7280?style=for-the-badge" alt="English"></a>
  <a href="./result.ko.md"><img src="https://img.shields.io/badge/Language-KO-111111?style=for-the-badge" alt="한국어"></a>
</p>

# HW-WFC 연구 결과

> **버전**: v2.9 (2026-03-21)
> **목표**: Wave Function Collapse (WFC) 알고리즘을 AI 하드웨어 컴파일러 자동 스케줄링에 적용하는 것의 타당성을, 통제된 소프트웨어 환경에서 평가한다.

---

## 1. 연구 질문

**제약 기반 붕괴 탐색(WFC)이 실제 커널 실행 없이 경쟁력 있는 하드웨어 스케줄링 결정을 내릴 수 있는가?**

가설: cost model이 타일 설정의 예상 성능 순위를 정확하게 매길 수 있다면, WFC의 제약 전파와 엔트로피 기반 붕괴가 조합 탐색 공간을 효율적으로 탐색하여 최적에 가까운 스케줄을 찾을 수 있다.

---

## 2. 성공한 것

### 2.1 알고리즘 정확성

WFC heuristic이 벤치마크 문제에서 exact dynamic programming (Viterbi 알고리즘)과 동일한 결과를 산출한다.

| 지표 | 값 |
|------|-----|
| WFC vs Exact DP 품질 | **100%** (`stress_gpu`, `toy_gpu` 모두 동일한 상태 선택) |
| WFC 스케줄링 시간 | ~2-3ms |
| Exact DP 시간 | ~7-10ms |
| 탐색 공간 | 33^1 × 36^5 ≈ 20억 조합 |

두 알고리즘 모두 12KB SRAM에서 Softmax에 `16x16`(working memory hard constraint), 나머지 레이어에 `32x32`를 선택한다. WFC heuristic은 greedy이므로 일반적으로 최적을 보장하지 않지만, 이 chain 구조 벤치마크에서는 exact 최적해에 도달한다.

### 2.2 Cost Model과 실제 GPU의 상관관계

RTX 3060에서 cost model 점수와 실제 Triton 커널 실행 시간의 Spearman 순위 상관계수:

| 레이어 타입 | 워크로드 | Spearman ρ | p-value | 유의성 |
|------------|----------|-----------|---------|--------|
| Softmax | Medium (512x1024) | **+1.00** | < 0.001 | *** |
| Softmax | Large (1024x2048) | **+1.00** | < 0.001 | *** |
| Softmax | Attn (512x512) | **+1.00** | < 0.001 | *** |
| MatMul | Large (2048x2048) | **+0.72** | < 0.05 | * |
| MatMul | GPT-2 FFN (768x3072) | +0.66 | 0.08 | ns |
| MatMul | Medium (1024x1024) | +0.60 | 0.10 | ns |
| MatMul | BERT QKV (512x768) | +0.40 | 0.33 | ns |
| MatMul | Small (256x512) | -0.69 | 0.07 | ns |
| Softmax | Small (256x512) | +0.50 | 0.67 | ns |

**평균 ρ = +0.52** (9개 워크로드).

해석:
- **Softmax**: 완벽한 순위 일치. Cost model의 memory traffic 추정이 `BLOCK_N` 증가에 따른 행 반복 감소를 정확히 반영.
- **MatMul (대형 워크로드)**: 양의 상관관계. 큰 타일이 데이터 재사용률 개선으로 대체로 빠르다는 것을 올바르게 예측.
- **MatMul (소형 워크로드)**: ~0.01ms 수준의 실행 시간에서 측정 노이즈가 지배적이어서 순위가 역전됨.

재현: `python examples/cost_model_correlation.py` (RTX 3060, CUDA 12.8).

### 2.3 Hard Constraint

물리적 제약이 정확하게 적용되며, 의미 있는 차별화를 유발한다:

| 제약 | 메커니즘 | 효과 (stress_gpu 12KB) |
|------|----------|------------------------|
| SRAM 용량 | `tile_bytes > sram_bytes` → 제거 | 큰 타일의 SRAM 후보 제거 |
| Alignment | `row_bytes % alignment != 0` → 제거 | 128B 정렬 스펙에서 8x8, 16x16 제거 |
| Working memory | `tile_bytes × multiplier > sram_bytes` → 제거 | Softmax 32x32 (4x × 4KB = 16KB > 12KB) 제거 |

Working memory 제약이 핵심 차별화 요인이다: Softmax가 연산 바운드 레이어보다 작은 타일을 사용하게 강제하며, 이 차별화는 WFC와 exact DP 모두에서 동일하게 확인된다.

---

## 3. 실패한 것

### 3.1 기존 방법 대비 우위 없음

| 비교 대상 | 결과 | 평가 |
|-----------|------|------|
| WFC vs Exact DP | 동일 품질, WFC ~3-5배 빠름 | 둘 다 ms 단위로 극히 빠름. 속도 차이에 실질적 의미 없음. |
| WFC vs Triton Autotuner | Autotuner가 대형 워크로드에서 5-20% 더 빠른 설정 발견 | Autotuner가 우세. 실제 실행 시간을 측정하기 때문. |
| WFC vs Naive Exhaustive | ~3500배 speedup | **허수.** §3.2 참고. |

WFC 알고리즘은 이 문제에서 어떤 잘 구현된 baseline도 이기지 못했다.

### 3.2 Naive Exhaustive Speedup은 유효하지 않음

~3500배 speedup (WFC ~2ms vs naive exhaustive ~7s)은 알고리즘적 우위가 아니라 구현 품질의 차이이다:

1. **최적화되지 않은 Python 순회**: Naive baseline은 벡터화, memoization, early termination 없이 tight Python 루프에서 점수를 재계산한다.
2. **가지치기 없음**: `8^6 = 262,144`개 조합을 모두 평가한다.
3. **Exact DP가 ~7-10ms에 해결**: Viterbi 알고리즘이 같은 chain objective의 증명된 최적해를 찾으므로, naive exhaustive 자체가 불필요하다.

공정한 speedup 주장을 하려면 baseline을 최적화해야 한다:
- Cost model 상한을 활용한 **branch-and-bound** 가지치기
- 레이어당 top-k 상태를 유지하는 **beam search**
- 전환 비용 행렬을 사전 계산하는 **벡터화 DP** (NumPy/C++)
- Python overhead를 제거하는 **C++/CUDA 구현**

이러한 baseline이 구현되기 전까지 speedup 주장은 유효하지 않다.

### 3.3 Cost Model 포화

GPU의 메모리 대역폭이 충분한 경우, 모든 MatMul 타일 설정의 roofline score가 1.0으로 포화된다. 이 경우 유일한 변별 요소는 `cache_efficiency` — 부차적 heuristic 컴포넌트 — 이며, workload별 타일 추천 능력이 제한된다.

| 워크로드 | Roofline 점수 | 변별 원인 |
|----------|--------------|-----------|
| MatMul (모든 크기) | 전부 1.0 | cache_efficiency만 |
| Softmax | BLOCK_N에 따라 변동 | memory traffic (올바름) |

이것이 cost model의 평균 상관관계(ρ = +0.52)가 신뢰할 수 있는 예측기에 미치지 못하는 주된 원인이다.

### 3.4 가상 스펙 의존성

핵심 차별화 시나리오(12KB SRAM)는 현존하는 어떤 GPU에도 없다. 실제 하드웨어에서는:
- A100 (192KB SRAM): 모든 타일이 여유롭게 들어감. 제약이 활성화되지 않음. 문제가 trivial해짐.
- H100 (256KB SRAM): 동일한 상황.

알고리즘은 인위적으로 tight한 제약 조건에서만 가치를 입증한다. 실제 스케줄링 문제에서 유사하게 제약된 시나리오가 발생하는지(예: multi-tenant SRAM 분할, fusion으로 확장된 working set)는 미해결 질문이다.

---

## 4. 실험 조건

| 파라미터 | 값 |
|----------|-----|
| GPU | NVIDIA RTX 3060 (GA106, SM86) |
| CUDA | 12.8 |
| Triton | OpenAI Triton (pip install) |
| dtype | float16 (MatMul), float32 (Softmax) |
| 벤치마크 방법 | `triton.testing.do_bench` (warmup=25, rep=100) |
| Correctness 확인 | Timing 전에 수행; rel_err < 5% (MatMul), max_diff < 1e-5 (Softmax) |
| 입력 텐서 | 워크로드당 모든 설정에서 동일 사용; `torch.manual_seed(42)` |
| 가상 스펙 | `stress_gpu.json` (12KB SRAM, 64B align), `toy_gpu.json` (64KB SRAM, 128B align) |

### 측정의 한계

- SRAM + ROW_MAJOR 설정만 측정 가능. Triton은 layout이나 compute location을 튜닝 파라미터로 노출하지 않음.
- `num_warps`와 `num_stages`는 고정 heuristic으로 파생. 설정별 최적화가 아님.
- 소형 워크로드(~0.01ms)는 노이즈에 취약. 해당 워크로드의 순위는 신뢰할 수 없음.
- 단일 GPU 모델의 결과. 다른 아키텍처에서 상관관계가 달라질 수 있음.

---

## 5. 결론

### 이 연구가 보여준 것

1. **WFC는 제약 기반 하드웨어 스케줄링의 유효한 탐색 알고리즘이다.** 탐색 공간을 올바르게 탐색하고, 물리적 제약을 준수하며, 벤치마크 문제에서 exact DP와 일치한다.
2. **물리 기반 cost model은 GPU 성능의 방향성 있는 proxy 역할을 할 수 있다.** Softmax는 완벽한 순위 상관관계(ρ = 1.0), MatMul은 non-trivial 워크로드에서 양의 상관관계(ρ = +0.72)를 달성한다.
3. **가장 의미 있는 스케줄링 결정은 heuristic 점수가 아니라 hard constraint가 이끈다.** Softmax 타일 차별화는 전적으로 working memory 제약에 의해 결정되며, 점수 최적화에 의한 것이 아니다.

### 이 연구가 보여주지 못한 것

1. **기존 방법 대비 우위 없음.** WFC heuristic은 exact DP, Triton autotuner, 또는 어떤 잘 최적화된 baseline도 이기지 못했다.
2. **실제 하드웨어에서의 실용적 가치는 미검증.** 의미 있는 제약 시나리오는 가상 스펙에서만 발생한다.
3. **Cost model은 workload별 추천을 내리기에 충분히 정밀하지 않다.** 평균 ρ = +0.52는 방향성은 맞지만 프로덕션 스케줄링 결정에는 부족하다.

### 실용적 가치로의 경로

이 연구는 하나의 명확한 경로를 식별한다: **cost model 교정**.

Cost model의 순위 상관관계를 ρ = +0.52에서 ρ ≥ +0.8로 개선할 수 있다면 — GPU 프로파일링 데이터 피팅을 통해 — WFC 접근법은 구체적인 이점을 제공할 수 있다: **하드웨어에서 커널을 실행하지 않고 좋은 타일 설정을 예측하는 것**. 이는 다음에 유용할 것이다:
- 물리적으로 사용 불가능한 GPU를 대상으로 하는 크로스 컴파일
- 하드웨어/컴파일러 공동 설계 시 빠른 design-space 탐색
- Autotuner 정밀 탐색 전의 초기 스케줄 생성

이 교정에는 다양한 워크로드에 걸친 실제 GPU 프로파일링 데이터가 필요하며, 이는 소프트웨어 전용 실험의 범위를 넘어선다.

---

## 6. 재현

```bash
# 핵심 벤치마크 (WFC vs exact DP)
python examples/attention_exhaustive_benchmark.py

# Cost model vs GPU 실행 시간 상관관계
python examples/cost_model_correlation.py

# Safety checks (5/5)
python tests/test_safety_checks.py

# Triton 커널 correctness 검증
python examples/triton_verify.py
```

요구사항: Python 3.10+, scipy, PyTorch, Triton, CUDA 지원 GPU.
