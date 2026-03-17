# Work Log — 2026-03-17

## 목표
README의 핵심 결과 표가 현재 코드 기준으로 사실인지 재검증하고, 재현 가능한 검증 경로를 남긴다.

## 완료한 작업

### 1. README 수치의 근거 추적

| 항목 | 확인 결과 |
|------|-----------|
| `396 → 6 states` | 현재 `examples/attention_block.py`와 benchmark 재실행으로 확인 |
| `262K combinations` | 각 레이어 top-8 후보만 전수 탐색할 때 `8^6 = 262,144`로 확인 |
| `100% quality` | 전환 비용 포함 objective에서 WFC와 exhaustive baseline의 최종 선택이 동일함을 확인 |
| `~2400x` | 현재 환경에서 `stress_gpu` 기준 약 2384x로 재현됨 |

핵심 발견:
- 기존 README의 `262K`는 **full grid search가 아니라 top-k exhaustive baseline**이었다.
- `examples/attention_block.py`는 exhaustive search가 아니라 **독립 최적 upper bound**를 출력한다.
- 따라서 README가 사실과 완전히 틀린 것은 아니지만, **검증 조건이 빠져 있어 오해 소지**가 있었다.

### 2. 재현용 benchmark 스크립트 추가

| 파일 | 변경 내용 |
|------|----------|
| `examples/attention_exhaustive_benchmark.py` | 6-layer attention block에 대해 WFC vs top-k exhaustive baseline을 직접 비교하는 재현 스크립트 추가 |
| `README.md` | benchmark 전제(top-8/layer), 현재 수치, 재현 명령 반영 |
| `README.ko.md` | EN과 동일하게 benchmark 전제와 현재 수치 반영 |
| `HANDOFF.md` | 다음 작업자가 `262K`의 의미를 오해하지 않도록 메모 추가 |
| `CHANGELOG.md` | 오늘의 benchmark 재검증 작업 기록 추가 |

### 3. benchmark 재실행 결과

#### A. README 기준 스펙: `stress_gpu.json` (12KB SRAM)

명령:
```bash
python3 examples/attention_exhaustive_benchmark.py
```

| 항목 | 값 |
|------|----|
| 탐색 공간 축소 | `396 → 6` |
| WFC 시간 | `3.065ms` avg (`8`회) |
| exhaustive baseline | `262,144` 조합, `7306.472ms` |
| 품질 | `100.0000%` |
| 최종 선택 동일성 | `True` |
| 속도 향상 | `2383.7x` |

최종 선택:
- `QKV_Proj` → `32x32 ROW_MAJOR SRAM`
- `QK_MatMul` → `32x32 ROW_MAJOR SRAM`
- `Softmax` → `16x16 ROW_MAJOR SRAM`
- `AV_MatMul` → `32x32 ROW_MAJOR SRAM`
- `Out_Proj` → `32x32 ROW_MAJOR SRAM`
- `LayerNorm` → `32x32 ROW_MAJOR SRAM`

#### B. 비교 스펙: `toy_gpu.json` (64KB SRAM)

명령:
```bash
python3 examples/attention_exhaustive_benchmark.py --spec toy_gpu.json
```

| 항목 | 값 |
|------|----|
| 탐색 공간 축소 | `396 → 6` |
| WFC 시간 | `4.627ms` avg (`8`회) |
| exhaustive baseline | `262,144` 조합, `7226.819ms` |
| 품질 | `100.0000%` |
| 최종 선택 동일성 | `True` |
| 속도 향상 | `1561.8x` |

관찰:
- `stress_gpu`에서는 Softmax만 `16x16`으로 달라져 이질적 레이어 차별화가 드러난다.
- `toy_gpu`에서는 여전히 전 레이어가 `64x64 ROW_MAJOR SRAM`으로 수렴한다.
- 절대 시간은 환경 영향이 크므로 README에는 `~3ms`, `~7s`, `~2400x`처럼 **반올림된 값**만 유지했다.

### 4. Safety Check 재실행

명령:
```bash
python3 tests/test_safety_checks.py
```

| Check | 이전 기준 | 오늘 결과 | 상태 |
|-------|----------|----------|------|
| #2 Cost Model spread | `> 0.01` | `1.1944` | 통과 |
| #3 전파 제거율 (`t=1.5`) | `> 25%` | `43%` | 통과 |
| #4 백트래킹 | `> 0` | `1회 발동` | 통과 |
| #5 복사기 문제 | `False` | `all_same_state=False` (`stress_gpu`) | 통과 |

악화/개선:
- 코어 알고리즘은 변경하지 않았으므로 Safety Check 결과는 기존과 동일했다.
- 이번 작업은 **문서 정합성 회복과 재현성 확보**가 목적이었다.

### 5. 추가 검증: top-k sweep (다방면 재검증)

명령:
```bash
python3 - <<'PY'
# stress_gpu / toy_gpu × top-k = 4,6,8,10 exhaustive sweep
PY
```

#### A. `stress_gpu.json`

| top-k | 조합 수 | exhaustive 시간 | 품질 | 최종 선택 동일성 |
|-------|--------|----------------|------|------------------|
| 4 | `4,096` | `126.756ms` | `100.0000%` | `True` |
| 6 | `46,656` | `1324.747ms` | `100.0000%` | `True` |
| 8 | `262,144` | `9038.135ms` | `100.0000%` | `True` |
| 10 | `1,000,000` | `30870.830ms` | `100.0000%` | `True` |

WFC 측정:
- 탐색 공간: `396 → 6`
- 평균 시간: `3.458ms`
- 최종 선택: `32x32, 32x32, 16x16, 32x32, 32x32, 32x32`

#### B. `toy_gpu.json`

| top-k | 조합 수 | exhaustive 시간 | 품질 | 최종 선택 동일성 |
|-------|--------|----------------|------|------------------|
| 4 | `4,096` | `117.247ms` | `100.0000%` | `True` |
| 6 | `46,656` | `1367.897ms` | `100.0000%` | `True` |
| 8 | `262,144` | `6780.626ms` | `100.0000%` | `True` |
| 10 | `1,000,000` | `31500.510ms` | `100.0000%` | `True` |

WFC 측정:
- 탐색 공간: `396 → 6`
- 평균 시간: `4.061ms`
- 최종 선택: 전 레이어 `64x64 ROW_MAJOR SRAM`

해석:
- top-k를 `4 → 10`으로 넓혀도 WFC의 선택이 계속 exhaustive 최적과 일치했다.
- 즉 README 숫자는 특정 `top-8` 한 번의 우연이 아니라, **적어도 현재 attention block과 두 대표 스펙에서는 안정적인 결과**였다.

### 6. 시각화 재생성 및 로그 정리

명령:
```bash
python3 tools/generate_all_visuals.py
```

결과:
- `collapse_animation.gif`
- `score_distribution.png`
- `propagation_sweep.png`
- `copier_problem.png`
- `backtracking.png`
- `sram_comparison.png`

추가 수정:
- `tools/generate_all_visuals.py`의 `VISUALS_LOG.md` 누적 로직을 소폭 수정
- 이전 로그를 재생성할 때 History 헤더가 중복되는 문제를 제거
- 신규 생성 로그 버전을 현재 작업 기준 `v2.7.1`로 수정

## 기술 결정

### 1. README에는 full grid search라는 표현을 쓰지 않음
- 이유: 현재 재현 가능한 baseline은 hard constraint 이후 각 레이어 top-k 후보만 대상으로 한 exhaustive search임
- 결과: `Grid Search equivalent`를 `Exhaustive baseline`으로 수정

### 2. benchmark 숫자는 현재 환경 기준으로 반올림
- 이유: `2.5ms`, `6000ms`처럼 고정 절대값은 장비/부하에 따라 쉽게 흔들림
- 결과: `~3ms`, `~7s`, `~2400x`로 완화하고 재현 명령을 함께 명시

### 3. `attention_block.py`는 유지, 별도 benchmark 스크립트 추가
- 이유: 기존 예제는 upper-bound 비교용으로도 유효하고, 코어 예제를 불필요하게 복잡하게 만들 필요가 없었음
- 결과: README 전용 검증 경로를 `examples/attention_exhaustive_benchmark.py`로 분리

### 4. visual log 생성기는 이번 작업 범위에서 바로 보정
- 이유: 시각화를 재생성하자 `assets/VISUALS_LOG.md`에 History 헤더가 중복되어 로그 가독성이 떨어졌음
- 결과: 생성기 로직을 최소 수정해 이후 재생성에서도 헤더가 한 번만 남도록 정리

## 다음 단계
1. `examples/attention_block.py` 출력에 "upper bound, not exhaustive"를 더 눈에 띄게 표시할지 검토
2. top-k를 10보다 더 넓히거나 다른 모델 블록(ResNet/MobileNet)에도 같은 검증을 적용할지 검토
3. Phase 4 Task 10 성능 최적화(엔트로피 계산 캐싱) 진행

---

## 추가 재감사 (Late Audit)

README 수치를 다시 의심하는 관점에서 full-candidate exact solver까지 비교해 보니, 기존 결론 일부가 과장되어 있었음을 확인했다.

### 1. 핵심 반례: `stress_gpu`에서 full exact DP가 WFC보다 더 좋은 해를 찾음

| 비교 | 점수 | 최종 선택 |
|------|------|-----------|
| WFC heuristic | `2.8494` | `32, 32, 16, 32, 32, 32` |
| Full exact DP | `2.8996` | `32, 32, 32, 32, 32, 32` |

품질:
- `2.8494 / 2.8996 = 98.2687%`

의미:
- 기존의 `quality 100%`는 **full candidate set 기준이 아니라 top-k로 잘린 후보 집합 기준**이었다.
- 현재 greedy WFC heuristic은 `stress_gpu`에서 full exact optimum과 일치하지 않는다.

### 2. top-k exhaustive baseline이 global optimum을 놓친 이유

`stress_gpu` Softmax의 unary ranking:
- `16x16 ROW_MAJOR SRAM` → rank 1
- `32x32 ROW_MAJOR SRAM` → **rank 35**

즉 `top-k = 8/10/20/32`에서는 full optimum에 필요한 `32x32 ROW_MAJOR SRAM`이 candidate set에 들어오지 않는다.
실제로:

| top-k | full optimum 포함 여부 | 결과 |
|-------|----------------------|------|
| 8 | X | WFC와 동일한 잘린 optimum |
| 10 | X | 동일 |
| 20 | X | 동일 |
| 32 | X | 동일 |
| 42 (전체) | O | full exact DP와 일치 |

따라서 `top-k sweep에서 quality 100% 유지`는 heuristic의 강건성을 보여주긴 하지만, **global optimality의 증거는 아니다**.

### 3. `~2400x`의 정체

기존 benchmark는 다음 비교였다:
- WFC heuristic vs
- top-8 후보를 대상으로 한 **naive exhaustive enumeration**

추가 측정 결과 (`stress_gpu`):

| baseline | 시간 | 비고 |
|----------|------|------|
| WFC heuristic | `2.607ms` avg | 현재 스케줄러 |
| Naive top-8 exhaustive | `7100.891ms` | 기존 README의 `~2400x` 근거 |
| Precomputed top-8 exhaustive | `363.276ms` avg | 동일 top-8이지만 score 재계산 제거 |
| Full exact DP | `13.742ms` avg | 전체 hard-constrained 후보에 대한 exact solver |
| Precomputed full exact DP | `1.9623ms` avg | chain objective 전용 exact solver |

해석:
- `~2400x`는 틀린 숫자는 아니지만, **naive truncated exhaustive 대비**라는 강한 조건이 붙는다.
- 더 공정한 exact baseline인 full DP와 비교하면 이야기가 완전히 달라진다.
- 특히 현재 attention block의 1D chain objective에서는 specialized DP가 존재하므로, `2400x`를 일반적인 algorithmic win으로 해석하면 안 된다.

### 4. 문서상 잘못된 물리 주장

기존 문서에는 다음과 같은 설명이 있었다:
- “Softmax 32x32는 12KB에서 working memory 초과라 사용 불가”

하지만 현재 코드에서는:
- [src/constraint.py](/mnt/d/devel/hw-wfc/src/constraint.py) 의 hard constraint는 raw tile bytes만 본다
- working-memory multiplier는 [src/cost_model.py](/mnt/d/devel/hw-wfc/src/cost_model.py) 의 cache penalty에서만 반영된다

즉 `32x32 Softmax`는 **hard-prune되지 않는다**.
문제는 “물리적으로 불가능”이 아니라 “현재 heuristic/cost model에서 강한 penalty를 받는다”가 정확한 설명이다.

### 5. 후속 조치

이번 추가 감사에 맞춰 아래를 수정했다:
- `examples/attention_exhaustive_benchmark.py`에 full exact DP 비교 추가
- README EN/KO에서 `~2400x`를 naive truncated exhaustive 기준으로 재정의
- README EN/KO에서 `Softmax 32x32는 사용 불가` 식 서술 제거
- architecture guide EN/KO에서 같은 물리 주장 완화
- visuals generator와 safety check 주석도 heuristic 중심 문구로 정정

### 6. 작업 지침 강화: README/benchmark 정직성 규칙 추가

요청:
- 앞으로 README에는 정직한 결과만 작성
- 테스트 함수 / benchmark 함수 / 비교 함수 작성 시 동일 조건, 공정 비교, 객관적 평가를 강제

수정 파일:
| 파일 | 변경 내용 |
|------|----------|
| `CLAUDE.md` | §6.5 추가: benchmark/test 비교 규칙 명문화 |
| `AGENTS.md` | §6.5 추가: benchmark/test 비교 규칙 명문화 |

핵심 규칙:
- 같은 objective가 아니면 비교 금지
- 가능하면 exact baseline을 먼저 구할 것
- top-k / sampling / truncation / precompute를 숨기지 말 것
- naive baseline만 보고 speedup을 과장하지 말 것
- hard constraint와 soft penalty를 문서에서 혼동하지 말 것
- 새 benchmark 버그가 보이면 기존 README/guide/visuals까지 전면 재감사 대상으로 볼 것

### 7. `.venv` 기반 GPU 재검증

배경:
- 로컬 기본 Python에는 `torch`/`triton`이 없었지만, `.venv`에는 `torch 2.10.0+cu128`, `triton 3.6.0`이 설치되어 있었음
- 샌드박스 밖에서는 RTX 3060이 정상 인식됨

확인 결과:

#### A. Triton correctness 재검증

명령:
```bash
.venv/bin/python examples/triton_verify.py
```

결과:
- GPU: `NVIDIA GeForce RTX 3060`
- Linear_0: `PASS [rel_err=2.44e-03]`
- ReLU_1: `PASS [max_diff=0.00e+00]`
- Softmax_2: `PASS [max_diff=1.12e-08]`
- Linear_3: `PASS [rel_err=3.45e-03]`

판단:
- README의 "GPU correctness verified" 문구는 현재 `.venv` 기준으로 재현됨

#### B. Triton autotuner 성능 비교 재검증

명령:
```bash
.venv/bin/python examples/triton_autotuner_compare.py
```

1차 실행:
- Average WFC/Autotuner ratio: `118.0%`

2차 실행:
- Average WFC/Autotuner ratio: `84.0%`

의미:
- 같은 `.venv`, 같은 RTX 3060에서도 평균 비율이 크게 흔들림
- 따라서 기존 README의 고정 TFLOPS 표와 `평균 90%` 문구는 현재 기준으로 신뢰하기 어려움

후속 조치:
- README EN/KO에서 autotuner 표를 제거하고 "재감사 중"으로 낮춤
- correctness는 유지, performance ratio는 benchmark 안정화 전까지 확정 수치로 쓰지 않음

### 8. 폐쇄/독립 자원 조건에서 autotuner variance 재검증

요청:
- 공유 캐시나 외부 GPU 부하가 아니라도 결과가 흔들리는지 확인

코드 레벨 점검:
- `examples/triton_autotuner_compare.py`는 WFC와 autotuner를 **서로 다른 랜덤 텐서**로 벤치한다
- correctness 검증 없이 throughput만 비교한다
- `Small (256x512x256)`는 실제 측정 시간이 `0.01~0.02ms` 수준이라 autotuner 내부 탐색이 매우 노이즈에 민감해질 수 있다
- 다만 WFC가 고르는 `64x128x64`는 autotuner config 집합 안에도 존재하므로, "WFC가 autotuner 탐색공간 밖의 설정을 쓴다"는 문제는 아니었다

격리 실행 명령:
```bash
for i in 1 2 3 4 5; do
  TRITON_CACHE_DIR=/tmp/hw_wfc_triton_iso_$i .venv/bin/python examples/triton_autotuner_compare.py
done
```

추가 확인:
- 각 런 전 `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` 결과는 비어 있었음
- 즉 관측 가능한 다른 GPU compute process는 없었음

격리 실행 결과:

| Run | Average WFC/Autotuner | Small auto tile | 해석 |
|-----|------------------------|-----------------|------|
| 1 | `88.8%` | `32x64x32` | autotuner 우세 |
| 2 | `94.6%` | `32x32x32` | Small만 반전, 나머지는 여전히 autotuner 우세 |
| 3 | `91.5%` | `32x64x32` | 비슷한 패턴 반복 |
| 4 | `89.8%` | `32x64x32` | Large/GPT-2/BERT에서 autotuner 우세 |
| 5 | `86.0%` | `32x64x32` | 가장 큰 격차, 그래도 방향성은 동일 |

워크로드별 패턴:
- `Small (256x512x256)`:
  - autotuner tile이 `32x64x32`와 `32x32x32` 사이에서 흔들림
  - 실행 시간이 너무 짧아 (`~0.01-0.02ms`) 선택 변동이 큼
- `Medium`, `GPT-2 FFN`, `BERT QKV`:
  - autotuner가 대부분 `64x64x32`
- `Large (2048^3)`:
  - autotuner가 `128x128x32`를 일관되게 선택
- 큰 workload에서는 현재 WFC `64x128x64`보다 autotuner가 대체로 `5-20%` 빠른 쪽으로 수렴

판단:
- **폐쇄된 환경 + 독립 cache**로 돌려도 variance는 사라지지 않았다
- 따라서 원인은 "외부 프로세스 간섭"보다는 benchmark 방법론과 autotuner의 작은 workload 선택 노이즈에 더 가깝다
- 다만 완전히 무작위는 아니고, 큰 workload에서는 autotuner 우세라는 방향성은 반복 재현된다

후속 조치:
- README EN/KO에 isolated rerun 결과 범위(`86.0% ~ 94.6%`) 반영
- HANDOFF의 기존 `평균 90%` 서술을 폐기하고 재감사 상태로 교체
- 다음 개선 항목으로 shared input / correctness check / tuning time vs steady-state time 분리 추가

### 9. 왜 흔들리는가: `Small` 특성인지, WFC(양자 붕괴) 특성인지 분리 분석

요청:
- 앞으로도 측정 설계가 흔들리지 않도록 작업 지침에 명시
- autotuner variance가 `Small` 때문인지, "양자 붕괴"의 특성 때문인지 확인

#### A. WFC는 이 benchmark에서 비결정적이지 않다

코드 확인:
- [src/collapse.py](/mnt/d/devel/hw-wfc/src/collapse.py) 의 `select_best_state()`는 랜덤 샘플링 없이 최고 점수 상태를 **결정론적으로** 선택
- `examples/triton_autotuner_compare.py`는 노드 1개짜리 MatMul benchmark라, 전파/백트래킹/복합 붕괴보다 unary score 선택이 사실상 전부

즉 이번 변동성은 "양자 붕괴라서 매번 다른 상태를 뽑는다"는 종류의 현상이 아니다.

#### B. 현재 MatMul unary cost model은 workload를 거의 구분하지 못한다

CPU 측 score audit 결과:

| Workload | WFC top score | 선택 |
|----------|---------------|------|
| Small | `64x128 ROW_MAJOR SRAM = 1.2905` | 동일 |
| Medium | `64x128 ROW_MAJOR SRAM = 1.2905` | 동일 |
| Large | `64x128 ROW_MAJOR SRAM = 1.2905` | 동일 |
| GPT-2 FFN | `64x128 ROW_MAJOR SRAM = 1.2905` | 동일 |
| BERT QKV | `64x128 ROW_MAJOR SRAM = 1.2905` | 동일 |

원인:
- [src/cost_model.py](/mnt/d/devel/hw-wfc/src/cost_model.py) 에서 MatMul top 후보들의 `roofline_score`가 위 workload 전부에서 `1.0000`으로 포화됨
- 결국 LINEAR의 점수는 layout affinity + cache bonus가 지배하고, 그 결과 `64x128`이 모든 workload에서 같은 점수로 선택됨

대표 분해:

| Tile | roofline | affinity | cache | total |
|------|----------|----------|-------|-------|
| `64x128` | `1.0000` | `0.1500` | `0.1405` | `1.2905` |
| `64x64` | `1.0000` | `0.1500` | `0.1116` | `1.2616` |
| `32x32` | `1.0000` | `0.1500` | `0.0084` | `1.1584` |

판단:
- WFC가 workload별로 다른 tile을 못 고르는 가장 큰 이유는 현재 single-node MatMul benchmark에서 cost model이 workload 차이를 거의 못 보는 구조이기 때문

#### C. autotuner variance는 주로 `Small`의 초단시간 측정 노이즈다

고정 config 직접 비교 (`.venv`, RTX 3060):

1회 대표 결과:

| Workload | Config | 시간 | TFLOPS |
|----------|--------|------|--------|
| Small | WFC `64x128x64` | `0.0136ms` | `4.918` |
| Small | `32x64x32` | `0.0103ms` | `6.509` |
| Small | `32x32x32` | `0.0102ms` | `6.564` |
| Large | WFC `64x128x64` | `0.9831ms` | `17.476` |
| Large | `128x128x32` | `0.8727ms` | `19.687` |

추가로 5회 isolated rerun:

| Run | Small WFC | Small 32x64x32 | Small 32x32x32 | Large WFC | Large 128x128x32 |
|-----|-----------|----------------|----------------|-----------|------------------|
| 1 | `4.471` | `6.222` | `6.254` | `17.382` | `18.952` |
| 2 | `4.976` | `5.494` | `5.432` | `18.180` | `19.701` |
| 3 | `5.132` | `6.570` | `6.282` | `19.899` | `20.316` |
| 4 | `5.063` | `5.599` | `5.755` | `19.129` | `22.023` |
| 5 | `4.998` | `6.493` | `5.273` | `17.676` | `20.126` |

해석:
- `Small`에서는 autotuner 후보인 `32x64x32`와 `32x32x32`가 실제 성능도 거의 비슷하고, 실행 시간이 `~0.01ms` 수준이라 순위가 쉽게 흔들린다
- `Large`에서는 `128x128x32`가 매번 `64x128x64`보다 빠르며 방향성이 안정적이다

최종 판단:
- variance의 **직접 원인**은 주로 `Small` workload의 초단시간 microbenchmark 특성
- 비교가 어색해진 **구조적 원인**은 현재 WFC MatMul unary cost model이 workload를 거의 구분하지 못해 모든 workload에서 같은 tile을 고른다는 점
- 따라서 이번 현상은 "양자 붕괴의 본질적 특성"이라기보다, benchmark 설계와 cost model 설계 문제에 가깝다

#### D. 작업 지침 강화

수정 파일:
| 파일 | 변경 내용 |
|------|----------|
| `CLAUDE.md` | §6.6 추가: 측정 설계 안정성 규칙 |
| `AGENTS.md` | §6.6 추가: 측정 설계 안정성 규칙 |

새 규칙 핵심:
- 같은 입력/같은 버퍼 조건 사용
- correctness 확인 후 timing
- compile/tune/search 시간과 steady-state 시간 분리
- 너무 짧은 microbenchmark는 noise-sensitive로 취급
- 최소 3~5회 isolated rerun 뒤 문서화
- 순위가 뒤집히면 단일 평균 대신 범위/분포 사용
