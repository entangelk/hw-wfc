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
