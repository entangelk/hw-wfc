# HANDOFF.md

## 프로젝트 상태
HW-WFC v2.9 연구 프로토타입 완료. Phase 1~4 전체 완료.
2026-03-20 전면 감사 후 working memory hard constraint 도입: Softmax 32x32 SRAM(4x=16KB>12KB)이 hard constraint에 의해 물리적으로 제거됨. WFC와 exact DP가 100% 동일한 결과를 산출.
2026-03-17 기준 README 벤치마크 문구를 재검증했고, `262K`는 full grid search가 아니라 top-8 exhaustive baseline임을 명시함.
추가 검증으로 `stress_gpu`/`toy_gpu`에서 `top-k = 4/6/8/10` sweep을 돌렸고, 모두 WFC와 exhaustive 최적 선택이 일치했다.
추가 late-audit 결과, `stress_gpu`에서는 full exact DP가 WFC보다 더 높은 점수(`2.8996` vs `2.8494`)를 찾았고, 기존 `~2400x`는 naive truncated exhaustive 대비 수치임이 확인되었다.
`CLAUDE.md`와 `AGENTS.md`에도 README/benchmark 정직성 규칙을 추가해, 앞으로 테스트/비교 함수 작성 시 exact baseline, 동일 objective, truncation 명시를 강제하도록 했다.
추가로 측정 설계 안정성 규칙도 넣어, 같은 입력 사용, correctness 선행, compile/tune vs steady-state 분리, isolated rerun 확인 없이 고정 benchmark 수치를 문서화하지 않도록 했다.
`.venv`를 사용한 GPU 재검증 결과, Triton correctness는 재현됐지만 autotuner 성능 비율은 같은 RTX 3060에서도 `118%`와 `84%`로 크게 흔들렸다.
추가로 별도 프로세스 + 별도 `TRITON_CACHE_DIR`로 5회 격리 재실행해도 평균 비율이 `86.0% ~ 94.6%` 범위에 머물러, README 성능 표는 계속 보류 상태로 유지한다.
원인 분석 결과, 이 흔들림은 "양자 붕괴"의 비결정성 때문이 아니라 `Small (256x512x256)`의 초단시간 microbenchmark 노이즈와 현재 MatMul unary cost model의 workload 둔감성이 겹친 결과로 보인다.

## 프로젝트 구조
```
hw-wfc/
├── src/
│   ├── state.py          # LayerNode (neighbors DAG), HWState, 직사각 타일 포함
│   ├── constraint.py     # Hard/Soft 제약, HardwareSpec, 전파, derive_transition_weight
│   ├── cost_model.py     # Roofline + affinity + cache (v2, Conv2D im2col, Depthwise)
│   ├── collapse.py       # 엔트로피 기반 붕괴 + 백트래킹 + 전환 비용 통합
│   ├── scheduler.py      # 메인 스케줄러 파이프라인 (BFS 그래프 전파, adaptive threshold)
│   ├── model_zoo.py      # GPT-2 Small (72L), BERT-Base (84L) 레이어 정의
│   ├── codegen.py        # Triton 커널 스켈레톤 코드 생성기
│   └── visualize.py      # 시각화 (퍼널, 점수분해, DAG, 스펙비교)
├── .venv/                 # Python venv (torch, triton)
├── specs/
│   ├── toy_gpu.json      # SRAM 64KB, alignment 128B
│   ├── tight_gpu.json    # SRAM 16KB, alignment 64B
│   ├── stress_gpu.json   # SRAM 12KB, alignment 64B (핵심 테스트 스펙)
│   ├── a100.json         # NVIDIA A100: SRAM 192KB, HBM2e 2039 GB/s
│   ├── h100.json         # NVIDIA H100: SRAM 256KB, HBM3 3352 GB/s, alignment 256B
│   └── rtx3060.json      # NVIDIA RTX 3060: SRAM 100KB, GDDR6 360 GB/s
├── examples/
│   ├── toy_model.py      # Linear→ReLU→Linear PoC
│   ├── attention_block.py # 6-layer Transformer Attention Block PoC
│   ├── attention_exhaustive_benchmark.py # README 수치 재현용 top-k exhaustive 비교
│   ├── resnet_block.py   # ResNet Residual Block (DAG skip connection)
│   ├── scaling_benchmark.py  # Phase 2: A100/H100 × GPT-2/BERT 벤치마크
│   ├── mobilenet_block.py    # MobileNetV2 Inverted Residual (Depthwise + skip)
│   ├── triton_verify.py      # Phase 4: Triton 커널 GPU 실행 correctness 검증
│   └── triton_autotuner_compare.py  # Phase 4: WFC vs Triton autotuner 성능 비교
├── tests/
│   ├── test_safety_checks.py     # 5가지 안전 점검 (stress_gpu 기반 #5b)
│   └── debug_score_breakdown.py  # 점수 분해 디버그 도구
├── docs/
│   ├── result.md / .ko.md              # 연구 결과 보고서 (EN/KO)
│   ├── concept.md / .ko.md             # 알고리즘 컨셉 & 동기 (EN/KO)
│   ├── architecture_guide.md / .ko.md  # 설계 의도 해설서 (EN/KO)
│   └── daily_logs/2026-03-{14,15,16,17,20,21}/
└── CLAUDE.md, AGENTS.md   # Section 6: HW-WFC 작업 지침
```

## 스케일링 벤치마크 (v2.5)
| Model | GPU | Layers | States | Time | Unique |
|-------|-----|--------|--------|------|--------|
| GPT-2 Small | A100 | 72 | 5,616 | 530ms | 2/72 |
| BERT-Base | A100 | 84 | 6,552 | 700ms | 2/84 |
| GPT-2 Small | H100 | 72 | 5,616 | 571ms | 1/72 |
| BERT-Base | H100 | 84 | 6,552 | 632ms | 1/84 |

Grid Search: 78^72 = 1.70×10^136 조합 vs WFC ~600ms

## Safety Check 현황 (v2.5)
| Check | 상태 | 비고 |
|-------|------|------|
| #1 Hard Constraint | OK | alignment 128B가 8x8, 16x16 제거 (경계) + working memory hard constraint 추가 |
| #2 Cost Model 변별력 | **통과** | spread=1.19 (toy_gpu), 동점 그룹 존재 (같은 면적 직사각 타일) |
| #3 전파 강도 | **통과** | t=1.5에서 43% (tile_shape_penalty 추가 후) |
| #4 백트래킹 | **통과** | 모순 감지→발동→보고 정상 |
| #5 복사기 문제 | **통과** | WFC: Softmax=16x16, 나머지=32x32. Exact DP도 동일 선택 → 100% 일치 (working memory hard constraint 후) |

## 핵심 발견 사항
1. **README 벤치마크 재검증** (2026-03-17)
   - `examples/attention_exhaustive_benchmark.py` 추가
   - 기본 설정: `stress_gpu.json`, top-8/layer truncated exhaustive baseline
   - 현재 환경 재측정: WFC ~2-3ms, naive truncated exhaustive ~7-9s, match 100% (truncated baseline 대비)
   - 추가 sweep: `top-k = 4/6/8/10`, `stress_gpu`/`toy_gpu` 모두 truncated baseline 기준 same choice=True
   - **Late audit:** `stress_gpu` full exact DP는 WFC와 다른 해를 선택하며 점수가 더 높음 (`2.8996` vs `2.8494`)
   - 따라서 README는 `quality 100%`를 full optimum 주장이 아닌 truncated baseline 일치도로 정정
   - `Softmax 32x32는 물리적으로 불가` 같은 문구도 제거. 현재 코드에서는 working-memory overflow가 hard constraint가 아니라 soft penalty임
   - `tools/generate_all_visuals.py`의 visual log 누적 포맷도 함께 수정
   - `CLAUDE.md` / `AGENTS.md`에 benchmark/test fairness 규칙 추가
   - `CLAUDE.md` / `AGENTS.md`에 측정 설계 안정성 규칙도 추가
   - `.venv` 기반 GPU 재검증: correctness는 PASS, autotuner ratio는 불안정하여 README 고정 TFLOPS 표 제거
   - 격리 재실행 5회에서도 autotuner ratio는 `86.0% ~ 94.6%` 범위. 작은 workload만 특히 불안정하고, 큰 workload에서는 autotuner가 대체로 우세
   - root cause 분석: single-node MatMul benchmark에서 WFC는 결정론적으로 항상 `64x128x64`를 선택하며, 이는 "붕괴"의 랜덤성 때문이 아니라 roofline score가 모든 workload에서 `1.0`으로 포화되어 cost model이 workload를 구분하지 못하기 때문
2. **대규모 스케일링 검증** (v2.5): GPT-2(72L)/BERT(84L) × A100/H100 벤치마크
   - A100: Softmax만 64x128 선택 (working memory 4x → SRAM 초과), 나머지 128x128
   - H100: 256KB SRAM이 충분 → 전부 128x128 (물리적으로 정상)
   - 500~700ms: 엔트로피 계산이 병목 (최적화 기회 있음)
3. **적응적 penalty threshold** (v2.4): `penalty_threshold="auto"` → 첫 전파 제거율 기반 자동 보정
4. **DAG 그래프 전파** (v2.3): BFS 기반 양방향 전파, `LayerNode.neighbors`로 그래프 구조 표현
5. **전환 비용 가중치 자동 파생** (v2.2): `weight = clamp(ref_tile_bytes / sram_bytes, 0.01, 0.5)`
6. **SRAM 크기에 따른 차별화**: 12KB에서 working memory hard constraint가 Softmax 32x32를 물리적으로 제거하여 진정한 차별화 유발
7. **Exhaustive baseline 비교**: truncated top-8 baseline과 WFC 동일 결과. Exact DP 대비 100% (stress_gpu, working memory hard constraint 적용 후). WFC와 DP가 동일한 결과 산출
8. **Triton GPU 검증** (v2.7): RTX 3060에서 codegen 커널 4종 correctness 통과
9. **Autotuner 비교 재감사** (v2.7.1): 기존의 `평균 90%` 고정 수치는 폐기

## Triton Autotuner 비교 현황 (v2.7.1)
| 항목 | 현재 판단 |
|------|----------|
| correctness | `.venv` + RTX 3060에서 PASS |
| same-process rerun | 평균 비율이 `118%`와 `84%`로 크게 흔들림 |
| isolated rerun (5회) | 평균 비율 `86.0% ~ 94.6%` |
| Small (256x512x256) | ~`0.01-0.02ms`로 너무 짧아 autotuner tile이 `32x64x32` / `32x32x32` 사이에서 흔들림 |
| Larger workloads | autotuner가 `64x64x32` 또는 `128x128x32`를 일관되게 고르며 현재 WFC보다 대체로 빠름 |
| root cause | WFC 쪽은 비결정적이지 않음. single-node MatMul unary score가 workload를 거의 구분하지 못하고, variance의 직접 원인은 Small microbenchmark 노이즈 |

관찰: 방향성은 보이지만, 단일 평균 TFLOPS 비율을 authoritative claim으로 올리기엔 아직 benchmark 방법론이 불안정하다.

## 완료된 작업 (Phase 1~4)

| Phase | Task | 완료 |
|-------|------|------|
| 1 | 전환 비용 가중치 자동 파생 | v2.2 |
| 1 | 2D 그래프 전파 (DAG 지원) | v2.3 |
| 1 | 적응적 penalty threshold | v2.4 |
| 2 | 실제 GPU 스펙 기반 테스트 | v2.5 |
| 2 | 대규모 모델 스케일링 | v2.5 |
| 2 | Conv2D 및 Depthwise Conv 지원 | v2.6 |
| 3 | 스케줄 결과 → 실행 코드 생성 | v2.6 |
| 3 | 시각화 모듈 | v2.6 |
| 4 | Triton 커널 실행 검증 | v2.7 |
| 4 | benchmark / objective 정합성 재설계 | v2.8 |
| 4 | Cost model ↔ GPU 실행 시간 상관관계 검증 | v2.9 |
| 4 | 문서 전면 재감사 및 정직성 확보 | v2.9 |

## 연구 결론 (v2.9)

이 프로젝트는 **WFC 알고리즘의 하드웨어 스케줄링 적용 가능성**을 통제된 소프트웨어 환경에서 검증하는 것을 목표로 했다.

**달성한 것:**
- WFC 알고리즘이 chain 구조에서 exact DP와 100% 동일한 결과를 산출 (stress_gpu, toy_gpu)
- Cost model 점수 순위가 실제 GPU 실행 시간 순위와 양의 상관관계 (평균 Spearman ρ = +0.52)
  - Softmax: 완벽한 순위 일치 (ρ = 1.0)
  - MatMul (대형): 양의 상관관계 (ρ = +0.72)
  - MatMul (소형): 측정 노이즈로 역전
- Hard constraint (SRAM, alignment, working memory)가 물리적으로 정확하게 동작
- 55억 조합 탐색 공간을 수 ms에 해결

**달성하지 못한 것:**
- 기존 방법(Triton autotuner, exact DP) 대비 명확한 우위는 없음
- Cost model의 MatMul roofline score 포화 → workload별 타일 추천 불가
- 가상 스펙(12KB SRAM) 기반이라 실제 GPU에서의 실용성은 미검증

## 미래 과제 (이 연구를 이어갈 경우)

### 높은 우선순위
- **Cost model 교정**: 실제 GPU 프로파일링 데이터로 cost model 가중치를 피팅. 현재 hand-tuned Roofline/affinity/cache의 가장 큰 약점. ρ = +0.52 → +0.8 이상을 목표로.
- **Workload 감응형 타일 선택**: roofline score 포화 문제 해결. 레지스터 압력, 점유율, L2 캐시 효과를 반영하는 workload-aware 모델.
- **최적화된 탐색 baseline**: naive Python exhaustive (~3500x speedup)은 공정한 비교가 아님. 아래 방법으로 baseline을 강화해야 의미 있는 speedup 주장 가능:
  - **Branch-and-bound**: cost model 상한을 활용한 가지치기. 최적해 보장하면서 탐색 공간 대폭 축소
  - **Beam search**: 레이어당 top-k 상태만 유지. DP보다 빠르고, greedy보다 정확할 가능성
  - **벡터화 DP**: 전환 비용 행렬을 NumPy/C++로 사전 계산. 현재 Python DP 대비 10-100x 가속 가능
  - **C++/CUDA 구현**: 현재 Python overhead가 지배적. 알고리즘 자체의 속도 이점을 보려면 구현 언어 통일 필요

### 중간 우선순위
- **성능 최적화**: 엔트로피 계산 캐싱 (현재 72-84L에서 500-700ms 병목)
- **Triton autotuner benchmark 안정화**: microbenchmark 노이즈 원인 해결, warmup/compile vs steady-state 분리
- **Multi-device 스케줄링**: 단일 디바이스 chain → pipeline/tensor parallelism

### 장기 과제
- **실제 컴파일러 통합**: TVM, XLA, Triton 백엔드 연결로 end-to-end 검증
- **인터랙티브 대시보드**: Streamlit/Jupyter widget (SRAM/threshold 슬라이더, collapse 애니메이션)
