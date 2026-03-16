# Work Log — 2026-03-16

## 목표
Phase 4 Task 9: Triton 커널 실행 검증 — RTX 3060에서 codegen 출력의 correctness 검증 및 autotuner 비교

## 완료한 작업

### 1. 개발 환경 구성

| 항목 | 내용 |
|------|------|
| 가상환경 | `.venv/` (venv, Python 3.12) |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| GPU | NVIDIA GeForce RTX 3060 (WSL2) |
| requirements.txt | `torch`, `triton` (최소 기재, freeze 미사용) |

### 2. RTX 3060 GPU 스펙 추가

| 파일 | 변경 내용 |
|------|----------|
| `specs/rtx3060.json` | 신규: SRAM 100KB, GDDR6 360GB/s, alignment 128B, 12.74 TFLOPS |

### 3. Triton 커널 Correctness 검증

| 파일 | 변경 내용 |
|------|----------|
| `examples/triton_verify.py` | 신규: Linear/ReLU/Softmax 커널의 GPU 실행 correctness 검증 |

**스케줄 결과 (RTX 3060, 100KB SRAM)**:
- Linear_0: 64x128 (warps=8, stages=2)
- ReLU_1: 64x128 (warps=8, stages=2)
- Softmax_2: 32x128 (warps=4, stages=3) ← SRAM 제약으로 차별화
- Linear_3: 64x128 (warps=8, stages=2)

**검증 결과**:
| 커널 | 타입 | 결과 | 오차 |
|------|------|------|------|
| Linear_0 | MatMul | PASS | rel_err=0.24% |
| ReLU_1 | Element-wise | PASS | max_diff=0.0 (정확 일치) |
| Softmax_2 | Row-parallel | PASS | max_diff=7.45e-9 |
| Linear_3 | MatMul | PASS | rel_err=0.17% |

### 4. HW-WFC vs Triton Autotuner 성능 비교

| 파일 | 변경 내용 |
|------|----------|
| `examples/triton_autotuner_compare.py` | 신규: 5개 워크로드에서 WFC vs autotuner 벤치마크 |

**결과**:
| 워크로드 | WFC 타일 | Autotuner 타일 | WFC TFLOPS | Auto TFLOPS | 비율 |
|----------|----------|---------------|-----------|------------|------|
| Small (256x512x256) | 64x128x64 | 32x64x32 | 5.08 | 7.17 | 70.8% |
| Medium (1024x1024x1024) | 64x128x64 | 64x64x32 | 16.04 | 20.17 | 79.5% |
| Large (2048x2048x2048) | 64x128x64 | 128x128x32 | 19.31 | 20.82 | 92.7% |
| GPT-2 FFN (768x3072x768) | 64x128x64 | 64x64x32 | 18.67 | 20.26 | 92.2% |
| BERT QKV (512x768x768) | 64x128x64 | 64x64x32 | 15.41 | 13.42 | 114.9% |

**평균: 90.0%** (autotuner 대비)

### 핵심 발견

1. **WFC는 워크로드 무관 정적 결정** — 모든 워크로드에 64x128x64 선택 (하드웨어 제약 기반)
2. **Autotuner는 워크로드별 동적 탐색** — 각 워크로드에 다른 타일 선택 (실측 기반)
3. **대규모에서 거의 동등** — 2048x2048에서 92.7%, BERT QKV에서 114.9% (WFC 승)
4. **소규모에서 차이** — 256x512에서 70.8% (작은 행렬에서 autotuner 유리)
5. **WFC의 강점: 속도** — 제약 전파 ~1ms vs autotuner 수십 회 커널 실행

## 기술적 결정
- `requirements.txt`는 `pip freeze` 대신 최소 패키지만 기재 (공통 로컬 환경)
- RTX 3060 스펙: SM당 shared memory 100KB (공식 스펙), GDDR6 360GB/s

## 다음 단계
- Task 10: 성능 최적화 (엔트로피 계산 캐싱)
- 워크로드별 타일 선택을 위한 cost model 개선 검토 (autotuner 격차 줄이기)
