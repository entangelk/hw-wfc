<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-6B7280?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-111111?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# HW-WFC

**Hardware Wave Function Collapse** — 제약 기반 붕괴적 탐색을 활용한 AI 하드웨어 컴파일러 자동 스케줄링

WFC(Wave Function Collapse) 알고리즘을 AI 하드웨어 컴파일러의 자동 스케줄링에 적용한 연구 프로토타입입니다. 각 레이어의 최적 타일 크기, 메모리 레이아웃, 연산 위치를 제약 기반 탐색으로 결정합니다.

> **처음 오신 분은** 알고리즘 컨셉과 동기를 설명하는 **[컨셉 문서](docs/concept.ko.md)**와 설계 의도 해설서인 **[아키텍처 가이드](docs/architecture_guide.ko.md)**를 먼저 읽어보세요.

## 동작 방식

![HW-WFC Collapse Animation](assets/collapse_animation.gif)

1. **Superposition** — 각 레이어가 모든 가능한 HW 구현 상태(tile size × layout × location)를 후보로 시작
2. **Hard Constraints** — 물리적으로 불가능한 후보 제거 (SRAM 용량, alignment 위반)
3. **Bottleneck-First Collapse** — FLOPs가 가장 큰 레이어부터 최적 상태로 붕괴
4. **Constraint Propagation** — 붕괴된 노드에서 인접 노드로 전환 비용 기반 후보 제거
5. **Entropy-Ordered Collapse** — 남은 노드를 Shannon 엔트로피 순서로 붕괴
6. **Backtracking** — 모순 발생 시 snapshot 기반 상태 복원

## 핵심 결과

| 지표 | 값 |
|------|-----|
| 탐색 공간 축소 | 396 → 6 states (98.5%) |
| Grid Search 대비 품질 | 100% (전환 비용 포함) |
| 속도 | ~2.5ms (6-layer attention block) |
| Grid Search 동등 비용 | 262K 조합, 6000ms |
| 속도 향상 | **~2400배** |

### 레이어 타입별 차별화 (12KB SRAM)

```
QKV_Proj   → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
QK_MatMul  → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
Softmax    → 16x16  ROW_MAJOR  SRAM   (SOFTMAX, working memory 4x)  ← 다름!
AV_MatMul  → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
Out_Proj   → 32x32  ROW_MAJOR  SRAM   (LINEAR,  working memory 3x)
LayerNorm  → 32x32  ROW_MAJOR  SRAM   (LAYERNORM, working memory 3x)
```

SRAM이 tight할 때 working memory 차이로 인해 레이어 타입별 다른 최적 타일을 선택합니다.

## 시각화

모든 시각화는 `python tools/generate_all_visuals.py`로 재생성됩니다.
코드 변경 후 시각화를 업데이트하면 이전과의 차이를 시각적으로 확인할 수 있습니다.
기록은 [assets/VISUALS_LOG.md](assets/VISUALS_LOG.md)에 자동 추가됩니다.

### Cost Model 점수 분포

레이어 타입별 후보 점수를 Roofline(파랑) + Affinity(주황) + Cache(초록)로 분해. 빨간색은 cache 패널티.

![Score Distribution](assets/score_distribution.png)

### 제약 전파 Sweep

threshold를 0.3 ~ 3.0까지 변화시켰을 때 후보 제거율. 빨간 선(25%)이 통과 기준. 현재 t=1.5에서 43%.

![Propagation Sweep](assets/propagation_sweep.png)

### 복사기 문제 (실패 → 해결)

SRAM이 넉넉하면(64KB) 모든 레이어가 같은 상태로 수렴하는 "복사기 문제" 발생.
SRAM을 줄이면(12KB) working memory 차이로 Softmax가 다른 타일을 선택하여 해결.

![Copier Problem](assets/copier_problem.png)

### 백트래킹: 모순 & 복구

의도적으로 해결 불가능한 제약 조합을 구성하여 백트래킹이 실제 발동하는지 검증.
ROW → ? → COL 체인에서 threshold=0.5일 때 모순 발생 → 백트래킹 1회 발동 → 실패 확인.

![Backtracking](assets/backtracking.png)

### SRAM 크기별 영향

동일한 6-layer Attention Block을 64KB / 16KB / 12KB SRAM에서 실행.
12KB에서만 Softmax가 작은 타일(16x16)을 선택하여 차별화 발생.

![SRAM Comparison](assets/sram_comparison.png)

## 아키텍처

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  LayerNode  │────▶│ HardConstraint│────▶│  CostModel   │
│ (candidates)│     │  (SRAM, align)│     │ (roofline +  │
│             │     │               │     │  affinity +  │
│ HWState[]   │     └───────────────┘     │  cache)      │
└─────────────┘                           └──────┬───────┘
                                                 │
       ┌──────────────────────────────────────────┘
       ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│CollapseEngine│────▶│  Propagation  │────▶│  Scheduler   │
│ (entropy,    │     │ (layout +     │     │ (bottleneck  │
│  backtrack)  │     │  tile_shape + │     │  first,      │
│              │     │  location)    │     │  bidirect)   │
└──────────────┘     └───────────────┘     └──────────────┘
```

## 빠른 시작

```bash
# Transformer Attention Block 스케줄링
python examples/attention_block.py

# Safety Checks 실행
python tests/test_safety_checks.py

# 전체 시각화 생성/업데이트
python tools/generate_all_visuals.py

# Collapse GIF만 재생성
python tools/visualize_collapse.py
```

## 비용 모델

3요소 합산:

- **Roofline Score** — 타일링에 의한 데이터 재사용률 반영. MatMul은 타일이 클수록, Softmax는 tile_n이 클수록 유리
- **Layout Affinity** — 레이어 타입별 선호 레이아웃 (LINEAR→ROW_MAJOR, SOFTMAX→ROW_MAJOR, CONV2D→BLOCK_TILED)
- **Cache Efficiency** — SRAM 활용률 가우시안 함수 (75%에서 피크, working memory multiplier 적용)

## 하드웨어 스펙

| 스펙 | SRAM | 용도 |
|------|------|------|
| `toy_gpu.json` | 64KB | 편안한 환경, 기본 검증용 |
| `tight_gpu.json` | 16KB | 중간 제약 |
| `stress_gpu.json` | 12KB | tight SRAM, 레이어별 차별화 발생 |

## Safety Checks (5/5 통과)

| # | 검증 항목 | 상태 | 기준 |
|---|----------|------|------|
| 1 | Hard Constraint 정확성 | 통과 | false positive/negative 없음 |
| 2 | Cost Model 변별력 | 통과 | 점수 spread > 0.01 (실제: 1.19) |
| 3 | 전파 실효성 | 통과 | t=1.5에서 제거율 > 25% (실제: 43%) |
| 4 | 백트래킹 활성화 | 통과 | 모순 시 발동 (검증됨) |
| 5 | 복사기 문제 없음 | 통과 | 이질적 레이어가 다른 상태 선택 |

## 프로젝트 구조

```
hw-wfc/
├── src/                    # 코어 알고리즘
│   ├── state.py            # HWState, LayerNode, superposition
│   ├── constraint.py       # Hard/Soft 제약, 전파, 자동 가중치
│   ├── cost_model.py       # Roofline + affinity + cache
│   ├── collapse.py         # 엔트로피 기반 붕괴 + 백트래킹
│   └── scheduler.py        # 메인 파이프라인
├── specs/                  # 하드웨어 스펙 (JSON)
├── examples/               # PoC 스크립트
├── tests/                  # Safety Checks
├── tools/                  # 시각화 생성기
├── assets/                 # 생성된 이미지/GIF
├── docs/
│   ├── concept.md / .ko.md             # 알고리즘 컨셉 & 동기 (EN/KO)
│   ├── architecture_guide.md / .ko.md  # 설계 의도 해설서 (EN/KO)
│   └── daily_logs/                     # 작업 일지
└── HANDOFF.md              # 다음 작업 & 아이디어
```

## 요구사항

- Python 3.10+
- Pillow (시각화 전용)

## 라이선스

연구용 프로토타입. 프로덕션 사용 불가.
