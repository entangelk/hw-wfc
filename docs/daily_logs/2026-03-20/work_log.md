# Work Log — 2026-03-20

## 목표
코드 정확성, 검증 타당성, 비교 공정성을 전면 감사하고 발견된 문제를 즉시 수정한다.

## 완료한 작업

### 1. 전면 감사: 코드 정확성

| 항목 | 발견 | 심각도 | 조치 |
|------|------|--------|------|
| Hard Constraint: working memory 미검사 | `check_sram_capacity`가 단일 타일만 검사, working memory (tile × buffer count) 미확인 | 중간 (설계 결정) | 문서에 명시. Task 11에서 hard constraint 전환 여부 결정 예정 |
| 전파 경계조건 `<` vs `<=` | `propagate_constraints`에서 `<` 사용 → penalty == threshold인 후보 제거됨. 문서는 "초과" | 낮음 | `<=`로 수정 |
| 백트래킹: 불필요한 재붕괴 | 모든 노드의 `collapsed_state`를 None으로 리셋 → 이미 확정된 노드도 재붕괴 | 낮음 | 결과는 동일하므로 문서에 기록만 |

### 2. 전면 감사: 검증(Safety Check) 타당성

| 항목 | 발견 | 조치 |
|------|------|------|
| Check #2 "54개 완전 구분" 주장 | 거짓. 같은 면적 직사각 타일끼리 동점 (toy_gpu: 21개 동점, stress_gpu: 12개 동점) | HANDOFF.md 수정 + check에 동점 그룹 카운트 추가 |
| Check #2 toy_gpu 전용 | stress_gpu에서의 변별력 미검증 | stress_gpu 검사 추가 (spread=0.72, 30단계 구분) |
| Check #5b "다르면 통과" | 차별화가 올바른 방향인지 미검증 | **Exact DP 비교 추가** → 핵심 발견 도출 |

### 3. **핵심 발견: WFC 차별화 = greedy 부산물**

Safety Check #5b에 exact DP 비교를 추가한 결과:
- **WFC**: Softmax=16x16, 나머지=32x32 (score: 2.8494)
- **Exact DP**: 전부 32x32 (score: 2.8996)

DP는 "복사기" 해(all-same)가 최적이라고 판단. 이유:
- Softmax 32x32의 cache penalty: -0.3
- 32→16→32 왕복 전환 비용: ~0.4 (tile_shape × weight × 2)
- 전환 비용이 cache penalty보다 큼 → all-same이 더 좋음

따라서 WFC의 "복사기 문제 해결"은 **품질을 1.7% 떨어뜨리는 greedy 부산물**이다.

### 4. 전면 감사: 비교 공정성

| 항목 | 발견 | 조치 |
|------|------|------|
| "~2400x speedup" | naive exhaustive(Python 순회) 대비. DP는 ~5x만 느리고 더 좋은 답 | README Key Results를 DP 비교 중심으로 재구성 |
| "396→6 (98.5%)" | 후보의 합(sum)이지 조합 공간이 아님. 실제 조합 공간은 42^6 ≈ 55억 | README에 "candidate-slot reduction, not combinatorial space" 명시 |
| DP baseline 구현 | 정확한 Viterbi DP, 같은 objective/weight/candidate set 사용 | 공정 ✓ |

### 5. 문서 동기화

| 파일 | 변경 |
|------|------|
| README.md / .ko.md | Key Results 재구성 (DP 주축), 복사기 문제 섹션 재서술, speedup 표현 수정 |
| architecture_guide.md / .ko.md | §8 한계 전면 재작성 (outdated 3건 제거, greedy/working-memory/tile-symmetry 추가), Check #5 caveat 추가 |
| concept.md / .ko.md | §3 시간 ~2ms→~2-3ms, §5 로드맵 Phase 1-4 완료 표시 |
| HANDOFF.md | "54개 완전 구분" → 동점 그룹 존재 명시, exhaustive baseline 설명 수정, Check #5 주의 |
| constraint.py | 전파 경계조건 `<` → `<=` |
| test_safety_checks.py | Check #2 stress_gpu 추가 + 동점 카운트, Check #5b DP 검증 추가 |

### 6. Safety Check 변경 전후 비교

| Check | 이전 | 이후 |
|-------|------|------|
| #2 | toy_gpu만, "완전 구분" 주장 | toy_gpu + stress_gpu, 동점 21/12개 보고 |
| #3 | threshold=0.5에서 94% 제거 | `<=` 적용 후 93% 제거 (BLOCK_TILED 1개 추가 생존) |
| #5b | "다르면 통과" | DP 비교 추가 → **DP는 all-same 선택, WFC 차별화가 최적이 아님을 보고** |

## 기술 결정

1. `propagate_constraints`의 `<`를 `<=`로 변경: penalty == threshold인 후보는 경계선에 있으므로 유지하는 것이 맞음
2. 백트래킹 불필요 재붕괴는 현재 규모에서 문제 없으므로 수정하지 않음
3. working memory hard constraint 도입은 Task 11에서 결정 (현재는 문서 정정만)

### 7. Working Memory Hard Constraint 도입

감사에서 발견된 핵심 문제(working memory 미검사)를 즉시 수정:

| 항목 | 변경 |
|------|------|
| `state.py` | `working_memory_multiplier()` 함수 추가 (cost_model.py에서 이동, 순환 의존성 방지) |
| `constraint.py` | `check_working_memory()` 함수 추가, `apply_hard_constraints()`에 통합 |
| `cost_model.py` | `working_memory_multiplier` import를 state.py에서 가져오도록 변경 |

**효과 — 이전 vs 이후:**

| 지표 | 이전 (soft penalty) | 이후 (hard constraint) |
|------|---------------------|------------------------|
| Hard constraint 제거 수 | 96 | **123** (+27) |
| Softmax 후보 수 (stress_gpu) | 42 | **33** (-9) |
| WFC vs Exact DP 품질 | 98.3% | **100%** |
| WFC Softmax 선택 | 16x16 (greedy 부산물) | 16x16 (물리적으로 유일한 최적해) |
| DP Softmax 선택 | 32x32 (all-same 최적) | 16x16 (WFC와 동일) |

**핵심**: Softmax 32x32 SRAM의 working memory (4x × 4KB = 16KB)가 12KB SRAM을 초과하므로 hard constraint에 의해 물리적으로 제거됨.
이로써 WFC의 차별화가 "greedy 부산물"이 아닌 "물리적 제약에 의한 진정한 차별화"가 됨.

### 8. 문서 동기화 (working memory hard constraint 반영)

| 파일 | 변경 |
|------|------|
| README.md / .ko.md | 98.3% → 100%, "greedy 부산물" 서술 제거, Hard Constraint에 working memory 추가, 비교 테이블 업데이트 |
| architecture_guide.md / .ko.md | §2.3 stress_gpu 설명, §6 Check #5 caveat 제거, §7 walkthrough 숫자, §8 working memory 한계 제거 |
| HANDOFF.md | Check #5 "주의" → "통과", 품질 98.3% → 100%, Task 11 완료 |
| CHANGELOG.md | v2.8 항목 업데이트 |

## 다음 단계
- Task 10: 성능 최적화 (엔트로피 캐싱, 500-700ms 병목)
- Task 12: 기존 결과 전면 재감사 (daily logs, visuals caption 등)
- Task 13: Triton autotuner benchmark 안정화
