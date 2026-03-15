# Framework Pivot To Serving

## 가설

- 실험 목적이 benchmark score 최적화에서 실제 추천 운영 가정으로 이동했으므로, phase/agent/skill/loop/protocol도 serving-oriented 의사결정 기준에 맞게 재정렬해야 한다.

## 비교 조건

### Baseline

- benchmark 중심 framework
- `P2/P3` 중심
- `B_NDCG` 우선
- champion 단일 관점

### Treatment

- serving-oriented framework
- `P4` 중심
- `research champion` / `serving companion` 분리
- `dual-best`, `T_MAP`, `T_HR`, slice-aware 해석 포함

### 고정 조건

- current research champion 유지
- 기존 baseline-vs-treatment 원칙 유지

## 결과

### 변경 파일

- `AGENT.md`
- `SELF_EVOLUTION_LOOP.md`
- `protocol.md`
- `EXPERIMENT_PHASES.md`
- `agents/orchestration.md`
- `skills/recsys-self-evolution-runbook/SKILL.md`

### 핵심 반영

- 현재 active phase를 `P4` 중심으로 재해석
- `decision_mode`를 `research / serving / hybrid`로 분리
- `dual-best`와 `serving companion` 개념을 protocol과 skill에 반영
- axis taxonomy에 `serving-proxy`, `retrieval`, `slice-analysis` 추가
- loop의 우선순위를 benchmark 튜닝보다 serving-safe decision으로 이동

## 해석

- 현재 실험 결과는 fast-scout PASS 후 full FAIL이 반복되고 있고, benchmark-best와 test-best도 자주 갈린다.
- 따라서 framework도 이제는 “무엇이 benchmark champion인가”보다 “실제 추천 운영에서 무엇을 함께 들고 갈 것인가”를 설명할 수 있어야 한다.

## 현재 판단

- gate verdict: `PASS`
- research champion:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15_all_features_product_type15.json`
- serving companion:
  - current dual-best report의 `test-best` checkpoint

## 다음 후보

1. serving note / companion summary artifact 추가
