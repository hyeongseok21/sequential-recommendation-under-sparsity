# Checkpoint Selection Policy 정렬

## 가설

- checkpoint 선택 기준을 `B_HR` 우선에서 runbook 기준인 `B_NDCG` 우선으로 바꾸면, offline selection quality를 더 일관되게 맞출 수 있다.

## 변경 내용

- [`hm_refactored/train.py`](../../hm_refactored/train.py)
  - `checkpoint_primary_metric`, `checkpoint_secondary_metric` 지원 추가
  - 기본 selection policy를 `benchmark_ndcg -> benchmark_hr`로 정렬
  - checkpoint metadata를 `best_primary`, `best_secondary`로 저장하고, 기존 `best_hr`, `best_ndcg`도 backward compatibility용으로 유지
  - final log 문구를 실제 metric label 기준으로 출력하도록 수정
- current champion config에 policy 명시
  - [`hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`](../../hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json)
  - [`hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`](../../hm_refactored/configs/config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json)
- offline 비교 스크립트 추가
  - [`scripts/report_checkpoint_policy.py`](../../scripts/report_checkpoint_policy.py)

## Offline 비교 결과

### Historical divergence case

- artifact: `hm_m1_local_meta_transformer_hash_benchmark_bs16_seq30_l2/loss.txt`
- `B_HR` 우선 선택:
  - epoch `2`
  - `B_HR 0.0054`
  - `B_NDCG 0.0021`
- `B_NDCG` 우선 선택:
  - epoch `1`
  - `B_HR 0.0048`
  - `B_NDCG 0.0022`

해석:

- `B_NDCG` 우선 selection이 primary metric을 `+0.0001` 개선했다.
- `B_HR`는 `-0.0006` 내려갔지만, runbook 기준상 새 정책이 더 맞다.

### Current champion safety check

- artifact: `hm_m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15/loss.txt`
- `B_HR` 우선과 `B_NDCG` 우선 모두 epoch `1` 선택
- 현재 champion에는 selection epoch 변화가 없었다.

## 코드 검증

- fast verification run:
  - config: `config.m1_local_meta_difsr_bs64_seq30_do01_concat_fast_lr2e4_hms15.json`
  - final log:
    - `Best epoch 000: BENCHMARK_NDCG = 0.0076, BENCHMARK_HR = 0.0178`

해석:

- selection policy와 final log label이 일관되게 맞춰졌다.
- 현재 DIF-SR champion의 behavior는 유지되고, checkpoint policy만 runbook에 맞게 정렬됐다.

## 현재 판단

- gate verdict: PASS on `P4`
- current champion 유지:
  - `concat DIF-SR + lr=2e-4 + history_meta_scale=1.5`
- new policy:
  - `checkpoint_primary_metric=benchmark_ndcg`
  - `checkpoint_secondary_metric=benchmark_hr`

## 다음 후보

1. `P4`에서는 benchmark와 test 괴리를 줄이기 위해 checkpoint 선택 외에 `best-epoch report`를 두 기준으로 같이 남기는 쪽을 본다.
2. `P3`로 돌아간다면, 구조보다 `evaluation-gap` 원인 분석이 다음 우선순위다.
