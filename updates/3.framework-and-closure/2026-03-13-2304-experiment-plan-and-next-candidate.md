# 실험 계획과 다음 후보 정의

## 현재 상태
- full champion은 `all_features + DIF-SR + concat + lr=2e-4 + history_meta_scale=1.5 + h2`
- fast-scout에서는 `h4`가 유망하지만 full validation에선 runtime과 quality가 아직 불안정했다.

## 계획 정리
- `attention-capacity`
  - `all_features + h4 + batch_size 32`
  - `all_features + h4 + drop_out 0.05`
- `metadata-input`
  - feature-specific scale
- `architecture`
  - feature-specific history projection / weighting
- `evaluation-policy`
  - dual-best / direct checkpoint evaluation 유지

## 즉시 실행 후보
- hypothesis:
  - `all_features + h4`는 `batch_size 32`에서 runtime과 quality 균형이 개선될 수 있다.
- treatment config:
  - `hm_refactored/configs/config.m1_local_meta_difsr_bs32_seq30_do01_concat_lr2e4_hms15_all_features_h4.json`

## 기대 효과
- `bs16 + h4`에서 보였던 과도한 runtime penalty를 완화할 수 있다.
- `h4`가 단순 fast-scout illusion인지, 아니면 budget mismatch 문제였는지 분리해서 볼 수 있다.
