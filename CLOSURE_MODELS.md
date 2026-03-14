# Closure Models

## 목적

- portfolio closure 주간에 사용할 canonical model set과 config path를 고정한다.
- 추가 탐색 없이 같은 질문에 답하는 비교 결과를 만들기 위한 기준 문서다.

## Canonical Model Set

1. `SASRec`
   - [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_sasrec.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_sasrec.json)
2. `SASRec + metadata`
   - [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_sasrec_meta.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_sasrec_meta.json)
3. `DIF-SR`
   - [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_difsr.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_difsr.json)
4. `DIF-SR + metadata`
   - [`/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_difsr_meta.json`](/Users/conan/projects/personalized-fashion-recommendation/hm_refactored/configs/config.closure_difsr_meta.json)

## 비교 원칙

- top-k는 모두 `20`
- seed는 모두 `42`
- 동일한 local userhash32 split 사용
- closure 주간에는 이 4개 config만 사용한다

## 주의사항

- `DIF-SR` baseline은 metadata-preprocessed split을 그대로 사용하되, `metadata_features=[]`로 metadata signal을 끈다.
- 이는 새 architecture가 아니라 동일한 backbone에서 metadata injection만 제거한 비교 설정이다.

## Preflight

- final run 전에는 반드시 [`/Users/conan/projects/personalized-fashion-recommendation/CLOSURE_PREFLIGHT.md`](/Users/conan/projects/personalized-fashion-recommendation/CLOSURE_PREFLIGHT.md)를 확인한다.
- 자동 점검:

```bash
python3 /Users/conan/projects/personalized-fashion-recommendation/scripts/check_closure_readiness.py
```
