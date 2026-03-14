# Closure Preflight

## 목적

- final run을 시작하기 전에 closure 비교가 공정하고 재실행 가능하도록 점검한다.
- closure 주간에는 실행보다 preflight가 먼저다.

## 필수 점검 항목

1. canonical model set이 고정됐는가
   - [`/Users/conan/projects/personalized-fashion-recommendation/CLOSURE_MODELS.md`](/Users/conan/projects/personalized-fashion-recommendation/CLOSURE_MODELS.md)
2. 4개 config가 모두 존재하는가
3. seed가 모두 `42`인가
4. `top_k`가 모두 `20`인가
5. raw split 조건이 같은가
   - `target_week`
   - `train_data_name`
   - `test_data_name`
   - `count_high`
   - `count_low`
   - `recent_two_weeks`
   - `remove_recent_bought`
6. `save_name`이 서로 충돌하지 않는가
7. checkpoint 저장 경로가 유효한가
8. dataset 원본/전처리 경로가 존재하는가
9. closure metric naming policy가 정리됐는가
   - `HR -> Recall@20`
   - `NDCG -> NDCG@20`
   - `MRR@20`은 별도 계산 필요
10. slice 정의가 고정됐는가
   - `sparse-history user`
   - `multi-interest user`

## 주의사항

- `SASRec`과 `SASRec + metadata`는 전처리 save name이 다를 수 있다.
- 이 경우에도 raw split 파라미터가 같으면 closure 비교는 허용한다.
- `DIF-SR` baseline은 metadata-preprocessed split을 그대로 쓰되 metadata injection을 끈 설정이다.

## 실행 명령

```bash
python3 scripts/check_closure_readiness.py
```

## PASS 기준

- 에러 없음
- 비교 핵심 파라미터 일치
- 출력 경로 충돌 없음
- canonical model set 4개가 모두 ready 상태
