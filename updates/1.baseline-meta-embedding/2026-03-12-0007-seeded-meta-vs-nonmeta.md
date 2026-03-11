# Seeded meta vs non-meta 비교

## 비교 조건

- 공통
  - `seed=42`
  - `model_type=Transformer`
  - `embed_size=32`
  - `train_epoch=3`
  - `sampler_type=Negative`
  - `target_week=27`
  - `recent_two_weeks=false`
- 차이
  - non-meta: `embed_metadata=false`, `batch_size=64`
  - meta: `embed_metadata=true`, `batch_size=32`

## 실행 명령

```bash
source .venv/bin/activate
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_transformer_hash_benchmark.json
python hm_refactored/train.py --config_path hm_refactored/configs/config.m1_local_meta_transformer_hash_benchmark.json
```

## 결과

### Non-meta

- 평가 유저: `164`
- start benchmark: `B_HR 0.0610`, `B_NDCG 0.0362`
- best epoch: `1`
- best benchmark: `B_HR 0.0671`, `B_NDCG 0.0374`
- best test: `T_HR 0.0061`, `T_MAP 0.0020`

### Meta

- 평가 유저: `1681`
- start benchmark: `B_HR 0.0065`, `B_NDCG 0.0041`
- best epoch: `1`
- best benchmark: `B_HR 0.0077`, `B_NDCG 0.0038`
- best test: `T_HR 0.0006`, `T_MAP 0.0003`

## 해석

- 같은 시드 기준에서도 현재 로컬 샘플과 하이퍼파라미터에서는 non-meta가 meta보다 명확히 우세하다.
- meta는 더 많은 유저와 아이템을 다루지만, 현재 설정만으로는 그 복잡도를 성능으로 전환하지 못하고 있다.
- 다음 meta 실험은 `lr`, `batch_size`, `seq_len`을 따로 탐색하지 않으면 개선 가능성이 낮다.

## 현재 판단

- 로컬 champion은 여전히 non-meta Transformer다.
