# Service-Style Evaluation

## Goal

Keep the canonical `userhash32` experiment unchanged and add a supplementary robustness analysis under more realistic serving conditions.

Relaxed filters:
- `remove_cold_user = false`
- `remove_zero_history = false`
- `remove_recent_bought = false`
- `remove_cold_item = true`

## Added Artifacts

- service preprocessing entry:
  - [`hm_refactored/hm_preprocess_service_eval.py`](../../hm_refactored/hm_preprocess_service_eval.py)
- service dataset:
  - [`datasets/hm/local/prep/hm_local_meta_userhash32_week27_service_eval.pkl`](../../datasets/hm/local/prep/hm_local_meta_userhash32_week27_service_eval.pkl)
- service configs:
  - [`hm_refactored/configs/config.service_eval_sasrec.json`](../../hm_refactored/configs/config.service_eval_sasrec.json)
  - [`hm_refactored/configs/config.service_eval_difsr.json`](../../hm_refactored/configs/config.service_eval_difsr.json)
  - [`hm_refactored/configs/config.service_eval_difsr_meta.json`](../../hm_refactored/configs/config.service_eval_difsr_meta.json)
- supplementary report:
  - [`results/service_style_eval.md`](../../results/service_style_eval.md)

## Service-Style Results

Overall:
- `TopPopular`: `Recall@20 0.0293`, `NDCG@20 0.0111`, `MRR@20 0.0061`
- `Corrected SASRec`: `Recall@20 0.0147`, `NDCG@20 0.0099`, `MRR@20 0.0085`
- `DIF-SR`: `Recall@20 0.0169`, `NDCG@20 0.0086`, `MRR@20 0.0063`
- `DIF-SR + Metadata`: `Recall@20 0.0202`, `NDCG@20 0.0093`, `MRR@20 0.0063`

Slice highlights:
- `cold-like users`:
  - `DIF-SR + Metadata` best with `Recall@20 0.0233`, `NDCG@20 0.0116`
- `short history users (<=5)`:
  - `DIF-SR + Metadata` best among personalized models with `Recall@20 0.0239`, `NDCG@20 0.0107`
- `repeat purchase cases`:
  - `Corrected SASRec` strongest with `Recall@20 0.3194`, `NDCG@20 0.2640`

## Interpretation

- Relaxing the filters did not uniformly degrade the personalized models; allowing repeat purchases made the task easier in aggregate.
- `DIF-SR + Metadata` remained the strongest personalized model overall under service-style conditions.
- Metadata helped most clearly for `cold-like` and `short history` users.
- Repeat-purchase cases behaved differently from the canonical benchmark and favored `Corrected SASRec` more than diffusion models.
- The canonical benchmark remains the primary artifact; this service-style result is a supplementary robustness analysis.
