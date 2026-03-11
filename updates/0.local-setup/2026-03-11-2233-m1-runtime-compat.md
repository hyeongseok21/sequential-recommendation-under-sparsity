# 2026-03-11 22:33 M1 런타임 호환성

## 포함된 변경

- `hm_refactored/train.py`에서 `cuda`, `mps`, `cpu`를 자동으로 선택하도록 장치 선택 로직을 추가했다.
- MPS 호환성을 높이기 위해 텐서 생성 방식을 `torch.as_tensor(..., device=...)`로 정리했다.
- `mlflow`가 설치되지 않은 환경에서도 학습이 계속되도록 fallback 처리를 추가했다.
- `benchmark=false`일 때 학습 종료 단계에서 깨지던 로직을 수정했다.
- `pandas 2.x` 호환성을 위해 전처리 코드의 `Series.append()`를 `pandas.concat()`으로 교체했다.
- `config.m1_smoke.json`이 smoke 데이터셋을 사용하고 benchmark를 비활성화하도록 수정했다.

## 커밋에서 제외한 변경

- `.venv`, `mlruns/`, 샘플 CSV, 전처리 pickle 같은 생성물은 포함하지 않았다.
- Finder 메타데이터 파일인 `.DS_Store`도 포함하지 않았다.
