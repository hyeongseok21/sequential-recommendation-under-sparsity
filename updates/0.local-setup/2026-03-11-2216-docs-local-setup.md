# 2026-03-11 22:16 문서 및 로컬 실행 설정

## 포함된 변경

- 루트 `README.md`에 프로젝트 개요, 구조, 데이터 컬럼, 평가 흐름, 로컬 실행 주의사항을 정리했다.
- 로컬 의존성 설치를 위해 `requirements.txt`를 추가했다.
- `hm_refactored/configs/config.json`을 저장소 기준 로컬 경로와 로컬 MLflow 저장소를 사용하도록 수정했다.
- Apple Silicon과 로컬 환경에서 가볍게 점검할 수 있도록 `hm_refactored/configs/config.m1_smoke.json`을 추가했다.

## 커밋에서 제외한 변경

- `hm_refactored/train.py`의 런타임 변경은 별도 커밋으로 분리하기 위해 포함하지 않았다.
