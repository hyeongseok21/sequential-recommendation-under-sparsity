# Automation Drafts

- 이 폴더는 멀티 에이전트 운영을 위한 역할별 automation 초안을 둔다.
- schedule은 여기서 정의하지 않고, prompt 본문만 관리한다.
- 실제 자동화 등록 시에는 이 초안을 기반으로 workspace와 실행 시점을 별도로 붙인다.

## 포함된 초안

- `analyst.md`
- `operator.md`
- `governor.md`

## 공통 규칙

- `AGENT.md`, `RUNBOOK.md`, `SELF_EVOLUTION_LOOP.md`, `protocol.md`를 먼저 읽는다.
- 현재 champion은 `hm_refactored/configs/config.m1_local_meta_difsr_bs16_seq30_do01_concat_lr2e4_hms15.json`으로 본다.
- 1순위 metric은 `B_NDCG`다.
- update log와 metric artifact를 반드시 남긴다.
