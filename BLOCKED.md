# BLOCKED.md

## 1. ~~CI workflow not at .github/workflows~~ RESOLVED 2026-11-23: token given workflow scope, file moved to `.github/workflows/ci.yml`
- **What:** GitHub rejects pushes that add files under `.github/workflows/` when the token has no `workflow` scope,
  and the build host's token has only `repo` scope.
- **What I did:** the workflow is committed at `ci/github-workflows/ci.yml` (runs `pytest` with mocked LLM, no secrets).
- **Next step for a human (one command, from a machine with a normal git login):**
  `git mv ci/github-workflows .github/workflows && git commit -m "Enable CI" && git push`

## 2. No GPU on the build host, so the full training was not run (by design, per the plan)
- **What:** `nvidia-smi` is not present, so only the CPU smoke training ran (150 steps, Qwen2.5-0.5B, 88 min).
- **Next step for a human:** run the Kaggle kernel (`bash kaggle/push.sh`, full steps in `docs/kaggle.md`), unzip
  `adapter.zip` into `models/`, then `make merge serve eval-ft eval-ood` (TESTING.md steps 6 and 7).

## 4. No Kaggle credentials on the build host (checked 2026-11-24)
- **What:** `~/.kaggle/kaggle.json` does not exist, `KAGGLE_USERNAME` is not in `~/overnight-build/.env`, and the
  Kaggle CLI is not installed, so the kernel was prepared but not pushed. (It would run on the owner's account
  anyway, so it should be started by the owner.)
- **Next step for a human:** the one-time setup in `docs/kaggle.md` (phone verification, API token, `pip install
  kaggle`), then `bash kaggle/push.sh`.

## 3. M5 skipped: `~/inputs/multi-invest-committee` was not provided
- Not a failure. The plan makes M5 conditional on that folder existing.
