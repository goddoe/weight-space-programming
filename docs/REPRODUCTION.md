# Weight-Space Programming: independent reproductions

Code and evidence for asking `gpt-6-astra` to write or edit small neural-network weights directly, with no model-side tools and no gradient-based fitting during the edits. The original car network was previously trained with PPO. This reproduction does not retrain it.

## Results

| Task / policy | Evaluation | Result |
|---|---|---|
| Fresh Breakout policy | 1,000 episodes | 416.797 mean score, 6.901 standard error |
| Released upstream Breakout policy, local evaluator | 1,000 episodes | 502.880 mean score, 7.747 standard error |
| Rounded original car policy | 1,024 test episodes | 0 successes; all timeouts |
| Car attempt 1 | 256 search / 1,024 test | 0 / 0 successes |
| Car attempt 2 | 256 search / 1,024 test | 0 / 0 successes |
| Car attempt 3 | 256 search / 1,024 test | 256 / 1,024 successes |
| Car attempt 4 | 256 search / 1,024 test | 0 / 0 successes |

All four car attempts are independent, use the same prompt, and start from the same rounded weights. Previous answers and evaluation results are never fed back to Astra. One of four generated policies succeeded; this small observation is not a calibrated estimate of the model's general success probability. Detailed counts and usage are in `results/car_four_attempts.csv`.

## Sources and attribution

Original work: **Sparks of Weight Space Programming: LLMs can understand and write neural network weights**, by the authors of the linked upstream repository:

- Source: https://github.com/RikiyaT/lm_understand-logs
- Commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`
- Original car experiment: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car
- Original Breakout experiment: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch

Copied unchanged: the original car prompt (also copied into each car run folder), original tools-off config, and upstream license notices. The exact Breakout prompt contains public author email addresses in its MinAtar copyright header. To satisfy the no-email export requirement without altering that notice, it is not bundled. Run `games/scripts/fetch_breakout_prompt.py` to retrieve the original at the fixed commit and verify its recorded SHA256. This fetch happens outside Astra; Astra still receives no network or tools. The small `lm_understand-logs/` directory here is an attributed extract, not a full clone. No upstream model-weight archive is bundled. The car prompt itself contains the rounded weight values required for reproduction.

Upstream declares logs/data/networks CC BY 4.0, its own Python/shell scripts MIT, and MinAtar source quoted in the Breakout prompt GPL-3.0. Preserve `third_party/LICENSE-lm_understand-logs` and `third_party/LICENSE-MinAtar`; the fetched Breakout prompt contains the GPL-covered source unchanged, including the original copyright notice. This export does not relicense upstream material. See https://creativecommons.org/licenses/by/4.0/legalcode and the included GPL text.

Newly authored here: `games/scripts/*.py`, evaluator results, fresh Astra final answers, documentation, summaries and render code. The edit parser and evaluators were written for this reproduction, not copied from upstream's Python files. `FILE_ORIGINS.json` inventories the files. This export assigns no new blanket license to these new files; the destination repository maintainer should apply the repository's chosen license without overriding third-party notices.

Public-export changes from the scripts used locally: removed automatic emma reporting, replaced platform-specific fonts with Pillow's bundled default, added creation of the isolated Codex config, and replaced absolute workspace paths in saved command metadata with `${REPO_ROOT}`. Evaluation equations and model prompts are unchanged. Media can therefore differ typographically from the original rendered videos. Private reasoning/event text, credentials, host identifiers, environments and caches are excluded.

## Install

Recorded environment: Python **3.12.10**, Codex CLI **0.167.0**, JAX/JAXlib **0.9.0** (CPU), Kinetix **3.0.2** at commit below, jax2d **1.0.1**, MinAtar **1.0.15**, NumPy **2.5.4**, Pillow **13.0.0**. Exact resolved Python packages are in `requirements-resolved.txt`. Install an FFmpeg executable on PATH for MP4 rendering. These are historical versions, not a promise that every package/model is available to every account.

```sh
uv venv --python 3.12.10 .venv-games
git clone https://github.com/FLAIROx/Kinetix.git games/Kinetix
git -C games/Kinetix checkout 80ee9c292837c825cafaf9889323fdcc212efd1b
UV_CACHE_DIR=.cache/uv uv pip install --python .venv-games/bin/python -r requirements-resolved.txt
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

## Reproduce scores from recorded answers (no LLM needed)

Use new output folders to preserve the published measurements:

```sh
.venv-games/bin/python games/scripts/breakout.py --answer games/runs/breakout/answer.md --episodes 1000 --out games/recheck/breakout --video games/recheck/breakout.mp4
.venv-games/bin/python games/scripts/car.py --base --episodes 1024 --seed 100000 --out games/recheck/car_base
.venv-games/bin/python games/scripts/car.py --base --episodes 1 --seed 0 --out games/recheck/car_base_video --trajectory
.venv-games/bin/python games/scripts/car.py --answer games/runs/car_repeat2/answer.md --episodes 256 --seed 0 --out games/recheck/car3_search --trajectory
.venv-games/bin/python games/scripts/car.py --weights games/recheck/car3_search/actor.npz --episodes 1024 --seed 100000 --out games/recheck/car3_test
.venv-games/bin/python games/scripts/car_video.py --before games/recheck/car_base_video/trajectory.npz --after games/recheck/car3_search/trajectory.npz --out games/recheck/car3_before_after.mp4 --after-label 'AFTER: fresh Astra attempt 3'
```

Attempt 1 is `games/runs/car`, attempts 2, 3, 4 are `car_repeat1`, `car_repeat2`, `car_repeat3`. Substitute the answer path to evaluate another policy. The public export uses these physical folder names; local `car_rN` aliases are not required. No `.npz` files are bundled; applying the recorded edits regenerates them.

## Request a new independent Astra answer (requires model access)

Authenticate with your own account in a local ignored Codex home, then invoke a unique run name. Never commit that directory:

```sh
.venv-games/bin/python games/scripts/fetch_breakout_prompt.py
CODEX_HOME="$PWD/.cache/codex-games" codex login
.venv-games/bin/python games/scripts/ask_astra.py car --run-name car_new1
# Or: .venv-games/bin/python games/scripts/ask_astra.py breakout --run-name breakout_new1
```

The runner copies `config/codex_config.toml` into the isolated home with its instructions path expanded. `--strict-config` plus the original tools-off settings disables shell, web, apps and other built-in capabilities, regardless of the parent session's permissions. It refuses to overwrite an existing invocation. The output log is audited for unexpected item types. Inspect `run.json`: `status` must be `complete`, `tool_free` true, and there must be one `agent_message` before using the result. A disabled Code Mode host startup notice is expected. A new response need not match the published one.

Breakout uses effort xhigh and detailed summaries; car uses high without a summary override. Both use an 872,000-token context setting and an auto-compact limit of 1,000,000. Prompts are submitted byte-for-byte via stdin. All usage figures count reasoning tokens as part of output tokens, not in addition.

## Evaluation details and limitations

Breakout: zero-initialized 600→32→32→6 tanh actor, 20,486 parameters, sampled softmax, sticky actions 0.1, 10,000-step cap. Environment seeds 0 through 999; independent action-generator seeds 1,000,000 through 1,000,999. Batch size 32 and float32 inference. Single-episode video inference may differ numerically from batched inference, so the video is an independent rollout of the same weights, not a replay of the stored evaluation episode.

Car: original prompt's rounded 679→128 ReLU embedding, five tanh layers, 16 logits. Weights are parsed as printed and converted to float32 for JAX inference. Kinetix `l/h2_car_ramp`, frame skip 2, maximum 256 steps, dense reward scale 0.2, six independently sampled categorical groups. Search seeds 0 through 255, test seeds 100000 through 101023, shared across policies. Nonfinite observations become zero; observations are clipped to [-10, 10]. A successful episode touches the goal, not merely passes the ramp.

The upstream exact random streams and dependency lock were not published; our deterministic streams and pinned Kinetix/JAX versions are a documented reconstruction. As evaluator checks, released upstream Breakout weights scored 502.880 locally, and released upstream car edits succeeded in 32/32 local smoke episodes. Those are separate from upstream's reported 517.4 Breakout and 1,024/1,024 car results. The original edit answers also reproduced their released edited arrays exactly in parser checks.

The runner/evaluators stop if free disk space falls below 5 GiB. Video rendering draws actual simulator state and freezes completed car episodes; it does not synthesize successful movement. The original media handoff is preserved as `results/HANDOFF.md`; filenames for external media are descriptions, not claims that MP4s are bundled in this source archive.
