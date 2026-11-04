VERDICT: DEFECTS

Phase 6 (T6.1 to T6.6), reviewed on `build` at 01fa9d7. All defects below are lesser gaps. None puts a false number on record. Each is assigned to a later task.

What was run (no GPU, Ollama or Claude calls):
- `check_run E14 --min-n 200 --split rag_dev` printed "E14 ok". `check_run E7 --min-n 50 --split rag_dev --all` printed "E7 ok".
- tests/test_serving.py, test_answer.py, test_e14.py and test_retrieve.py through the one-test script: 91 passed, 0 skipped.
- No `rag_test` read found in any E7 or E14 record. E14 config: split `rag_dev`, source judgments sha256-pinned, and the source is the non-superseded record E7-20261104T040010301317Z.
- `serving.py` matches `results/e7/v2/chosen.json` and the DECISIONS entries. The top-level `results/e7/chosen.json` is the kept v1 record (dense, no rerank, base+faithful). It is not what is served, and nothing reads it.
- The E7 and E14 numbers in DECISIONS match the run records: E14 agreement 0.874, kappa 0.630 [0.576, 0.684], served 28 vs 1. README and RESUME-TARGETS carry no E7 or E14 number, so nothing overclaims there.
- The `superseded.json` keys exist. The current v2 records are not listed in it.

## Defects

### 1. A `NEED:` reply returns advice with no citation and no label note
- File: `src/sprout/rag/answer.py:210-214`.
- Repro, with `FakeLLM(["NEED: photo\nSpray bleach on the plant weekly and burn it."])` and a retriever that returns any passage. The result is `state='clarify'`, `grounded=False`, and `text='Spray bleach on the plant weekly and burn it.'`. There is no LABEL_NOTE, no UNGROUNDED prefix and no citation.
- Impact: this breaks the DESIGN rule that treatment advice has a source or a label, and the CLAUDE.md rule that chemical or disposal guidance says to check the label and local rules. The T6.4 verdict said clarify "has no advice", but `rest` is whatever text follows the `NEED:` line.
- How likely: the served prompt `base+faithful+abstain` does not mention `NEED:`, so it only happens when the model emits `NEED:` unprompted. The code still takes the branch unguarded.
- Owner: T7.4. When `rest` is non-empty, either drop it and use the generated "Could you tell me ..." question, or run it through the ungrounded path with the label.

### 2. The floors were tuned on torch/CUDA scores but serve on fastembed ONNX scores, and only ranks were compared
- Files: `src/sprout/rag/serving.py:17`, `src/sprout/rag/gpu_models.py`, `results/runs/E7-20261103T235630916335Z.rerank_cpu_onnx.predictions.jsonl`.
- The floor `hybrid+rerank` = 0.73875 is an absolute threshold on a sigmoid of the cross-encoder logit. E7 selected it with the HF torch reranker and torch embedder on CUDA. Serving uses the Xenova ONNX reranker and the fastembed embedder.
- Evidence of equivalence on record:
  - DECISIONS says recall@5 is equal (0.833) across the three paths.
  - The ONNX prediction rows hold only `{id, rank}`, so no score was compared.
  - Embedding cosine agreement (at least 0.99999) is recorded only for the embedder (DECISIONS lines 164 and 475). The reranker's logits were never compared across torch and ONNX.
- Impact: a small score shift near the floor changes which passages are dropped, so the served decline rates (1.000 / 1.000 / 0.462) may not hold on the real serving path. The same-model argument makes a large gap unlikely, but it is unmeasured. The "served" numbers came from the torch path.
- Owner: T7.4, which measures the whole path end to end. Compare per-passage reranker scores for ONNX versus torch on the 60 dev questions, and re-run the answer and decline rates on the ONNX path. This adds to the known k=10 vs k=5 floor mismatch: the rerank pool also differs (30 versus 15 candidates).

### 3. DECISIONS cites a gitignored file for the adjudication numbers
- File: `docs/DECISIONS.md` (T6.6 "Reading", in the section starting at line 616). It cites `orchestration/evidence/E14-adjudication.md`.
- Repro: `git check-ignore -v orchestration/evidence/E14-adjudication.md` reports `.gitignore:20:orchestration/evidence/`. `git ls-files orchestration/evidence` returns nothing.
- Impact: the figures "served 29, local right 15, Sonnet right 14", "random 20: local right 6, Sonnet right 14" and "21 of 48 overly literal" have no run record. They exist only in an untracked hand-read file. They are labelled as a verifier's reading, not a result, so this is not a false claim. It is not reproducible from the repo, and it leaves the "local judge errs kind, Sonnet errs literal" conclusion resting on a file a clone will not have.
- Owner: T8.1. Commit the adjudication as a tracked document, or say in DECISIONS that it is a local-only hand read. Do not use these figures in any claim.

### 4. A serving-match test passes silently if its evidence file is missing
- File: `tests/test_serving.py`, `test_the_config_matches_the_e7_choice_on_record`. It does `if not chosen.is_file(): return`.
- Impact: in a clone where `results/e7/v2/chosen.json` is absent, the only check that `serving.py` matches the E7 choice becomes a passing no-op, not a skip. It currently runs, since the file exists, so nothing is hidden today.
- Owner: T7.4. Use `pytest.skip` or assert the file exists.

## Deeper findings on the known items (no new rule changes proposed)
- **E14 judge kindness, depth.** The 141-to-14 split is across all 220 answers. On the 54 served answers it is 28 to 1. Sonnet marks 37 of 54 served answers as having an unsupported claim, the local judge 24 of 54, and the verifier's hand read of 20 found about 7 of 20. The judge prompt contains no question or diagnosis, so no grader can catch a wrong-disease transfer by construction (supported-031). This is a design limit of `judge.txt`, not only of the model. T8.1 should know that swapping judges will not fix it.
- **Sample composition.** The pooled E14 "sample" mixes all systems (prompt variants), and only 54 of its 220 answers are served. The pooled kappa 0.630 is therefore not the served figure (0.653). DECISIONS reports both, so this is disclosed.
- **Citations for `[1,2]` and the first-plant rule.** Both known items are as described in the T6.4 verdict. In `answer.py:216`, `_MARK` matches only `[n]`, so a model answering `[1,2]` gets the "no supporting document" prefix even though the evidence was used. That is safe-side, but the user sees "ungrounded" for a grounded answer.
- **Leakage check.** No future-turn or test-split path was found. History is bounded to 4 turns of 500 characters and is stripped of the tags. The blind file carries no local verdict. E14 refuses answers outside `rag_dev`.
- **Bloat check.** Nothing unrequested found. `serving.py` and `chosen.json` exist for the "one setup" requirement. The v1 files are kept as labelled evidence.
