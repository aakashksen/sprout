# Deploying SPROUT

This describes how the serving image is built and run. Nothing here has been pushed or deployed; deploying is the
person's decision (H3 in `docs/HUMAN-GATES.md`).

## What the image is

`deploy/Dockerfile` builds a `python:3.12-slim` image from `requirements/serve.txt`, which has exact pins and no
PyTorch. Retrieval embeds with `fastembed` on ONNX Runtime. The image runs as user `sprout` (uid 10001) and has no
`.env`, datasets, raw documents, `legacy/`, `orchestration/`, or model files (`.dockerignore` and a test check
this). COCO and PlantDoc images never enter an image. The only knowledge file copied is `knowledge/manifest.csv`.

```
docker compose build app
SPROUT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" docker compose up app
```

Training dependencies live in `requirements/train.txt` and are never installed in the image.

## Configuration (environment only)

| Variable | Meaning |
| --- | --- |
| `SPROUT_SECRET_KEY` | Required, at least 16 characters. The app refuses to start without it. |
| `SPROUT_ENV` | `production` (default), `development`, or `testing`. |
| `SPROUT_DATA_DIR` | Database and private uploads. `/data` in the image. |
| `SPROUT_MANIFEST` | Source titles and links for chat. Defaults to the copied manifest. |
| `SPROUT_INDEX_DIR` | Retrieval index (`chunks.jsonl`). `/models/index`. Without it chat answers "no supporting document". |
| `SPROUT_MODEL_PATH`, `SPROUT_CALIBRATION_PATH` | The ONNX classifier and the E5 calibration file (`results/e5/chosen.json` shape, with `model_sha256` and the `classes` that may be named; a refit for the served model writes both). The two must name the same checkpoint or `Identifier` refuses to start, and so does the app. Default `/models/model.onnx` and `/models/chosen.json`. Both must exist or `/readyz` says "model missing". |
| `SPROUT_OLLAMA_URL`, `SPROUT_WRITER_MODEL` | Where the writer runs and which model. Default `http://127.0.0.1:11434` and `qwen3.5:4b-q4_K_M`. `localhost` is refused (it adds about 2 s per call). In a container, 127.0.0.1 is the container itself, so `docker-compose.yml` sets `http://host.docker.internal:11434`; Ollama on the host must listen beyond loopback (`OLLAMA_HOST=0.0.0.0`). |
| `SPROUT_CHAT_TIMEOUT_SECONDS`, `SPROUT_CHAT_LLM_TIMEOUT_SECONDS` | Budget for one generated answer (default 50 s) and for its model call (default 40 s). Chosen, not measured, and kept under gunicorn's 60 s timeout; revisit against E8's generation p95. |
| `SPROUT_MEASUREMENT` | `1` only for E8 runs: lifts the rate limits and adds a `Server-Timing` header to the upload and chat routes. `/readyz` reports `"measurement": true`. Never set it in a deployment. |
| `SPROUT_HTTPS` | `1` makes cookies secure. Set it behind HTTPS (Cloud Run is). |
| `SPROUT_WORKERS`, `SPROUT_THREADS`, `SPROUT_TIMEOUT` | gunicorn settings; see Workers. |

A bind-mounted `/data` (anything other than the compose named volume) must be writable by uid 10001, for example
`chown 10001 <host folder>`; otherwise the app cannot write its database or uploads.

`SPROUT_FAKE_IDENTIFIER=1` loads a test double that answers "tomato early blight" for every image. It is off unless
the variable is exactly `1`, and `deploy/wsgi.py` raises at start-up unless `SPROUT_ENV=testing`, so a production or
development container cannot run it (`test_fake_identifier_refused_outside_testing`,
`test_the_fake_identifier_is_never_on_by_default_and_never_replaces_a_real_model_silently`). Otherwise `wsgi.py` loads
the real `Identifier` from the two paths above. Without them, a production container starts, `/healthz` is 200, and
`/readyz` is 503 with "model missing".

Chat answers through `sprout.rag.answer.answer()` with the served setup (`sprout.rag.serving`). An Ollama outage or a
slow writer makes chat say the service is unavailable and stores nothing; it never makes `/readyz` fail (a restart
would not fix it), but `/readyz` reports the writer's last known state (`unknown`, `ok`, `unavailable`).

## Providing the model and index

Neither is baked into the image; both are mounted read-only at `/models`.

- ONNX classifier and calibration: mount them read-only at `/models/model.onnx` and `/models/chosen.json` (or point
  `SPROUT_MODEL_PATH` and `SPROUT_CALIBRATION_PATH` elsewhere). The model comes from E12's choice; none exists yet,
  so no container has been started with a real one.
- Index: build with `sprout.rag.ingest` on the dev machine from `knowledge/raw`, which stays out of the image, then
  copy the index folder to `/models/index`. Retrieval caches passage vectors in that folder, so a read-only mount
  needs the cache file created before mounting, or each start re-embeds every chunk.
- fastembed downloads its embedding and reranker models on first use into `FASTEMBED_CACHE_PATH`
  (`/models/fastembed`). A container with no network must have that folder filled beforehand. This has not been
  exercised yet; the chain smoke test below needs it.

## Smoke test and what it proves

`tests/test_container.py::test_smoke_in_clean_container` (marked `real`, needs Docker) builds the image, starts a
fresh container, and over HTTP does: sign up, upload, prediction page, a chat question, history, delete, and checks
the stored file is gone, plus non-root, no `.env`/datasets/legacy in the image, and no `torch` package installed.

The full chain, with the real model, index, reranker and writer, is `tests/test_chain_smoke.py` (marked `real`; needs
Docker, Ollama on the host and five environment variables it names). It has been written but not run.

What ran with a test double in `test_smoke_in_clean_container`: the identifier (`FakeIdentifier`, under `SPROUT_ENV=testing`). No index was mounted, so
the question step checks the "no supporting document" path, not retrieval or generated answers. The real chain
(real `Identifier`, index and generated answers, no double) is `tests/test_chain_smoke.py`.

## PyTorch in the serving path

The serving import path (`deploy/wsgi.py`, `sprout.web`, `sprout.rag`, `sprout.serve.types`) imports no torch
(`test_serving_import_path_does_not_import_torch`). Still torch-dependent, and not changed here:

- `sprout.preprocess` imports `torch`, and `sprout.eval.predict` and `sprout.train.*` do too. The real
  `Identifier` (T5.2) must not import `sprout.preprocess` as it stands, or must get a numpy preprocessing twin
  that matches it. `wsgi.py` now imports `sprout.serve.identify`, which does not import torch
  (`test_the_container_path_imports_no_torch_or_pandas`).
- `pyproject.toml` lists `torch` and `torchvision` as main dependencies. The Dockerfile avoids that with
  `pip install --no-deps .` after the pinned set; installing the package with its dependencies would pull torch in.

## Workers

Set `SPROUT_WORKERS` from measured memory. The default is 1 worker with 4 threads. No figure for the real service
exists yet: it needs the real model and index, so it is measured at T7.4 and entered through a run record. Raw notes
from a check made without a model (not a result, not citable) are in `results/baselines/T7.3-memory.json`.

E8 (`python -m sprout.experiments.e8_perf measure`, see docs/DECISIONS.md) starts this image with its own
entrypoint, sends about 5 real HTTP requests one at a time and reads the container's memory from its cgroup. It shows the real
parts answer together; it is too small to size workers or say how the app behaves under load. Each worker loads its own ONNX
session and index, so memory grows by about one worker's size per worker. Choose workers so the total stays under the
container limit with headroom.
On Cloud Run the CPU count also caps useful workers.

## Google Cloud Run

Steps for the person to run (not run by the build). Replace `PROJECT`, `REGION`, and names.

1. Create an Artifact Registry repository and push the image:
   `gcloud auth configure-docker REGION-docker.pkg.dev`, then
   `docker tag sprout-app REGION-docker.pkg.dev/PROJECT/sprout/app:1` and `docker push` it.
2. Put the secret in Secret Manager: `gcloud secrets create sprout-secret-key --data-file=-`.
3. Deploy:
   `gcloud run deploy sprout --image REGION-docker.pkg.dev/PROJECT/sprout/app:1 --region REGION --port 8080
   --set-secrets SPROUT_SECRET_KEY=sprout-secret-key:latest --set-env-vars SPROUT_HTTPS=1,SPROUT_WORKERS=1
   --memory <from the T7.4 measurement> --max-instances 1`
4. Health: use `/healthz` for the startup and liveness checks. `/readyz` is 503 until the model is loaded.
5. Provide the model and index (see above).

Keep `--max-instances 1` while the database is SQLite. Two instances would have two databases.

### Persistent storage

Cloud Run's own disk is in-memory and lost on restart, so users, history, and uploads need one of:

- Cloud Storage volume mount (Cloud Run can mount a bucket at `/data`). Simple, but SQLite on a bucket mount is not
  safe for concurrent writers or locking; use it for `/models` (read-only) and uploads, not the database.
- Filestore (NFS) mounted at `/data`. Works with SQLite at one instance, with a minimum monthly fee.
- Move the database to Cloud SQL (Postgres). The right choice for more than one instance, but the app is written
  for SQLite today, so it is a code change that no task asks for yet.
- A Compute Engine VM running the container with a persistent disk, if Cloud Run's limits do not suit.

The model and index can be a Cloud Storage mount at `/models` (read-only) or be copied into a derived image the
person builds privately. Never put them in a public image.
