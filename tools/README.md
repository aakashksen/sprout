# tools

One line per script. Run from the repo root.

- `backfill_mlflow.py`: log existing run records into the local MLflow store.
- `build_extra_sources.py`: rebuild the extra training manifest and class map.
- `build_index_gpu.py`: build a document index and embed its chunks on the GPU.
- `check_candidates_doc.py`: fail unless each uncovered class has a section in the candidates doc.
- `check_datasets_doc.py`: fail unless every dataset in docs/datasets.md has URL, license, classes and approval lines.
- `check_report_numbers.py`: fail unless every number in the report traces to a run record.
- `download_plant_pathology.py`: fetch the Plant Pathology 2021 mirror from Hugging Face.
- `e11_answer_check.py`: check the saved E11 answers against an answerer that reads the list, not the photo.
- `extra_trim.py`: list the extra photos to remove when PlantDoc is held out (gate G1).
- `fetch_documents.py`: politely download public documents listed in a CSV into knowledge/raw.
- `label_review.py`: run the T4.7 label-review stages in order.
- `leak_orb.py`: independent train/test leak search with ORB keypoints and RANSAC.
- `live_demo.py`: the live demo server (real model, guide search, Claude Haiku questions and answers).
- `run_e6.py`: run the E6 ONNX stages in order.
- `run_e7.py`: run experiment E7 (RAG on rag_dev) one resumable stage at a time.
- `run_loop.py`: optional unattended loop giving each task batch a fresh Claude Code session.
- `taskctl.py`: the plan's state machine; a task is done only through this tool.
