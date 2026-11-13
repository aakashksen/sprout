"""SPROUT live demo server: the real model, the 237-guide search, and Claude Haiku writing questions and answers.

Run:  .venv/Scripts/python tools/live_demo.py --site <folder with index.html, tour.html, live.html>
Then open http://127.0.0.1:8765/live.html

Key: SPROUT_ANTHROPIC_API_KEY, read from the environment or from .env at the repo root. Never printed.
The odds are updated by code here, never by the model. Spending is capped per visitor, per IP and for the whole server.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, session

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "e6" / "model.onnx"
CALIBRATION = ROOT / "results" / "e15" / "refit" / "chosen.json"
INDEX = ROOT / "knowledge" / "index"
MANIFEST = ROOT / "knowledge" / "manifest.csv"
FALLBACK_QUESTIONS = ROOT / "datasets" / "followup" / "questions.jsonl"

LLM = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6  # dollars per token, Haiku 5.5 list price
CAP_VISITOR, CAP_IP, CAP_SERVER = 1.00, 3.00, 10.00  # dollars per day
K_YES, K_NO, K_CHAT_FOR, K_CHAT_AGAINST = 3.0, 0.35, 2.0, 0.5
MAX_QUESTIONS = 5
CANDIDATE_FLOOR = 0.05
CHECK_TTL, MAX_CHECKS = 3600, 2000  # seconds a check is kept, and how many at most
CHECKS_PER_MINUTE = 12  # photos one IP may send per minute

# class -> (crop in the index, disease tag in the index or None, name shown)
CLASSES = {
    "Apple___Apple_scab": ("apple", "apple scab", "Apple scab"),
    "Apple___Black_rot": ("apple", "black rot", "Apple black rot"),
    "Apple___Cedar_apple_rust": ("apple", "cedar apple rust", "Cedar apple rust"),
    "Apple___healthy": ("apple", None, "Healthy apple"),
    "Blueberry___healthy": ("blueberry", None, "Healthy blueberry"),
    "Cherry_(including_sour)___healthy": ("cherry", None, "Healthy cherry"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("corn", "gray leaf spot", "Corn gray leaf spot"),
    "Corn_(maize)___Common_rust_": ("corn", "common rust", "Corn common rust"),
    "Corn_(maize)___Northern_Leaf_Blight": ("corn", "northern corn leaf blight", "Northern corn leaf blight"),
    "Grape___Black_rot": ("grape", "black rot", "Grape black rot"),
    "Grape___healthy": ("grape", None, "Healthy grape"),
    "Orange___Haunglongbing_(Citrus_greening)": ("orange", "citrus greening", "Citrus greening"),
    "Peach___healthy": ("peach", None, "Healthy peach"),
    "Pepper,_bell___Bacterial_spot": ("pepper", "bacterial spot", "Pepper bacterial spot"),
    "Pepper,_bell___healthy": ("pepper", None, "Healthy bell pepper"),
    "Potato___Early_blight": ("potato", "early blight", "Potato early blight"),
    "Potato___Late_blight": ("potato", "late blight", "Potato late blight"),
    "Raspberry___healthy": ("raspberry", None, "Healthy raspberry"),
    "Soybean___healthy": ("soybean", None, "Healthy soybean"),
    "Squash___Powdery_mildew": ("squash", "powdery mildew", "Squash powdery mildew"),
    "Strawberry___healthy": ("strawberry", None, "Healthy strawberry"),
    "Tomato___Bacterial_spot": ("tomato", "bacterial spot", "Tomato bacterial spot"),
    "Tomato___Early_blight": ("tomato", "early blight", "Tomato early blight"),
    "Tomato___Late_blight": ("tomato", "late blight", "Tomato late blight"),
    "Tomato___Leaf_Mold": ("tomato", "leaf mold", "Tomato leaf mold"),
    "Tomato___Septoria_leaf_spot": ("tomato", "septoria leaf spot", "Tomato Septoria leaf spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "yellow leaf curl", "Tomato yellow leaf curl virus"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "tomato mosaic virus", "Tomato mosaic virus"),
    "Tomato___healthy": ("tomato", None, "Healthy tomato"),
}
PLANT_NAME = {"pepper": "Bell pepper", "corn": "Corn", "orange": "Orange"}


def plant_of(cls: str) -> str:
    crop = CLASSES[cls][0]
    return PLANT_NAME.get(crop, crop.capitalize())


def _env_value(names: tuple[str, ...]) -> str:
    for n in names:
        if os.environ.get(n.upper(), "").strip():
            return os.environ[n.upper()].strip()
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if _ and name.strip().lower() in names:
                return value.strip().strip('"').strip("'")
    return ""


def load_key() -> str:
    return _env_value(("sprout_anthropic_api_key", "key"))


def load_workspace() -> str:
    """Only needed for a key that is not scoped to a workspace (the API then asks for anthropic-workspace-id)."""
    return _env_value(("sprout_anthropic_workspace_id", "workspace"))


# ---------- spending ----------
class Budget:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.day = time.strftime("%Y-%m-%d")
        self.spent: dict[str, float] = {}

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.spent = today, {}

    def allowed(self, visitor: str, ip: str) -> str | None:
        with self.lock:
            self._roll()
            if self.spent.get("server", 0) >= CAP_SERVER:
                return "The demo has used today's allowance. It resets tomorrow."
            if self.spent.get("v:" + visitor, 0) >= CAP_VISITOR or self.spent.get("ip:" + ip, 0) >= CAP_IP:
                return "You've used today's allowance for the live demo. It resets tomorrow."
            return None

    def charge(self, visitor: str, ip: str, dollars: float) -> None:
        with self.lock:
            self._roll()
            for k in ("server", "v:" + visitor, "ip:" + ip):
                self.spent[k] = self.spent.get(k, 0) + dollars


# ---------- the model call ----------
class Writer:
    def __init__(self, key: str) -> None:
        import anthropic

        ws = load_workspace()
        self.client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=1,
                                          default_headers={"anthropic-workspace-id": ws} if ws else None)

    def json(self, system: str, user: str, spend, max_tokens: int = 1200, text_key: str | None = None) -> dict:
        kwargs = dict(model=LLM, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        try:
            msg = self.client.messages.create(**kwargs, output_config={"effort": "low"})
        except TypeError:
            msg = self.client.messages.create(**kwargs, extra_body={"output_config": {"effort": "low"}})
        spend(msg.usage.input_tokens * PRICE_IN + msg.usage.output_tokens * PRICE_OUT)  # charged before any parsing
        if msg.stop_reason == "refusal":
            raise ValueError("the model declined")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else {}
        except (AttributeError, ValueError):
            if text_key and text.strip():
                return {text_key: text.strip()}  # the model answered in prose: keep the answer
            raise ValueError(f"no JSON in the reply (stop: {msg.stop_reason})")


# ---------- one check ----------

POOL_SIZE = 6          # questions written per batch, picked from locally after each answer
LIKELY = 0.70          # below the calibrated bar but worth naming as "likely"
PLANT_DOWN = 0.02      # an answered plant question keeps other plants' classes at 2% of their weight


@dataclass
class Check:
    id: str
    probs: dict[str, float]
    status: str  # "asking", "final", "not_plant", "invalid"
    asked: list[dict] = field(default_factory=list)
    question: dict | None = None
    final: dict | None = None
    message: str = ""
    pool: list[dict] = field(default_factory=list)
    pool_jobs: list = field(default_factory=list)
    pool_sets: list = field(default_factory=list)
    advice: dict[str, Future] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    plant_known: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)  # guards the question pool
    step: threading.RLock = field(default_factory=threading.RLock)  # one answer, stop or chat turn at a time
    final_cls: str = ""
    created: float = field(default_factory=time.time)

    def top(self) -> tuple[str, float]:
        return max(self.probs.items(), key=lambda kv: kv[1])

    def candidates(self) -> list[str]:
        ranked = sorted(self.probs, key=self.probs.get, reverse=True)
        return ranked[:3] + [c for c in ranked[3:5] if self.probs[c] >= CANDIDATE_FLOOR]


def normalise(p: dict[str, float]) -> dict[str, float]:
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def apply(p: dict[str, float], supports: list[str], factor: float) -> dict[str, float]:
    return normalise({k: v * (factor if k in supports else 1.0) for k, v in p.items()})


def expected_entropy(p: dict[str, float], supports: list[str]) -> float:
    """How settled the odds would be after this question, averaged over a yes and a no."""
    s = sum(p.get(c, 0) for c in supports)
    p_yes = s * 0.8 + (1 - s) * 0.15
    return p_yes * entropy(apply(p, supports, K_YES)) + (1 - p_yes) * entropy(apply(p, supports, K_NO))


class Engine:
    def __init__(self, writer: Writer | None) -> None:
        from sprout.rag import serving
        from sprout.rag.answer import load_sources
        from sprout.rag.retrieve import Retriever
        from sprout.serve.identify import Identifier

        self.identifier = Identifier(MODEL, CALIBRATION)
        self.retriever = Retriever(INDEX, config=serving.retrieval_config())
        self.sources = load_sources(MANIFEST)
        self.writer = writer
        self.threshold = self.identifier.threshold
        self.fallback = [json.loads(line) for line in FALLBACK_QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.checks: dict[str, Check] = {}
        self.jobs = ThreadPoolExecutor(max_workers=8)
        self.search_lock = threading.Lock()  # the retriever is shared; keep its searches one at a time

    def warm(self) -> None:
        self.retriever.warm(rerank=True)

    def search(self, *a, **kw):
        with self.search_lock:
            return self.retriever.search(*a, **kw)

    # ----- the photo -----
    def probs_for(self, image_bytes: bytes):
        import numpy as np

        from sprout.serve.calibrate import apply_temperature

        ident = self.identifier.identify(image_bytes)
        if ident.status in ("invalid", "not_plant"):
            return ident, None
        idf = self.identifier
        raw = idf.model.predict(idf._decode(image_bytes))[idf._columns]
        if not raw.sum() > 0:
            return ident, None
        raw = raw / raw.sum()
        probs = apply_temperature(np.log(np.maximum(raw, 1e-12)), idf.temperature)
        return ident, {c: float(p) for c, p in zip(idf.classes, probs)}

    def start(self, image_bytes: bytes, spend) -> Check:
        ident, probs = self.probs_for(image_bytes)
        cid = uuid.uuid4().hex
        if probs is None:
            status = ident.status if ident.status in ("invalid", "not_plant") else "invalid"
            chk = Check(cid, {}, status, message=ident.guidance or "This photo couldn't be read. Try a closer photo of one leaf.")
        else:
            chk = Check(cid, probs, "asking")
            self.prefetch(chk, spend)
        self.checks[cid] = chk
        self.evict()
        return chk

    def evict(self) -> None:
        now = time.time()
        for cid in [c for c, k in self.checks.items() if now - k.created > CHECK_TTL]:
            self.checks.pop(cid, None)
        while len(self.checks) > MAX_CHECKS:
            self.checks.pop(next(iter(self.checks)))

    # ----- background work: questions and advice, written while the visitor reads -----
    def prefetch(self, chk: Check, spend) -> None:
        """Start, in the background, everything the next screens may need: a question batch for each plant the
        candidates span (so answering the plant question never waits), and the advice for the two likeliest classes."""
        if not self.writer:
            return
        ranked = sorted(chk.probs, key=chk.probs.get, reverse=True)
        by_crop: dict[str, list[str]] = {}
        for c in ranked[:8]:
            if CLASSES[c][1] and chk.probs[c] >= 0.01:
                by_crop.setdefault(CLASSES[c][0], []).append(c)
        groups = [frozenset(v[:4]) for v in by_crop.values() if len(v) >= 2]
        cands = frozenset(c for c in chk.candidates() if CLASSES[c][1])
        if len(cands) >= 2:
            groups.append(cands)
        for g in groups:
            if not any(g <= done for done in chk.pool_sets):
                chk.pool_sets.append(g)
                chk.pool_jobs.append(self.jobs.submit(self.write_pool, chk, sorted(g), spend))
        for c in ranked[:2]:
            if c not in chk.advice and CLASSES[c][1] and chk.probs[c] >= 0.1:
                chk.advice[c] = self.jobs.submit(self.write_advice, c, spend)

    def symptom_passages(self, cls: str) -> list[dict]:
        crop, disease, name = CLASSES[cls]
        found = self.search(f"{disease} symptoms on leaves, what the spots look like", crop=crop, disease=disease, k=3) \
            or self.search(f"{name} symptoms on leaves", crop=crop, k=3)
        return [self._passage(p, cls) for p in found[:3]]

    def _passage(self, p, cls: str | None) -> dict:
        info = self.sources.get(p.source)
        return {"text": p.text[:900], "source": p.source, "cls": cls,
                "title": info.title if info else p.source, "publisher": info.publisher if info else ""}

    @staticmethod
    def cite(pas: dict) -> str:
        return f"{pas['publisher'] or 'Extension guide'}, {pas['title']}"

    def write_pool(self, chk: Check, cands: list[str], spend) -> list[dict]:
        lines, passages = [], {}
        for c in cands:
            for pas in self.symptom_passages(c):
                n = str(len(passages) + 1)
                passages[n] = pas
                lines.append(f"[{n}] ({c}) {pas['text']}")
        if not lines:
            return []
        system = ("You write yes/no questions that help a gardener tell plant diseases apart by looking at the plant. "
                  "Use ONLY the numbered guide passages. Each question is one short, plain sentence about something "
                  "visible, ends with '?', and has no jargon. Return JSON only.")
        user = ("Candidate classes: " + ", ".join(cands) + "\nPassages:\n" + "\n".join(lines)
                + f'\n\nReturn {{"questions": [{{"text": "...", "supports": ["<class>", ...], "passages": ["<n>"]}}]}} with '
                  f"{POOL_SIZE} questions that cover different differences. 'supports' = the classes a YES points to; "
                  "never all of them. Cite passages about a class the question supports.")
        try:
            out = self.writer.json(system, user, spend, max_tokens=4000)
        except Exception as e:
            print("question batch failed, using the saved questions:", type(e).__name__, str(e)[:160], flush=True)
            return []
        pool = []
        for q in out.get("questions", []):
            if not isinstance(q, dict):
                continue
            text = str(q.get("text", "")).strip()
            sup = [s for s in q.get("supports", []) if s in cands]
            ps = [str(x) for x in q.get("passages", []) if str(x) in passages and passages[str(x)]["cls"] in sup]
            if text.endswith("?") and len(text.split()) <= 30 and sup and len(sup) < len(cands) and ps:
                pool.append({"id": uuid.uuid4().hex[:8], "text": text, "supports": sup, "source": self.cite(passages[ps[0]])})
        with chk.lock:
            chk.pool.extend(pool)
        return pool

    def write_advice(self, cls: str, spend) -> dict:
        crop, disease, name = CLASSES[cls]
        found = self.search(f"how to manage and control {disease}", crop=crop, disease=disease, k=4) \
            or self.search(f"how to manage and control {name}", crop=crop, k=4)
        pas = [self._passage(p, cls) for p in found[:4]]
        if not pas:
            return {"text": "SPROUT's guides don't cover treatment for this. Your local extension office can help.", "source": ""}
        system = ("You give a home gardener clear care advice for a diagnosed plant disease from the numbered guide passages. "
                  "Three short sentences: what to do now, how to stop it spreading, and whether the plant can recover. "
                  "Plain words. If a product is named, add 'follow the label'. Return JSON only.")
        user = f"Diagnosis: {name}\nPassages:\n" + "\n".join(f"[{i + 1}] {x['text']}" for i, x in enumerate(pas)) \
            + '\nReturn {"advice": "...", "passage": <number used most>}'
        try:
            out = self.writer.json(system, user, spend, max_tokens=2500)
        except Exception as e:
            print("advice failed:", type(e).__name__, str(e)[:160], flush=True)
            return {"text": "The advice couldn't be written just now. Ask in the chat below.", "source": ""}
        n = out.get("passage")
        used = pas[n - 1] if isinstance(n, int) and 1 <= n <= len(pas) else pas[0]
        return {"text": str(out.get("advice", "")).strip(), "source": self.cite(used)}

    # ----- the next question: picked locally, no waiting on the model -----
    def plant_question(self, chk: Check) -> dict | None:
        if chk.plant_known:
            return None
        crops = []
        for c in chk.candidates():
            crop = CLASSES[c][0]
            if chk.probs[c] >= CANDIDATE_FLOOR and crop not in crops:
                crops.append(crop)
        if len(crops) < 2:
            return None
        names = [PLANT_NAME.get(c, c.capitalize()) for c in crops]
        return {"id": "plant", "kind": "plant", "options": [{"value": c, "label": n} for c, n in zip(crops, names)],
                "text": "Is this a " + " or a ".join(n.lower() for n in names) + " plant?",
                "source": "The photo fits diseases of more than one plant, and the plant settles it."}

    def pick(self, chk: Check, wait: bool) -> dict | None:
        q = self.plant_question(chk)
        if q:
            return q
        cands = chk.candidates()
        seen = {a["text"].lower() for a in chk.asked}

        def best(pool):
            scored = []
            for q in pool:
                sup = [c for c in q["supports"] if c in cands]
                if q["text"].lower() in seen or not sup or len(sup) == len(cands):
                    continue
                scored.append((expected_entropy(chk.probs, sup), q))
            return min(scored, key=lambda t: t[0])[1] if scored else None

        with chk.lock:
            q = best(chk.pool)
        pending = [j for j in chk.pool_jobs if not j.done()]
        if q is None and pending and not wait:
            q = self.best_fallback(chk, best)  # a saved guide question now beats waiting for the written batch
            return q if q is not None else "pending"
        if q is None and pending:
            for j in pending:
                try:
                    j.result(timeout=25)
                except Exception as e:
                    print("question batch failed:", type(e).__name__, str(e)[:160], flush=True)
                with chk.lock:
                    q = best(chk.pool)
                if q is not None:
                    break
        if q is None:
            q = self.best_fallback(chk, best)
        return q

    def best_fallback(self, chk: Check, best):
        fb = []
        for f in self.fallback:
            info = self.sources.get(f["document"])
            fb.append({"id": f["id"], "text": f["text"], "supports": f["classes"],
                       "source": f"{info.publisher}, {info.title}" if info else f["document"]})
        return best(fb)

    def next_step(self, chk: Check, spend, wait: bool = True) -> None:
        top, p = chk.top()
        if p >= self.threshold or len(chk.asked) >= MAX_QUESTIONS:
            return self.finish(chk, spend)
        q = self.pick(chk, wait)
        if q == "pending":
            chk.question = None
            chk.message = f"{round(p * 100)}% sure so far."
            return
        if q is None:
            return self.finish(chk, spend)
        chk.question = q
        chk.message = ("The photo leaves a few possibilities. Answer a couple of questions to narrow it down."
                       if not chk.asked else f"{round(p * 100)}% sure so far.")
        self.prefetch(chk, spend)

    def answer(self, chk: Check, qid: str, ans: str, spend) -> None:
        q = chk.question
        if q is None or q["id"] != qid:
            return False
        before = chk.top()[1]
        if q.get("kind") == "plant":
            chk.plant_known = True
            if ans in [o["value"] for o in q["options"]]:
                chk.probs = normalise({k: v * (1.0 if CLASSES[k][0] == ans else PLANT_DOWN) for k, v in chk.probs.items()})
                shown = PLANT_NAME.get(ans, ans.capitalize())
            else:
                shown = "Not sure"
        else:
            if ans == "yes":
                chk.probs = apply(chk.probs, q["supports"], K_YES)
            elif ans == "no":
                chk.probs = apply(chk.probs, q["supports"], K_NO)
            shown = {"yes": "Yes", "no": "No"}.get(ans, "Not sure")
        chk.asked.append({"text": q["text"], "answer": shown, "before": before})
        chk.question = None
        self.prefetch(chk, spend)
        self.next_step(chk, spend, wait=False)
        return True

    # ----- the answer -----
    def finish(self, chk: Check, spend) -> None:
        chk.question = None
        chk.status = "final"
        top, p = chk.top()
        chk.final_cls = top
        crop, disease, name = CLASSES[top]
        pct = round(p * 100)
        if p >= self.threshold:
            level, chk.message = f"{pct}% sure", ""
        elif p >= LIKELY:
            level, chk.message = f"Likely, {pct}%", (f"Fairly confident. SPROUT only says it's sure above {round(self.threshold * 100)}%, "
                                                     "the bar its tests set. A closer photo in daylight could confirm it.")
        else:
            level, chk.message = f"Best guess, {pct}%", "It couldn't settle it. Try a closer photo in daylight, or ask your local extension office."
        if disease is None:
            chk.final = {"name": name, "sure": level, "source": "",
                         "advice": "No disease stood out. A healthy-looking result never rules out disease; check again if spots spread."}
            return
        job = chk.advice.get(top)
        if job is None and self.writer:
            job = chk.advice[top] = self.jobs.submit(self.write_advice, top, spend)
        chk.final = {"name": name, "sure": level, "advice": "", "source": "", "advice_pending": bool(job)}
        if job is not None and job.done():
            self.fill_advice(chk, wait=False)

    def fill_advice(self, chk: Check, wait: bool) -> None:
        """The final answer shows at once; its care advice fills in when the background writing is done."""
        job = chk.advice.get(chk.final_cls)
        if job is None or chk.final is None or not chk.final.get("advice_pending"):
            return
        try:
            advice = job.result(timeout=30 if wait else 0)
        except TimeoutError:
            return  # still writing: the page asks again
        except Exception as e:
            print("advice job failed:", type(e).__name__, str(e)[:160], flush=True)
            advice = {"text": "The advice couldn't be written just now. Ask in the chat below.", "source": ""}
        chk.final.update(advice=advice["text"], source=advice["source"], advice_pending=False)

    # ----- the chat: a conversation, grounded in the guides where they help -----
    def chat(self, chk: Check, text: str, spend) -> dict:
        top = chk.top()[0]
        crop, disease, name = CLASSES[top]
        query = f"{disease or name}: {text}"
        found = (self.search(query, crop=crop, disease=disease, k=3) if disease else []) \
            or self.search(query, crop=crop, k=3) or self.search(query, k=3)
        manage = self.search(f"how to manage, control and stop the spread of {disease}", crop=crop, disease=disease, k=2) if disease else []
        seen, pas = set(), []
        for p in list(found) + list(manage):
            if (p.source, p.text[:80]) not in seen:
                seen.add((p.source, p.text[:80]))
                pas.append(self._passage(p, None))
        pas = pas[:5]
        cands = chk.candidates()
        answered = "; ".join(f"{a['text']} {a['answer']}" for a in chk.asked) or "none"
        system = (
            "You are SPROUT's plant doctor, talking with a home gardener about their plant. Be warm, direct and practical. "
            "Answer the actual question in the first sentence. At most 4 short sentences, under 90 words, plain words, no lists. "
            "Use the numbered guide passages when they help and cite them like [2]. If the guides don't cover something, "
            "still help from general gardening knowledge and mark that sentence with '(general advice)'. Never invent product "
            "rates; for any chemical say to follow the label. Only add 'evidence' when the gardener DESCRIBES something they "
            "see on the plant; a question like 'can it be saved' is not evidence. Return JSON only."
        )
        context = (f"Diagnosis so far: {name} ({round(chk.top()[1] * 100)}%). Plant: {plant_of(top)}. "
                   f"Other candidates: {', '.join(c for c in cands if c != top)}. Their answers: {answered}.\n"
                   "Guide passages:\n" + ("\n".join(f"[{i + 1}] {x['publisher']}, {x['title']}: {x['text']}" for i, x in enumerate(pas))
                                          or "(none found)"))
        convo = "".join(f"\n{'Gardener' if h['role'] == 'user' else 'SPROUT'}: {h['text']}" for h in chk.history[-6:])
        user = (context + "\n\nConversation so far:" + (convo or " (none)") + f"\nGardener: {text}\n\n"
                + 'Return {"reply": "...", "cited": [<passage numbers used>], "evidence": [{"class": "<candidate class>", "direction": "for"|"against", "because": "<the gardener exact words describing what they see>"}]}')
        try:
            out = self.writer.json(system, user, spend, max_tokens=3000, text_key="reply")
        except Exception as e:
            print("chat failed:", type(e).__name__, str(e)[:160], flush=True)
            return {"reply": "The chat couldn't reach the AI model just now. Try again in a moment.", "sources": [], "shift": ""}
        before_cls, before = chk.top()
        said = text.lower()
        for ev in out.get("evidence", []) or []:
            if not isinstance(ev, dict) or ev.get("direction") not in ("for", "against"):
                continue
            c, because = ev.get("class"), str(ev.get("because", "")).strip().lower()
            # only something the gardener actually described counts: the quote must be their own words
            if c in cands and len(because.split()) >= 3 and because in said:
                chk.probs = apply(chk.probs, [c], K_CHAT_FOR if ev.get("direction") == "for" else K_CHAT_AGAINST)
        after_cls, after = chk.top()
        shift = ""
        if after_cls != before_cls:
            shift = f"{CLASSES[before_cls][2]} {round(before * 100)}% → {CLASSES[after_cls][2]} {round(after * 100)}%"
        elif abs(after - before) >= 0.01:
            shift = f"{CLASSES[after_cls][2]} {round(before * 100)}% → {round(after * 100)}%"
        reply = str(out.get("reply", "")).strip()
        # renumber the citations so [1], [2] in the reply match the source list shown under it
        order: list[str] = []
        for m in re.finditer(r"\[(\d+)\]", reply):
            n = int(m.group(1))
            if 1 <= n <= len(pas) and self.cite(pas[n - 1]) not in order:
                order.append(self.cite(pas[n - 1]))

        def renum(m):
            n = int(m.group(1))
            return f" [{order.index(self.cite(pas[n - 1])) + 1}]" if 1 <= n <= len(pas) else ""
        reply = re.sub(r"\s*\[(\d+)\]", renum, reply)
        chk.history += [{"role": "user", "text": text}, {"role": "sprout", "text": reply}]
        return {"reply": reply, "sources": order, "shift": shift}

    def view(self, chk: Check) -> dict:
        if chk.status in ("invalid", "not_plant"):
            return {"id": chk.id, "status": chk.status, "message": chk.message, "odds": [], "plant": ""}
        ranked = sorted(chk.probs, key=chk.probs.get, reverse=True)
        q = chk.question
        return {
            "id": chk.id, "status": chk.status, "plant": plant_of(ranked[0]), "message": chk.message,
            "odds": [{"name": CLASSES[c][2], "p": round(chk.probs[c], 4)} for c in ranked[:3]],
            "question": {k: q[k] for k in ("id", "text", "source") if k in q} | ({"options": q["options"]} if q and q.get("options") else {}) if q else None,
            "asked": [{"text": a["text"], "answer": a["answer"]} for a in chk.asked],
            "final": chk.final,
        }


def create(site: Path, engine: Engine) -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_bytes(32)
    app.config.update(MAX_CONTENT_LENGTH=20 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
    budget = Budget()
    gone = "This check is no longer on the server (it was restarted or the check is old). Your photo will be checked again."

    def guard():
        if "v" not in session:
            session["v"] = uuid.uuid4().hex
        v, ip = session["v"], request.remote_addr or "?"
        msg = budget.allowed(v, ip)
        if msg:
            return None, (jsonify(message=msg), 429)
        return (lambda dollars: budget.charge(v, ip, dollars)), None

    recent: dict[str, list[float]] = {}

    def body() -> dict:
        b = request.get_json(force=True, silent=True)
        return b if isinstance(b, dict) else {}

    def find(body: dict, status: str | tuple | None = None):
        chk = engine.checks.get(str(body.get("check", "")))
        if chk is None:
            return None, (jsonify(message=gone, code="gone"), 410)
        if status and chk.status not in ((status,) if isinstance(status, str) else status):
            return None, (jsonify(engine.view(chk) | {"message": "That step is already done.", "code": "stale"}), 409)
        return chk, None

    @app.post("/api/check")
    def check():
        spend, err = guard()
        if err:
            return err
        ip, now = request.remote_addr or "?", time.time()  # behind a proxy, wrap the app in werkzeug's ProxyFix
        recent[ip] = [t for t in recent.get(ip, []) if now - t < 60] + [now]
        if len(recent[ip]) > CHECKS_PER_MINUTE:
            return jsonify(message="That's a lot of photos in a minute. Wait a moment and try again."), 429
        f = request.files.get("photo")
        if f is None:
            return jsonify(message="No photo was sent."), 400
        chk = engine.start(f.read(), spend)
        if chk.status == "asking":
            engine.next_step(chk, spend, wait=False)  # a question only if one is ready; /api/next waits for the batch
        return jsonify(engine.view(chk) | {"pending": chk.status == "asking" and chk.question is None})

    @app.post("/api/next")
    def nxt():
        spend, err = guard()
        if err:
            spend = lambda dollars: None  # noqa: E731  past the cap the saved questions still work, nothing new is written
        chk, err = find(body(), "asking")
        if err:
            return err
        with chk.step:
            if chk.question is None and chk.status == "asking":
                engine.next_step(chk, spend, wait=True)
        return jsonify(engine.view(chk) | {"pending": False})

    @app.post("/api/answer")
    def answer():
        spend, err = guard()
        if err:
            spend = lambda dollars: None  # noqa: E731  answering and stopping use code only; never trap a visitor at the cap
        b = body()
        chk, err = find(b, "asking")
        if err:
            return err
        with chk.step:
            if chk.status != "asking":
                return jsonify(engine.view(chk) | {"message": "That step is already done.", "code": "stale"}), 409
            if b.get("stop"):
                engine.finish(chk, spend)
            elif not engine.answer(chk, str(b.get("question", "")), str(b.get("answer", "")), spend):
                return jsonify(engine.view(chk) | {"message": "That question was already answered.", "code": "stale"}), 409
        return jsonify(engine.view(chk) | {"pending": chk.status == "asking" and chk.question is None})

    @app.post("/api/advice")
    def advice():
        chk, err = find(body(), "final")
        if err:
            return err
        engine.fill_advice(chk, wait=True)
        return jsonify(engine.view(chk))

    @app.post("/api/chat")
    def chat():
        spend, err = guard()
        if err:
            return err
        b = body()
        chk, err = find(b, ("asking", "final"))
        if err:
            return err
        text = str(b.get("text", "")).strip()[:500]
        if not text:
            return jsonify(message="Type a question first."), 400
        if engine.writer is None:
            return jsonify(message="The chat needs the API key."), 503
        with chk.step:
            out = engine.chat(chk, text, spend)
        return jsonify(out | {"odds": engine.view(chk)["odds"]})

    @app.get("/")
    def home():
        return send_from_directory(site, "index.html")

    @app.get("/<path:name>")
    def files(name: str):
        if name == "live.html":
            html = (site / "live.html").read_text(encoding="utf-8")
            html = html.replace('<meta name="sprout-api" content="">', '<meta name="sprout-api" content="same-origin">')
            return Response(html, mimetype="text/html", headers={"Cache-Control": "no-store"})
        return send_from_directory(site, name)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    key = load_key()
    writer = Writer(key) if key else None
    print("API key:", "found" if key else "missing (saved questions only, no chat)",
          "| workspace id:", "found" if load_workspace() else "not set", flush=True)
    engine = Engine(writer)
    print("warming the guide search...", flush=True)
    engine.warm()
    print(f"ready: http://127.0.0.1:{a.port}/live.html", flush=True)
    create(Path(a.site), engine).run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
