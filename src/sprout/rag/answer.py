"""Grounded answers: retrieve, show the model numbered evidence, and cite only passages that were shown."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sprout.llm.client import LLMClient, LLMError
from sprout.rag.query import build_query, crop_of_label
from sprout.serve.types import Identification

GENERAL_CROP = "general"  # the crop name general-guidance documents carry in the index
PROMPT_DIR = Path(__file__).parent / "prompts"
CONFIDENT = 0.8  # a diagnosis below this is described as a possibility
UNCONFIRMED_PREFIX = "unknown:"

UNAVAILABLE = "The answer service is unavailable right now, so I can't answer this. Please try again later."
SEARCH_FAILED = "The document search failed, so I can't answer this. Please try again later."
NO_DOCUMENT = (
    "No supporting document was found for this question, so I can't give sourced advice. "
    "A local extension service can help with it."
)
UNGROUNDED = "No supporting document was cited for this; treat it as general care advice, not sourced advice. "
LABEL_NOTE = "Check the product label and your local rules before using or disposing of any chemical."
CHEMICAL = re.compile(
    r"\b(fungicides?|pesticides?|insecticides?|herbicides?|bactericides?|miticides?|spray(?:s|ed|ing)?|"
    r"copper|mancozeb|chlorothalonil|azoxystrobin|chemicals?|dispos\w+)\b", re.I)
_TAGS = re.compile(r"</?\s*(diagnosis|history|evidence|question)\s*>", re.I)
_MARK = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class SourceInfo:
    title: str
    publisher: str
    url: str  # an http(s) link, or "" when the manifest has none
    note: str  # "" when the source is confirmed, else why there is no link


@dataclass
class Citation:
    source: str
    page: int
    title: str
    url: str
    note: str
    quote: str  # the passage opening, so a one-page document still points at the place


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    asks_for: list[str]
    state: str = "answered"  # answered, ungrounded, no_document, clarify, no_plant, retrieval_failed, unavailable
    passages: list = field(default_factory=list)
    query: str = ""
    prompt_version: str = ""


@dataclass(frozen=True)
class AnswerConfig:
    prompt_version: str = "v1"
    mode: str = "hybrid"
    rerank: bool = True
    k: int = 5
    query_mode: str = "question"
    history_turns: int = 4
    history_chars: int = 500
    max_tokens: int = 400
    timeout_s: float = 180.0
    sources: dict[str, SourceInfo] | None = None


def load_sources(manifest: Path) -> dict[str, SourceInfo]:
    """file -> where to read it. A source the manifest could not confirm says so instead of showing a blank link."""
    from sprout.rag.ingest import load_manifest

    out = {}
    for e in load_manifest(Path(manifest)):
        url = e.source_url.strip()
        if url.startswith(("http://", "https://")):
            out[e.file] = SourceInfo(e.title, e.publisher, url, "")
        elif url.startswith(UNCONFIRMED_PREFIX):
            out[e.file] = SourceInfo(e.title, e.publisher, "", "source not confirmed (inherited from the legacy project)")
        else:
            out[e.file] = SourceInfo(e.title, e.publisher, "", "no link recorded")
    return out


def load_prompt(version: str) -> str:
    path = PROMPT_DIR / f"{version}.txt"
    if not path.is_file():
        raise ValueError(f"unknown prompt version {version!r}")
    return path.read_text(encoding="utf-8").strip()


def _clean(text: str) -> str:
    """Untrusted text may not close or open the tags that separate data from instructions."""
    return _TAGS.sub("", text)


def _shown(label: str) -> str:
    return label.replace("___", " - ").replace("_", " ")


def diagnosis_summary(ident: Identification | None) -> str:
    if ident is None or ident.top is None:
        return "No photo diagnosis is available."
    top = f"{_shown(ident.top.label)} ({ident.top.confidence:.0%} sure)"
    others = "; ".join(f"{_shown(a.label)} ({a.confidence:.0%})" for a in ident.alternatives)
    uncertain = ident.status == "unsure" or ident.top.confidence < CONFIDENT
    text = f"The photo suggests {top}." + (f" Other possibilities: {others}." if others else "")
    return text + (" This is uncertain; say it may be this, never that it is." if uncertain else "")


def _uncertain(ident: Identification | None) -> bool:
    return ident is not None and ident.top is not None and (ident.status == "unsure" or ident.top.confidence < CONFIDENT)


def bounded_history(history: list[dict], cfg: AnswerConfig) -> list[dict]:
    kept = [h for h in history if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)]
    return [{"role": h["role"], "content": h["content"][: cfg.history_chars]} for h in kept[-cfg.history_turns :]]


def build_messages(question: str, ident, history: list[dict], passages: list, cfg: AnswerConfig) -> list[dict]:
    turns = "\n".join(f"{h['role']}: {_clean(h['content'])}" for h in history) or "(none)"
    evidence = "\n".join(f"[{i}] ({p.source}, page {p.page}) {_clean(p.text)}" for i, p in enumerate(passages, 1))
    user = (
        f"<diagnosis>\n{diagnosis_summary(ident)}\n</diagnosis>\n<history>\n{turns}\n</history>\n"
        f"<evidence>\n{evidence}\n</evidence>\n<question>\n{_clean(question)}\n</question>"
    )
    return [{"role": "system", "content": load_prompt(cfg.prompt_version)}, {"role": "user", "content": user}]


def _citation(p, sources: dict[str, SourceInfo] | None) -> Citation:
    info = (sources or {}).get(p.source)
    title, url, note = (info.title, info.url, info.note) if info else (p.source, "", "no link recorded")
    return Citation(p.source, p.page, title, url, note, " ".join(p.text.split())[:200])


def _split_need(reply: str) -> tuple[list[str], str] | None:
    for i, line in enumerate(reply.splitlines()):
        if line.strip().upper().startswith("NEED:"):
            items = [x.strip().lower() for x in line.split(":", 1)[1].split(",") if x.strip()]
            rest = "\n".join(reply.splitlines()[i + 1 :]).strip()
            return items, rest
    return None


def answer(
    question: str,
    *,
    identification: Identification | None,
    history: list[dict],
    retriever,
    llm: LLMClient,
    config: AnswerConfig | None = None,
    crop: str | None = None,
) -> Answer:
    cfg = config or AnswerConfig()
    history = bounded_history(history, cfg)
    version = cfg.prompt_version
    if identification is not None and identification.status in ("not_plant", "invalid"):
        text = identification.guidance or "I couldn't see a plant in that photo. Please send a clear photo of a leaf."
        return Answer(text, [], False, ["photo"], "no_plant", prompt_version=version)
    unsure = identification is not None and identification.status == "unsure"
    if crop is not None:  # the plant the user named; one with no documents of its own uses general guidance
        crop = crop.strip().lower()
        crop = crop if crop in retriever.crops else GENERAL_CROP
    elif identification is not None and identification.top is not None and not unsure:
        crop = crop_of_label(identification.top.label, retriever.crops) or GENERAL_CROP
    asks = ["clearer photo"] if unsure else []
    query = build_query(question, identification, history, cfg.query_mode)
    try:
        found = retriever.search(query, crop=crop, k=cfg.k, mode=cfg.mode, rerank=cfg.rerank)
    except Exception:
        return Answer(SEARCH_FAILED, [], False, asks, "retrieval_failed", query=query, prompt_version=version)
    if not found:
        return Answer(NO_DOCUMENT, [], False, asks, "no_document", query=query, prompt_version=version)
    messages = build_messages(question, identification, history, found, cfg)
    try:
        reply = llm.complete(messages, max_tokens=cfg.max_tokens, timeout_s=cfg.timeout_s).text.strip()
    except LLMError:
        return Answer(UNAVAILABLE, [], False, asks, "unavailable", found, query, version)
    if reply.upper().startswith("NO_SUPPORT"):
        return Answer(NO_DOCUMENT, [], False, asks, "no_document", found, query, version)
    need = _split_need(reply)
    if need is not None:
        items, rest = need
        return Answer(rest or "Could you tell me " + " and ".join(items) + "?", [], False, asks + items, "clarify",
                      found, query, version)
    cited: list[int] = []
    for n in map(int, _MARK.findall(reply)):
        if 1 <= n <= len(found) and n not in cited:
            cited.append(n)
    text = _MARK.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(found) else "", reply).strip()
    citations = [_citation(found[n - 1], cfg.sources) for n in cited]
    state = "answered" if citations else "ungrounded"
    if not citations:
        text = UNGROUNDED + text
    if _uncertain(identification):
        top = identification.top
        text = f"This is only a possible match from the photo, not a certain one ({_shown(top.label)}, {top.confidence:.0%}). " + text
    if CHEMICAL.search(text) and not re.search(r"product label", text, re.I):
        text = f"{text}\n\n{LABEL_NOTE}"
    return Answer(text, citations, bool(citations), asks, state, found, query, version)
