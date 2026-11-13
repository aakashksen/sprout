# System overview

[Back to README](../../README.md) · Next: [Life of a check](check-lifecycle.md)

SPROUT has four parts: a static website, one Python server, a set of background threads inside that server, and an external LLM API that only writes text. A separate offline pipeline produces the model and the guide index the server loads at start-up.

## Where everything runs

```mermaid
flowchart TB
    subgraph Browser ["Browser"]
        P["Pages<br/>home, walkthrough, how it's built"]
        LD["Live demo client<br/>live.html + JavaScript"]
    end

    subgraph Server ["Flask server, one Python process"]
        API["HTTP API<br/>/api/check /api/next /api/answer<br/>/api/advice /api/chat"]
        GUARD["Spend guard<br/>visitor, IP and daily caps"]
        ID["Identifier<br/>photo gate + ONNX ConvNeXt V2"]
        ENG["Odds engine<br/>updates, question choice, stop rule"]
        RET["Retriever<br/>BM25 + embeddings + reranker"]
        STORE[("Checks in memory<br/>expire after 1 hour")]
    end

    subgraph Workers ["Background thread pool, 8 workers"]
        QW["Question writer"]
        AW["Advice writer"]
    end

    LLM["LLM API<br/>returns JSON text"]

    subgraph Offline ["Offline, built once"]
        TR["Training on a CUDA GPU"]
        IX["Guide ingest and indexing"]
    end

    LD -->|JSON over HTTP| API
    API --> GUARD --> ENG
    API --> ID --> ENG
    ENG <--> STORE
    ENG -->|suspects| QW
    ENG -->|likely diagnoses| AW
    QW --> RET
    AW --> RET
    QW <-->|passages in, questions out| LLM
    AW <-->|passages in, advice out| LLM
    API -->|chat| RET
    TR -->|model.onnx + calibration| ID
    IX -->|chunks + vectors| RET
```

## Who owns what

| Part | Owns | Never does |
|---|---|---|
| Browser | Rendering, the current check id, optimistic button states | Hold odds or decide anything |
| Flask server | Every check's odds, history and questions | Trust anything the browser computes |
| Odds engine | Probabilities, the next question, when to stop | Call the LLM |
| Background workers | Writing questions and advice ahead of time | Change the odds |
| LLM API | Wording questions, advice and chat replies as JSON | Produce numbers that reach the user |
| Retriever | Finding guide passages for a plant and a disease | Generate text |

## Data that crosses each boundary

```mermaid
flowchart LR
    B["Browser"] -- "photo (multipart)" --> S["Server"]
    B -- "check id + answer" --> S
    B -- "check id + chat text" --> S
    S -- "plant, top 3 odds, question, final answer" --> B
    S -- "guide passages + instructions" --> L["LLM API"]
    L -- "JSON: questions, advice, reply, evidence" --> S
```

The photo is read in memory and never written to disk. The LLM never sees the photo; it sees guide passages, the candidate disease names, and the visitor's chat text.

## Two web apps

```mermaid
flowchart LR
    subgraph Demo ["tools/live_demo.py"]
        D1["Serves the website and its API"]
        D2["LLM API for text, spend caps"]
    end
    subgraph Full ["src/sprout/web"]
        F1["Accounts, upload history, chat"]
        F2["Local LLM through Ollama"]
    end
    Core["Shared core: Identifier, Retriever, guide manifest"]
    Demo --> Core
    Full --> Core
```

Both apps load the same identifier and retriever from `src/sprout`, so a photo gets the same reading in either.

Related: [Live demo server](live-demo-server.md), [Website](website.md).
