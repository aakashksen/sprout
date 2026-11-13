# Live demo server

[Back to README](../../README.md) · Previous: [Vision model](vision-model.md) · Next: [Website](website.md)

`tools/live_demo.py` is a single Flask process that serves the website and the API behind the live demo. It is built to feel instant and to be safe to leave running.

## Endpoints

| Method and path | Body | Returns | Calls the LLM? |
|---|---|---|---|
| `POST /api/check` | photo (multipart) | plant, top 3 odds, first question or `pending` | starts background jobs |
| `POST /api/next` | check id | the next question once one is ready | no |
| `POST /api/answer` | check id, question id, answer or `stop` | new odds, next question or the final answer | no |
| `POST /api/advice` | check id | the final answer with its advice | no (waits for the job) |
| `POST /api/chat` | check id, text | reply, numbered sources, odds shift | yes |
| `GET /*` | | the website; `live.html` is served with the API switched on | no |

## Why taps are instant

```mermaid
sequenceDiagram
    participant B as Browser
    participant S as Request thread
    participant W as Worker threads
    participant L as LLM API

    B->>S: POST /api/check
    S->>W: question batch per plant group
    S->>W: advice for the two likeliest
    S-->>B: odds + first question (no waiting)
    par written while the gardener reads
        W->>L: passages -> 6 questions
        L-->>W: JSON
        W->>W: validate, add to the pool
    and
        W->>L: passages -> advice
        L-->>W: JSON
    end
    B->>S: POST /api/answer
    S->>S: update odds, pick from the pool
    S-->>B: next question in milliseconds
```

## Safety and cost limits

```mermaid
flowchart TD
    REQ["Request"] --> BODY{"JSON object?"}
    BODY -->|no| E400["Clean error"]
    BODY -->|yes| CAP{"Spend guard"}
    CAP -->|"over a cap and the step<br/>needs the LLM"| E429["'Today's allowance is used'"]
    CAP -->|"over a cap, step needs no LLM<br/>(answer, stop, next)"| OK
    CAP -->|under| OK["Continue"]
    OK --> RATE{"Photos from this IP<br/>in the last minute > 12?"}
    RATE -->|yes| E429b["'Wait a moment'"]
    RATE -->|no| LOCK["Per-check lock:<br/>one answer or chat turn at a time"]
    LOCK --> RUN["Handle the request"]
    RUN --> CHARGE["Every LLM call is charged<br/>before its reply is parsed"]
```

| Limit | Value |
|---|---|
| Spend per visitor (signed session cookie) | $1 a day |
| Spend per IP address | $3 a day |
| Spend for the whole server | $10 a day |
| Photos per IP | 12 a minute |
| A check is kept for | 1 hour, at most 2000 at once |
| LLM call timeout | 30 s, one retry |
| Upload size | 20 MB |

Photos are read in memory and discarded. The API key is read from `.env` and never sent to the browser. If no key is set, questions come from the 52 saved guide questions and the chat is off.

## Concurrency model

```mermaid
flowchart LR
    subgraph Check ["One check"]
        STEP["step lock<br/>answers, stop, chat"]
        POOLL["pool lock<br/>question pool"]
    end
    REQ1["Answer request"] --> STEP
    REQ2["Duplicate answer"] -->|"waits, then gets 409"| STEP
    WK["Worker adding questions"] --> POOLL
    STEP --> POOLL
    SL["Search lock<br/>shared retriever"] --- WK
```

A repeated answer (double click, retry, second tab) is applied once; the copy gets a JSON 409 with the current state.

Related: [Life of a check](check-lifecycle.md), [Website](website.md).
