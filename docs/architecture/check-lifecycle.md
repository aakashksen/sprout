# Life of a check

[Back to README](../../README.md) · Previous: [System overview](system.md) · Next: [Question engine](question-engine.md)

One "check" is one photo, the questions that follow it, the diagnosis and any chat. These diagrams follow a real run: a tomato leaf with late blight that the photo alone read as potato late blight.

## 1. Photo in, odds out (about 0.4 s)

```mermaid
sequenceDiagram
    autonumber
    actor G as Gardener
    participant B as Browser
    participant S as Flask server
    participant I as Identifier
    participant E as Odds engine
    participant W as Background workers

    G->>B: picks a photo
    B->>B: shows the photo and "Reading the leaf..."
    B->>S: POST /api/check (photo)
    S->>S: spend guard and per-IP photo limit
    S->>I: decode, turn upright, gate, model
    I-->>S: odds for 29 conditions
    S->>E: new check
    E->>W: write question batches and advice (async)
    E->>E: suspects span two plants, so ask the plant
    S-->>B: plant, top 3 odds, "Is this a potato or a tomato plant?"
    B-->>G: odds bars and the question
```

## 2. Answering (median 6.5 ms per tap)

```mermaid
sequenceDiagram
    autonumber
    actor G as Gardener
    participant B as Browser
    participant S as Flask server
    participant E as Odds engine
    participant P as Question pool

    G->>B: taps "Tomato"
    B->>B: marks the button, disables the others
    B->>S: POST /api/answer (check, question, "tomato")
    S->>E: lock the check, apply the answer
    E->>E: other plants x0.02, renormalise: tomato late blight 90%
    E->>P: best question for the suspects left
    alt a written question is ready
        P-->>E: question with its guide source
    else the batch is still being written
        E->>E: use a saved question from the guides
    end
    S-->>B: new odds and the next question
    B-->>G: bars slide, question appears
```

## 3. The answer, with advice that was written early

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant S as Flask server
    participant E as Odds engine
    participant A as Advice job

    B->>S: POST /api/answer ("yes")
    S->>E: apply: tomato late blight 97%
    E->>E: 97% passes the 93% bar, so stop
    E->>A: advice for tomato late blight ready?
    alt already written
        A-->>E: advice + guide source
        S-->>B: final answer with advice
    else still writing
        S-->>B: final answer, advice pending
        B->>S: POST /api/advice
        S->>A: wait for it
        A-->>S: advice + guide source
        S-->>B: advice fills in
    end
```

## 4. Chat that can move the odds

```mermaid
sequenceDiagram
    autonumber
    actor G as Gardener
    participant B as Browser
    participant S as Flask server
    participant R as Retriever
    participant L as LLM API
    participant E as Odds engine

    G->>B: "The lowest leaves are turning yellow"
    B->>S: POST /api/chat
    S->>R: search: diagnosis + message, this plant
    R-->>S: guide passages, plus management passages
    S->>L: passages, conversation so far, candidates
    L-->>S: reply, cited passages, evidence quoting the gardener
    S->>E: evidence counts only if it quotes the gardener's own words
    E->>E: early blight 94% to 97%
    S-->>B: reply with numbered sources, "Odds updated"
```

## States a check moves through

```mermaid
stateDiagram-v2
    [*] --> Reading: photo posted
    Reading --> NotPlant: gate refuses
    Reading --> Invalid: unreadable file
    Reading --> Asking: odds ready
    Asking --> Asking: answer, still under the bar
    Asking --> Pending: no question ready yet
    Pending --> Asking: batch written
    Asking --> Final: bar reached, 5 questions, or nothing splits the suspects
    Asking --> Final: visitor skips
    Final --> Final: chat turns
    Final --> [*]: expires after 1 hour
    NotPlant --> [*]
    Invalid --> [*]
```

Related: [Question engine](question-engine.md), [Live demo server](live-demo-server.md).
