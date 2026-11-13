# Question engine

[Back to README](../../README.md) · Previous: [Life of a check](check-lifecycle.md) · Next: [Guide search](guide-search.md)

A photo alone often cannot separate look-alike diseases. The question engine asks the fewest yes or no questions that settle it. The rule that shapes everything here: **the LLM writes words, code does the counting.**

## The loop

```mermaid
flowchart TD
    START(["Odds from the photo"]) --> TOP{"Top condition<br/>at or above 93%?"}
    TOP -->|yes| FINAL(["Diagnosis"])
    TOP -->|no| COUNT{"5 questions<br/>already asked?"}
    COUNT -->|yes| FINAL
    COUNT -->|no| PLANT{"Do the suspects<br/>span two plants?"}
    PLANT -->|"yes, and not asked yet"| PQ["Ask which plant"]
    PLANT -->|no| PICK["Pick the question with the<br/>lowest expected uncertainty"]
    PICK --> ANY{"Any question splits<br/>the suspects?"}
    ANY -->|no| FINAL
    ANY -->|yes| ASK["Ask it"]
    PQ --> ANS[/"Answer"/]
    ASK --> ANS
    ANS --> UPD["Update odds in code"]
    UPD --> TOP
```

## Who the suspects are

The candidates are the top three conditions, plus the fourth and fifth if each still holds at least 5%. Questions are only useful if they separate these.

## Updating the odds

```mermaid
flowchart LR
    A["Answer"] --> K{"Which answer?"}
    K -->|"Yes"| Y["Diseases the question<br/>supports x 3"]
    K -->|"No"| N["Diseases the question<br/>supports x 0.35"]
    K -->|"Not sure"| U["No change"]
    K -->|"Plant: Tomato"| PL["Every other plant's<br/>conditions x 0.02"]
    Y --> R["Renormalise to 100%"]
    N --> R
    PL --> R
    U --> R
```

Worked example: early blight at 54%, Septoria 27%, late blight 11%, the rest 8%. A yes to "rings inside the spots, like a target?" multiplies early blight by 3: 162 against 27 + 11 + 8 = 46, so early blight becomes 162 / 208 = 78%.

## Choosing the next question

Every candidate question says which diseases a "yes" supports. For each one, the engine simulates both answers and keeps the question that leaves the least uncertainty on average.

```mermaid
flowchart TD
    Q["Candidate question<br/>supports set S"] --> PY["P(yes) = 0.8 x P(S) + 0.15 x (1 - P(S))"]
    PY --> YES["Odds if yes: S x 3, renormalised"]
    PY --> NO["Odds if no: S x 0.35, renormalised"]
    YES --> HY["Entropy after yes"]
    NO --> HN["Entropy after no"]
    HY --> EX["Expected entropy =<br/>P(yes) x H(yes) + P(no) x H(no)"]
    HN --> EX
    EX --> BEST["Lowest expected entropy wins"]
```

A question is skipped if it was already asked, if it supports none of the current suspects, or if it supports all of them (it could not separate anything).

## Where the questions come from

```mermaid
flowchart LR
    SUS["Suspects"] --> SR["Guide search per suspect<br/>symptoms, this plant, this disease"]
    SR --> P["Numbered passages"]
    P --> LLM["LLM writes 6 questions<br/>JSON: text, supports, cited passages"]
    LLM --> V{"Code checks each one"}
    V -->|"cites a passage of a disease it supports<br/>splits the suspects<br/>ends in '?', 30 words or fewer"| POOL[("Question pool")]
    V -->|fails any check| X["Dropped"]
    SAVED[("52 questions pre-extracted<br/>from the guides")] -.->|"used when the pool<br/>has nothing ready"| PICK2["Picker"]
    POOL --> PICK2
```

Batches are written in the background as soon as the photo is read: one batch for the current suspects and one for each plant group in the running, so answering the plant question never waits for writing.

## When it stops, and what it says

```mermaid
flowchart LR
    T{"Top condition"} -->|"93% or more"| S1["'97% sure'"]
    T -->|"70% to 93%"| S2["'Likely, 85%'<br/>+ try a closer photo"]
    T -->|"under 70%"| S3["'Best guess, 58%'<br/>+ try a closer photo"]
```

The 93% bar is the model's calibrated abstention threshold: on held-out photos, the ones it answers at that bar are right 95% of the time. A "healthy" result needs a stricter bar (96.5%).

## Chat as evidence

```mermaid
flowchart TD
    M["Gardener's message"] --> L["LLM reads it with guide passages"]
    L --> EV["Evidence items:<br/>class, for or against, quoted words"]
    EV --> C{"Quote is 3+ words and<br/>appears in the message?"}
    C -->|no| DROP["Ignored"]
    C -->|yes| F{"For or against?"}
    F -->|for| X2["That class x 2"]
    F -->|against| X5["That class x 0.5"]
    X2 --> RN["Renormalise, show the shift"]
    X5 --> RN
```

A plain question such as "can it be saved?" carries no evidence and never moves the odds.

Related: [Guide search](guide-search.md), [Life of a check](check-lifecycle.md).
