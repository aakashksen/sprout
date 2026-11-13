# Guide search

[Back to README](../../README.md) · Previous: [Question engine](question-engine.md) · Next: [Vision model](vision-model.md)

Every question, piece of advice and chat reply starts from passages in 237 university extension guides. The search has to find the passage about *this* disease on *this* plant, not a generic one.

## Building the index (offline)

```mermaid
flowchart LR
    SRC["237 guides<br/>PDF and web pages"] --> EXT["Text extraction<br/>pypdf, HTML to text"]
    EXT --> MAN[("Manifest<br/>title, publisher, URL,<br/>crop, diseases")]
    EXT --> CH["Chunking<br/>passages with page numbers"]
    MAN --> TAG["Tag each passage<br/>with crop and diseases"]
    CH --> TAG
    TAG --> CHUNKS[("chunks.jsonl")]
    CHUNKS --> BM["BM25 keyword index"]
    CHUNKS --> EMB["BGE-small embeddings<br/>fastembed, cosine"]
    EMB --> VEC[("Cached vectors<br/>keyed by model and text")]
```

A guide that covers several plants is tagged with each plant its title or disease list names, so a pepper question can still reach a page filed under tomato.

## One search

```mermaid
flowchart TD
    Q["Query + plant + disease"] --> F1{"Filter passages<br/>to this plant and disease"}
    F1 --> D["Dense scores<br/>embedding similarity"]
    F1 --> K["Keyword scores<br/>BM25"]
    D --> H["Hybrid merge"]
    K --> H
    H --> RR["Cross-encoder reranker<br/>MiniLM, reads query and passage together"]
    RR --> FL{"Above the<br/>score floor?"}
    FL -->|yes| OUT["Top passages with source and page"]
    FL -->|"none for this plant"| GEN["Fall back to general guidance"]
    GEN --> OUT
```

## Why the reranker

The hybrid merge finds the right neighbourhood; the reranker, which reads the question and the passage together, puts the right passage first. On the evaluation questions, the right passage lands in the top five 83% of the time with reranking, against 72% for embedding search alone.

## How each feature uses it

```mermaid
flowchart LR
    subgraph Uses ["Callers"]
        QW["Question writer"]
        AW["Advice writer"]
        CH["Chat"]
    end
    QW -->|"'disease symptoms on leaves'<br/>per suspect, 3 passages"| R["Retriever"]
    AW -->|"'how to manage and control disease'<br/>4 passages"| R
    CH -->|"'disease: the gardener's message'<br/>3 passages + 2 management passages"| R
    R --> SRC2["Each passage keeps its guide title<br/>and publisher for citation"]
```

Searches are serialised behind one lock because the retriever and its models are shared by all background workers.

Related: [Question engine](question-engine.md), [System overview](system.md).
