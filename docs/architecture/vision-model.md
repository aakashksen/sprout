# Vision model

[Back to README](../../README.md) · Previous: [Guide search](guide-search.md) · Next: [Live demo server](live-demo-server.md)

ConvNeXt V2 Tiny, fine-tuned to recognise 29 conditions across 14 plants. The first model, trained on clean lab photos only, fell apart on real field photos; training on lab and field photos together is what fixed it.

## From photos to a served model

```mermaid
flowchart LR
    subgraph Data ["Data"]
        LAB["Lab photos<br/>PlantVillage"]
        FIELD["Field and web photos<br/>PlantDoc, PlantSeg,<br/>PlantWild, Plant Pathology"]
    end
    LAB --> DEDUP["Group near-duplicates<br/>source IDs + perceptual hashes"]
    FIELD --> DEDUP
    DEDUP --> SPLIT["Train / dev / test splits<br/>a group never crosses a split"]
    SPLIT --> TRAIN["Fine-tune ConvNeXt V2 Tiny<br/>PyTorch + timm, CUDA, mixed precision"]
    TRAIN --> PICK["Pick on dev photos<br/>by a rule fixed in advance"]
    PICK --> CAL["Calibrate<br/>temperature scaling,<br/>abstention bar"]
    CAL --> EXP["Export to ONNX"]
    EXP --> SERVE["ONNX Runtime on CPU"]
    TRAIN -. "every run" .-> ML[("MLflow")]
    DEDUP -. "data versions" .-> DVC[("DVC")]
```

## Keeping the test honest

```mermaid
flowchart TD
    P["Every photo"] --> H["Perceptual hash<br/>and its flips and rotations"]
    H --> G{"Same source ID or<br/>a near-identical hash?"}
    G -->|yes| SAME["Same group"]
    G -->|no| NEW["New group"]
    SAME --> S["Groups assigned to splits"]
    NEW --> S
    S --> X["Extra training sources checked against<br/>dev and test with DINOv2 similarity"]
```

Look-alike photos of the same leaf can otherwise sit on both sides of the split and inflate the score.

## Reading one photo

```mermaid
flowchart TD
    IN["Photo bytes"] --> SZ{"Readable, under<br/>20 MB and 60 MP?"}
    SZ -->|no| INV["'Upload a JPEG or PNG'"]
    SZ -->|yes| OR["Turn upright<br/>EXIF orientation"]
    OR --> RES["Resize, normalise<br/>224 px"]
    RES --> M["ConvNeXt V2 forward pass"]
    M --> MASK["Keep the 29 served classes,<br/>renormalise"]
    MASK --> T["Divide logits by the<br/>fitted temperature"]
    T --> GATE{"Photo gate:<br/>confident enough and<br/>not spread across classes?"}
    GATE -->|no| NP["'This doesn't look like a plant leaf'"]
    GATE -->|yes| ODDS["Calibrated odds for 29 conditions"]
```

## Results that shaped the design

| Model | Macro-F1 on held-out field photos |
|---|---|
| Trained on lab photos only | 0.24 |
| DINOv2 with a linear head | 0.72 |
| ConvNeXt V2 Tiny, lab and field photos together | **0.81** |

The right condition is in the top three 96% of the time, and on the photos the model answers at its calibrated bar it is right 95% of the time.

Related: [System overview](system.md), [Question engine](question-engine.md).
