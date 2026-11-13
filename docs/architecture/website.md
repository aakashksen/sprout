# Website

[Back to README](../../README.md) · Previous: [Live demo server](live-demo-server.md) · Start: [System overview](system.md)

`site/` is plain HTML, CSS and JavaScript with no framework and no build step. It is hosted as static files on Cloudflare Pages, published from the `site/` folder.

## Pages

```mermaid
flowchart LR
    HOME["index.html<br/>what it does, a recorded run,<br/>how a check works, contact"] --> TOUR["tour.html<br/>8-step walkthrough of one leaf"]
    HOME --> SPECS["specs.html<br/>architecture diagrams, stack"]
    HOME --> LIVE["live.html<br/>the live demo"]
    TOUR -->|finish| LIVE
    TOUR -->|finish| SPECS
    SPECS --> TOUR
    SPECS --> LIVE
```

## Live demo: on or off

```mermaid
flowchart TD
    LOAD["live.html loads"] --> META{"meta sprout-api"}
    META -->|"empty (the published site)"| OFF["Demo off: shows the<br/>recorded run and the walkthrough"]
    META -->|"same-origin (served by the demo server)"| ON["Demo on: upload or pick one of<br/>10 sample leaves the model never trained on"]
    META -->|"a URL"| REMOTE["Demo on, calling that server"]
```

The published site ships with the demo off; the server rewrites the tag when it serves the page itself.

## Client state and responsiveness

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Reading: photo chosen (generation + 1)
    Reading --> Question: odds and a question
    Reading --> Waiting: odds, question pending
    Waiting --> Question: /api/next returns
    Question --> Updating: tap (button marked, others disabled)
    Updating --> Question: next question
    Updating --> Final: diagnosis
    Final --> AdviceWriting: advice pending
    AdviceWriting --> Final: /api/advice returns
    Final --> Chatting: message sent (typing indicator)
    Chatting --> Final: reply shown
    Reading --> Reading: new photo (older replies dropped)
```

- Every request remembers the check generation it started in; a reply from an older check is dropped, so switching photos never mixes results.
- A tap marks the chosen button at once and disables the rest, then the bars animate to the new odds.
- If the server forgot the check (restart), the page re-sends the photo instead of showing an error.
- Every server string is escaped before it reaches the page.

Related: [Live demo server](live-demo-server.md), [System overview](system.md).
