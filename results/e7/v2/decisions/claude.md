### T6.4 - Claude (Sonnet) against the local writer on 100 dev questions (method v2, 2026-11-03)
- **Chose:** reported as a comparison only; the local writer stays the shipped default.
- **Evidence:** record `E7-20261104T012205498018Z`, judged by gemma4:e4b-it-qat (never Claude), 62 Claude calls, prompt `base` with the chosen floors for both writers. local_base: faithfulness 0.799, answers 0.800, declines 0.929; claude_sonnet: faithfulness 0.671, answers 0.800, declines 0.929
- **Device:** writer cuda; judge cuda (Claude runs remotely)
- **Would change if:** the person changes the model or the cap.