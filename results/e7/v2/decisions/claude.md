### T6.4 - Claude (Sonnet) against the local writer on 100 dev questions (method v2, 2026-11-03)
- **Chose:** reported as a comparison only; the local writer stays the shipped default.
- **Evidence:** record `E7-20261104T040723229114Z`, judged by gemma4:e4b-it-qat (never Claude), 62 Claude calls, prompt `base+faithful+abstain` with the chosen floors for both writers. local_base_faithful_abstain: faithfulness 0.877, answers 0.783, declines 1.000; claude_sonnet: faithfulness 0.798, answers 0.767, declines 1.000
- **Device:** writer cuda; judge cuda (Claude runs remotely)
- **Would change if:** the person changes the model or the cap.