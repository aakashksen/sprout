# E14 adjudication: who is right on the claims the two graders split on

One reader (the verifier, a model), not truth. Method: a seeded script listed each claim with only the passages it cites (as the judge sees them: cited passages, or all if none cited), with no grade. I read the rubric first (`judge.txt`: SUPPORTED if the passages state the claim or clearly imply it, including numbers and names; NOT_SUPPORTED if it adds facts, numbers or names the passages lack, contradicts them, or is about a different plant or disease), wrote my call for every claim, and only then unblinded. "close" means I could argue it either way. Scripts: scratchpad `adj.py`, `unblind.py`.

Passages are the text stored in the blind file, which is cut at about 1,000 characters, often mid-sentence. Both graders saw the same cut text. Several of my "not supported" calls (B1, B6, A3) depend on that cut.

Groups: A = all 29 served-answer splits (28 local supported / Sonnet not, 1 reverse). B = seeded random 20 (seed 2027) of the 113 non-served splits where local said supported and Sonnet not.

| Group | n | I say supported | I say not supported | Local judge right | Sonnet right | Close calls | Clear calls only (n): local / Sonnet |
|---|---|---|---|---|---|---|---|
| A served splits | 29 | 16 | 13 | 15 | 14 | 19 | 10: 5 / 5 |
| B non-served, local kinder | 20 | 6 | 14 | 6 | 14 | 16 | 4: 3 / 1 |

Sonnet "not supported" on the 48 local-kinder claims I read (28 served + 20 sampled): 27 agreed with me (13 + 14), 21 I would have passed. Of those 21:
- 15 in A, 6 in B. "Overly literal" is my label: the passage states or plainly implies the claim, and Sonnet still said no. Clear cases (verbatim or near-verbatim support): A4, A7, A20, A23, A29, B4, B7, B11. The rest are close: a vague antecedent ("These are also recommended"), a small generalization, a clause the cut passage drops, or a hedge.
- The real Sonnet catches I am most sure of: A8 (hardening by cool temperatures "causes catfacing"; the claim recommends it), A15 (six-hour puddling number is not in the cited passage), A16 (cited passage does not give "initial symptoms"), A21 (passage says certified seed, claim says seed treatment), B2 (mottled leaves belong to the mosaic-virus row of the table, not herbicide damage), B10 (tapping technique not in the passage).
- The one served reverse split (A1, local not / Sonnet supported): the passage states it (shot-holes from dried, torn-away tissue on peach). Sonnet right.

Reading:
- On the served set the two graders are about equally right on the split claims (15 vs 14 of 29). Sonnet is not the truth there: roughly half its extra "not supported" calls were reasonable-to-strict readings, and a fifth of them (5 of 29, A4 A7 A20 A23 A29) were plainly wrong.
- On the non-served splits Sonnet was right on 14 of 20 and the local judge on 6. That sample is small (a 95% interval on 14/20 is roughly 0.5 to 0.9) and 16 of the 20 were close, so it points to the local judge being kinder, not to a size.
- Many real "not supported" calls are citation faults: the fact is true and sits in another shown passage but the claim cites a different one (A12, A15, A16, B13, B16). A per-claim check against the cited passages is a citation-accuracy test as much as a truth test. Neither grader looks at passages the claim did not cite.
- Hedges and framing sentences ("Without a photo it is uncertain...", "I need to know the type of plant...", "your diagnosis suggests X with 90% certainty...") are not claims about the passages. B8, B15, B18 are in this group; either grade is defensible and the local judge passes them all. `split_claims` keeps them as claims.

Per-claim table (my call before unblinding; L = local judge, S = Sonnet; 1 = supported; system, question id):

| Tag | System / question | L | S | Me | Close |
|---|---|---|---|---|---|
| A1 | served ambiguous-011 | 0 | 1 | 1 | |
| A2 | served supported-025 | 1 | 0 | 0 | yes |
| A3 | served supported-015 | 1 | 0 | 1 | yes |
| A4 | served supported-038 | 1 | 0 | 1 | |
| A5 | served supported-032 | 1 | 0 | 0 | yes |
| A6 | served ambiguous-019 | 1 | 0 | 0 | yes |
| A7 | served supported-017 | 1 | 0 | 1 | |
| A8 | served supported-040 | 1 | 0 | 0 | |
| A9 | served supported-012 | 1 | 0 | 1 | yes |
| A10 | served supported-039 | 1 | 0 | 1 | yes |
| A11 | served supported-013 | 1 | 0 | 0 | yes |
| A12 | served supported-038 | 1 | 0 | 0 | yes |
| A13 | served supported-036 | 1 | 0 | 0 | yes |
| A14 | served supported-023 | 1 | 0 | 1 | yes |
| A15 | served supported-031 | 1 | 0 | 0 | |
| A16 | served supported-045 | 1 | 0 | 0 | |
| A17 | served ambiguous-017 | 1 | 0 | 1 | yes |
| A18 | served supported-005 | 1 | 0 | 0 | yes |
| A19 | served ambiguous-011 | 1 | 0 | 1 | yes |
| A20 | served supported-046 | 1 | 0 | 1 | |
| A21 | served supported-020 | 1 | 0 | 0 | |
| A22 | served supported-017 | 1 | 0 | 0 | yes |
| A23 | served supported-028 | 1 | 0 | 1 | |
| A24 | served supported-043 | 1 | 0 | 0 | yes |
| A25 | served supported-024 | 1 | 0 | 1 | yes |
| A26 | served ambiguous-009 | 1 | 0 | 1 | yes |
| A27 | served ambiguous-017 | 1 | 0 | 1 | yes |
| A28 | served supported-041 | 1 | 0 | 1 | yes |
| A29 | served supported-021 | 1 | 0 | 1 | |
| B1 | base supported-037 | 1 | 0 | 0 | yes |
| B2 | base ambiguous-005 | 1 | 0 | 0 | yes |
| B3 | faithful ambiguous-017 | 1 | 0 | 0 | yes |
| B4 | combined supported-017 | 1 | 0 | 1 | |
| B5 | abstain supported-010 | 1 | 0 | 1 | yes |
| B6 | ask supported-037 | 1 | 0 | 0 | yes |
| B7 | faithful multi_turn-006 | 1 | 0 | 1 | |
| B8 | abstain ambiguous-011 | 1 | 0 | 0 | yes |
| B9 | abstain supported-034 | 1 | 0 | 1 | yes |
| B10 | base supported-043 | 1 | 0 | 0 | |
| B11 | abstain supported-028 | 1 | 0 | 1 | |
| B12 | faithful multi_turn-013 | 1 | 0 | 0 | yes |
| B13 | abstain supported-009 | 1 | 0 | 0 | yes |
| B14 | ask multi_turn-002 | 1 | 0 | 0 | yes |
| B15 | base multi_turn-011 | 1 | 0 | 0 | yes |
| B16 | abstain supported-041 | 1 | 0 | 0 | yes |
| B17 | base ambiguous-011 | 1 | 0 | 0 | yes |
| B18 | ask ambiguous-011 | 1 | 0 | 0 | yes |
| B19 | base ambiguous-023 | 1 | 0 | 1 | yes |
| B20 | base multi_turn-001 | 1 | 0 | 0 | yes |

## supported-031 (served set): the wrong-disease transfer

Question: "How do I reduce leaf scorch when planting strawberries?" Seven claims; local judge said all seven supported; Sonnet said claim 3 not supported and the other six supported.

- The transfer is in two claims. Claim 5, "Planting on raised beds will also help reduce the problem [5]", and claim 3, "avoiding water puddling for more than six hours [3]". Both come from passage [5], `075-extension-arizona-edu ... az1667-2015-pdf.pdf` page 4. That chunk starts mid-sentence ("...rown in color. Wilted plants eventually collapse and then die. Proper water management is needed to reduce or eliminate this problem. If water puddles for more than six hours then disease spores will be activated and may infect the plants water conducting system... Planting on raised beds will also help"). It is about a wilt disease of the water-conducting system, not leaf scorch. The disease name falls before the cut.
- Claim 5 cites [5] and the passage does say raised beds "will also help", so a check that asks only "do the cited words say this" passes it. Nothing in the chunk names leaf scorch, and the claim says "the problem", which borrows the passage's own vague noun. Both graders passed it.
- Claim 3 cites [3] (the UConn page), but the six-hour rule is in [5]. Sonnet said not supported because six hours is not in [3]. That is the right call for a reason that is partly accidental: it caught the wrong citation, not the wrong disease. The local judge passed it. So DECISIONS' line that Sonnet "graded six of seven supported" is true but leaves out that Sonnet's one "no" sat on half of the transfer.
- Why per-claim checks pass it: the unit checked is claim against cited text. A wrong-disease transfer is a relation between the question's disease and the passage's, which only shows if the grader is told the diagnosis and the chunk shows its disease. Here the chunk is cut before the name, and the judge prompt carries neither the question nor the diagnosis.

## Which grader for final numbers

Recommendation (a judgment from one reader; the choice is the coordinator's): grade final faithfulness with Sonnet, or report both. Reasons: the local judge passed 6 of 20 sampled non-served splits that I would pass, and about half of the served ones; Sonnet's misses ran toward over-literal, so it errs on the safe side for a safety-minded claim, while the local judge errs on the flattering side. Do not report Sonnet's 0.785 as the truth either: about 5 of 29 served splits were plainly wrong "no" calls. Report the two as a range (local 0.877 is an upper bound, Sonnet 0.785 a conservative one), say that Sonnet grading E7 numbers needs the "Claude is never the judge" rule changed, and add the question and diagnosis to the judge prompt before any final run, since neither grader can see a wrong-disease transfer now.
