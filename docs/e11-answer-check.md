# E11 answer check and wording sensitivity

Computed by `tools/e11_answer_check.py` from the saved answers, `plan.json`, the manifest and the sheet key. It changes no answer and no record. The class labels are used only to describe the answers after they were saved. Development numbers on cleaned `field_dev`.

## Does the answerer say yes more to the true class?

Yes rate by kind of question, over all 6,190 saved answers. A decoy is a question about a class outside the photo's top 3; a candidate is a question the flow could ask. The true class is the photo's label.

| kind | answers | yes rate | yes rate of answered | unknown rate |
|---|---|---|---|---|
| decoy | 2630 | 0.030 | 0.037 | 0.206 |
| candidate, wrong class | 2766 | 0.078 | 0.097 | 0.193 |
| candidate, true class | 794 | 0.233 | 0.320 | 0.272 |

## Sensitivity to the first-attempt wording (sheets s001-s070)

For this report only, not a new selection. 704 of 6,190 answers differ when s001-s070 use the first attempt's answers (kept in the git-ignored `datasets/followup/attempt1/`); the flow is run on them with nothing written. E11's records use the one-wording answers.

| answers | view | system | unknown rate | better | worse | whole view accuracy | macro-F1 | unsure photos accuracy |
|---|---|---|---|---|---|---|---|---|
| saved (one wording) | seen | joint | 0.272 | 20 | 26 | 0.7695 -> 0.7636 [-0.0185, +0.0068] | 0.7712 -> 0.7618 | 0.5308 -> 0.5103 |
| saved (one wording) | heldout_plantdoc | joint | 0.228 | 11 | 13 | 0.7525 -> 0.7492 [-0.0196, +0.0131] | 0.7398 -> 0.7394 | 0.5000 -> 0.4888 |
| saved (one wording) | heldout_plantdoc | heldout | 0.161 | 30 | 23 | 0.4377 -> 0.4492 [-0.0115, +0.0347] | 0.3789 -> 0.3947 | 0.2953 -> 0.3158 |
| saved (one wording) | heldout_plantseg | joint | 0.342 | 9 | 13 | 0.7943 -> 0.7847 [-0.0313, +0.0121] | 0.7757 -> 0.7703 | 0.5789 -> 0.5439 |
| saved (one wording) | heldout_plantseg | heldout | 0.324 | 13 | 14 | 0.5478 -> 0.5455 [-0.0269, +0.0235] | 0.5182 -> 0.5345 | 0.3904 -> 0.3850 |
| attempt 1 on s001-s070 | seen | joint | 0.164 | 20 | 29 | 0.7695 -> 0.7607 [-0.0222, +0.0048] | 0.7712 -> 0.7605 | 0.5308 -> 0.5000 |
| attempt 1 on s001-s070 | heldout_plantdoc | joint | 0.135 | 11 | 14 | 0.7525 -> 0.7475 [-0.0213, +0.0115] | 0.7398 -> 0.7386 | 0.5000 -> 0.4831 |
| attempt 1 on s001-s070 | heldout_plantdoc | heldout | 0.108 | 33 | 22 | 0.4377 -> 0.4557 [-0.0049, +0.0425] | 0.3789 -> 0.4015 | 0.2953 -> 0.3275 |
| attempt 1 on s001-s070 | heldout_plantseg | joint | 0.211 | 9 | 15 | 0.7943 -> 0.7799 [-0.0378, +0.0095] | 0.7757 -> 0.7681 | 0.5789 -> 0.5263 |
| attempt 1 on s001-s070 | heldout_plantseg | heldout | 0.214 | 13 | 21 | 0.5478 -> 0.5287 [-0.0477, +0.0096] | 0.5182 -> 0.5216 | 0.3904 -> 0.3476 |
