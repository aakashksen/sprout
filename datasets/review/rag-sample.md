# RAG evaluation set: review sample (#53)

Check each gold label against the source page. Mark any entry that is wrong or unclear.

### supported-001 (supported, dev)

- question: What do early blight lesions look like on potato leaves?
- expected: answer
- gold: potato.pdf, page 1
- answer terms: concentric rings
- passage: ...s range in size from a pinpoint to 1/4 inch. figure 2 . early blight lesions are characterized by an alternating series of light tan and dark-colored concentric rings surrounded by a narrow band of chlorotic tissue....

### supported-014 (supported, dev)

- question: When should spraying for potato early blight begin?
- expected: answer
- gold: potato.pdf, page 5
- answer terms: first sign of disease, row closure
- passage: ...oliar fungicides is not necessary in plants at the vegetative stage, when they are relatively resistant. accordingly, spraying should commence at the first sign of disease or at row closure (between rows). the frequency of subsequent sprays should be determined according to the genotype and age-related resistance of the cultivar. protectant fungicides should be applied initially at relatively long...

### supported-028 (supported, dev)

- question: How do strawberry leaves look as leaf scorch progresses?
- expected: answer
- gold: strawberry.pdf, page 2
- answer terms: purplish, scorched
- passage: ...(mycosphaerella fragariae). the blotches coalesce irregularly when numerous and tissue between the blotches turns purplish to bright red. as the disease progresses, leaves turn brown, dry out and turn up at the margins, assuming a burnt or "scorched" appearance, as indicated in the name leaf scorch. leaf stems (petioles): lesions are typically elongated, sunken,...

### supported-041 (supported, dev)

- question: Can a tomato virus infection be cured with chemicals?
- expected: answer
- gold: tomatodoc1.md, page 1
- answer terms: cannot be cured
- passage: ... prior to the development of symptoms and in conjunction with appropriate cultural practices. viruses: once plants become infected with a virus, they cannot be cured. chemicals are not an effective management option for virus infections. many viruses are transmitted by insects, such as aphids (tmv, cmv, tev), thrips (tswv), and whiteflies (tylcv), and can cause significant losses in tomatoes. othe...

### supported-055 (supported, test)

- question: Where has strobilurin resistance in early blight been reported?
- expected: answer
- gold: Potato - early-blight-bulletin.pdf, page 5
- answer terms: strobilurin, Michigan
- passage: ...thalonil and triphenyl tin hydroxide) are also effective against early blight when applied at approximately 7- to 10-day intervals. resistance to the strobilurin group of fungcides (group 11; table 1) has been reported in michigan. the geographical spread of this resis- tance is not known, but applications of strobilurins should be made in combination with tank mixtures of the fungicides listed ab...

### supported-068 (supported, test)

- question: What is the most critical period for yield protection in corn?
- expected: answer
- gold: S-0003-45-Corn.pdf, page 3
- answer terms: critical period, R2, R5
- passage: ...low ear leaf and higher leaves at this stage, scout again at r2 growth stage. if the current weather is wet and warm, scout every four days. the most critical period for yield protection is between r2 and r5 (dent). diseases occurring past r5 will cause minimal yield loss. protection of ear leaf and leaves above the ear from fungal infection protects against yield loss. scout early and continue sc...

### not_in_documents-001 (not_in_documents, dev)

- question: How do I treat leaf spot on my magnolia tree?
- expected: abstain
- not in the documents: magnolia

### not_in_documents-005 (not_in_documents, dev)

- question: How do I manage Panama disease in bananas?
- expected: abstain
- not in the documents: banana

### not_in_documents-010 (not_in_documents, test)

- question: How do I save a pineapple with heart rot?
- expected: abstain
- not in the documents: pineapple

### not_in_documents-015 (not_in_documents, dev)

- question: How do I treat celery late blight?
- expected: abstain
- not in the documents: celery

### not_in_documents-019 (not_in_documents, dev)

- question: How do I treat fruit rot in cranberry bogs?
- expected: abstain
- not in the documents: cranberry

### not_in_documents-024 (not_in_documents, test)

- question: How do I treat downy mildew on basil?
- expected: abstain
- not in the documents: basil

### wrong_crop-001 (wrong_crop, dev)

- question: My grape leaves have early blight lesions with concentric rings. How do I treat early blight?
- expected: crop_mismatch
- stated crop: grape
- the disease is documented for another crop in: potato.pdf

### wrong_crop-005 (wrong_crop, dev)

- question: My blueberry leaves have early blight rings. What should I do?
- expected: crop_mismatch
- stated crop: blueberry
- the disease is documented for another crop in: potato.pdf

### wrong_crop-010 (wrong_crop, test)

- question: My apple leaves have leaf scorch caused by Diplocarpon earliana. How do I manage it?
- expected: crop_mismatch
- stated crop: apple
- the disease is documented for another crop in: strawberry.pdf

### wrong_crop-014 (wrong_crop, test)

- question: Do I remove debris at renovation to control leaf scorch on my blueberries?
- expected: crop_mismatch
- stated crop: blueberry
- the disease is documented for another crop in: strawberry.pdf

### wrong_crop-019 (wrong_crop, dev)

- question: Can citrus greening spread to my apple tree and how do I manage it?
- expected: crop_mismatch
- stated crop: apple
- the disease is documented for another crop in: Orangeax739e.pdf

### wrong_crop-023 (wrong_crop, dev)

- question: Cedar apple rust has appeared on my corn. How do I control it?
- expected: crop_mismatch
- stated crop: corn
- the disease is documented for another crop in: Apple.pdf

### ambiguous-001 (ambiguous, dev)

- question: My plant has yellow leaves. What is it?
- expected: clarify
- missing detail: crop

### ambiguous-005 (ambiguous, dev)

- question: The leaves are curling and I do not know why.
- expected: clarify
- missing detail: crop

### ambiguous-009 (ambiguous, dev)

- question: Leaves are turning black at the edges.
- expected: clarify
- missing detail: crop

### ambiguous-014 (ambiguous, test)

- question: The spots on the leaf have a yellow ring. What is it?
- expected: clarify
- missing detail: crop

### ambiguous-018 (ambiguous, test)

- question: What does a disease look like on a plant?
- expected: clarify
- missing detail: crop, symptoms

### ambiguous-022 (ambiguous, test)

- question: My leaves look scorched but it has not been hot.
- expected: clarify
- missing detail: crop

### multi_turn-001 (multi_turn, dev)

- user: My potato leaves have dark brown spots with rings on the lower leaves.
- assistant: That matches early blight, caused by Alternaria solani.
- question: How does it come back each year?
- expected: answer
- stated crop: potato
- gold: potato.pdf, page 3
- answer terms: plant debris, tubers
- passage: ...issues. disease cycle the dark-colored spores (figure 7) and mycelia of the pathogen (alternaria solani ) survive between growing seasons in infested plant debris and soil, in infected potato tubers, and in overwintering debris of susceptible solanaceous crops and weeds, including hairy nightshade ( solanum sarrachoides ). overwintering spores and mycelia of a. solani are melanized (darkly pigment...

### multi_turn-005 (multi_turn, dev)

- user: My pepper leaves have water-soaked spots on the underside.
- assistant: That sounds like bacterial spot.
- question: What does it do to the fruit?
- expected: answer
- stated crop: pepper
- gold: PP362_PP362_ Bacterial Spot of Pepper.pdf, page 2
- answer terms: warts, scabs
- passage: ...all, raised blemishes that are a slightly lighter green than the normal fruit color. fruit lesions enlarge, turn brown to black, and appear as raised warts or scabs (figure 4). under humid conditions, other microbes, such as those causing bacterial soft rot, may enter the fruit lesions and cause massive fruit decay. figure 1. initial symptoms of bacterial spot on pepper leaves exhibiting water-soa...

### multi_turn-009 (multi_turn, dev)

- user: My strawberry leaves have purplish blotches that look scorched.
- assistant: That fits leaf scorch from Diplocarpon earliana.
- question: How long after infection do spots show up?
- expected: answer
- stated crop: strawberry
- gold: strawberry.pdf, page 3
- answer terms: 6-15 days
- passage: ...r and probably by arthropods. conidia directly penetrate the cuticle, and develop into a subcuticular intercellular mycelium. lesions begin to appear 6-15 days after infection at . conidia directly penetrate the cuticle, and develop into a subcuticular intercellular mycelium. lesions begin to appear 6-15 days after infection at favorable temperatures (59-86 °f) provided a post infection wetness pe...

### multi_turn-014 (multi_turn, dev)

- user: I am planning how to manage corn diseases.
- assistant: Several cultural practices help.
- question: Why does rotating crops help?
- expected: answer
- stated crop: corn
- gold: Corn1.pdf, page 3
- answer terms: Crop rotation, reduced tillage
- passage: ...”) is not as easily lost by the host, because a pathogen is less likely to overcome the resistance conferred by the combined effect of several genes. crop rotation crop rotation is a host, because a pathogen is less likely to overcome the resistance conferred by the combined effect of several genes. crop rotation crop rotation is a sound cultural practice that also aids in the control of many dise...

### multi_turn-018 (multi_turn, test)

- user: When do I start checking potato fields for early blight?
- assistant: Regular inspection is recommended.
- question: At what height of the plants?
- expected: answer
- stated crop: potato
- gold: Potato - early-blight-bulletin.pdf, page 5
- answer terms: 12 inches
- passage: ....org. the fi rst application for early blight control should be timed at 200 p days after emergence. regular inspection of fi elds after plants reach 12 inches in height is ication for early blight control should be timed at 200 p days after emergence. regular inspection of fi elds after plants reach 12 inches in height is recommended to detect early infections. table 1. product name, fraca resist...

### multi_turn-022 (multi_turn, test)

- user: My peach leaves have shot holes from bacterial spot.
- assistant: Bacterial spot also affects fruit.
- question: What does the fruit look like?
- expected: answer
- stated crop: peach
- gold: BacterialSpotPeach.pdf, page 1
- answer terms: pitting, cracking
- passage: ...lar nature of the lesions and by the fruit symptoms. fruit symptoms initially appear as small circular brown spots on the surface of the fruit. later pitting and cracking may occur around the spots. although this ptoms initially appear as small circular brown spots on the surface of the fruit. later pitting and cracking may occur around the spots. although this symptom damages the appearance of th...
