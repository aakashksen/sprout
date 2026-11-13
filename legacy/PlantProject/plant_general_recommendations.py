# plant_general_recommendations.py
# General care recommendations for each supported plant/disease class.
# These are defaults and can be updated as needed.

PLANT_GENERAL_RECOMMENDATIONS = {
    'Apple___Apple_scab': {
        'summary': "Looks like Apple Scab, a common fungal foe! 🍄 Don't worry, we can manage it. Focus on cleaning up fallen leaves/fruit and improving airflow. Fungicides can help too!",
        'possible_reasons': [
            "🌧️ Fungal spores thriving in wet, cool spring weather",
            "🍂 Infected leaves or fruit left on the ground from last season"
        ],
        'tips': [
            "🧹 Remove and destroy fallen leaves and fruit promptly (don't compost!)",
            "✂️ Prune your apple tree to open up the canopy for better air circulation",
            "🛡️ Apply appropriate fungicides starting early in the season if scab is a recurring issue",
            "🍎 Consider planting scab-resistant apple varieties in the future",
            "💧 Avoid overhead watering; water at the base"
        ]
    },
    'Apple___Black_rot': {
        'summary': "Oh no, Black Rot! ⚫ This fungus affects fruit, leaves, and branches. Sanitation is key! Remove infected parts and keep things airy. Fungicides might be needed.",
        'possible_reasons': [
            "🤢 Infected 'mummy' fruit or dead wood left in the tree or on the ground",
            "🌡️ Warm, humid, and wet weather conditions favouring the fungus"
        ],
        'tips': [
            "✂️ Prune out dead or diseased branches and remove infected fruit ('mummies')",
            "🧹 Clean up fallen leaves and fruit around the tree base",
            "💨 Improve air circulation through proper pruning",
            "✨ Sanitize pruning tools between cuts, especially after cutting diseased wood",
            "🛡️ Apply fungicides during bloom and early fruit development if necessary"
        ]
    },
    'Apple___Cedar_apple_rust': {
        'summary': "Seeing orange spots? It might be Cedar Apple Rust! 🍊 This tricky fungus needs both apple/crabapple and cedar/juniper trees to survive. Management involves breaking this cycle.",
        'possible_reasons': [
            "🌲 Nearby cedar or juniper trees hosting the other stage of the fungus",
            "💦 Wet spring weather splashing spores onto apple leaves/fruit"
        ],
        'tips': [
            "👋 If possible, remove nearby wild cedar or juniper hosts (within a mile or two helps!)",
            "✂️ Prune out any visible galls (swellings) on apple twigs if practical",
            "🛡️ Apply fungicides protectively, starting at bud break, especially during wet springs",
            "🍎 Plant rust-resistant apple varieties if available and suitable",
            "🍂 Rake and destroy fallen apple leaves in autumn"
        ]
    },
    'Apple___healthy': {
        'summary': "Your apple tree looks fantastic! ✨ Keep up the great work. Consistent care is the secret to happy, productive trees.",
        'possible_reasons': [
            "✅ You're providing great care!",
            "☀️ Good growing conditions (sun, water, soil)",
            "🚫 Lucky break! No major pests or diseases currently active"
        ],
        'tips': [
            "💧 Water regularly, especially during fruit development and dry spells",
            "🍽️ Fertilize appropriately in early spring based on soil needs",
            "✂️ Prune annually during dormancy to maintain shape, remove dead wood, and encourage fruiting",
            "👀 Keep monitoring regularly for any signs of pests or disease",
            "🍎 Harvest fruit at the right time for best flavour!"
        ]
    },
    'Background_without_leaves': {
        'summary': "Hmm, I can't seem to find a plant in this picture. 🤔 Could you try uploading a clearer photo focused on a plant leaf?",
        'possible_reasons': [
            "❓ The image doesn't contain a plant or leaf",
            "🌫️ The image quality might be too low (blurry, dark, too far away)",
            "🤷 The condition might be something our system doesn't recognize yet",
            "⚙️ A temporary technical hiccup might have occurred"
        ],
        'tips': [
            "📸 Ensure your photo features a single plant or leaf clearly",
            "💡 Take the picture in good, natural light if possible",
            "🎯 Focus specifically on the affected area if diagnosing a problem",
            "🖼️ Make sure the plant part fills a good portion of the frame",
            "🔄 Try again, perhaps with a different angle or image format (JPG/PNG)"
        ]
    },
    'Blueberry___healthy': {
        'summary': "Your blueberry bush looks berry happy and healthy! 🫐 Keep providing the right conditions, and you'll be rewarded.",
        'possible_reasons': [
            "✅ Great care practices!",
            "☀️ Suitable acidic soil, good sunlight, and proper watering",
            "🚫 No current pest or disease pressures"
        ],
        'tips': [
            "💧 Water consistently, especially during dry periods (blueberries have shallow roots!)",
            "🌲 Maintain a layer of acidic mulch (like pine needles or peat moss) to conserve moisture and suppress weeds",
            "✂️ Prune annually after harvest (or in late winter) to remove old canes and encourage new growth",
            "🧪 Test soil pH occasionally; blueberries love acidic soil (pH 4.5-5.5)",
            "👀 Monitor for common pests (like birds or fruit worms) and diseases"
        ]
    },
    'Cherry___Powdery_mildew': {
        'summary': "Looks like Powdery Mildew on your cherry! 🍄 It's that typical white, dusty coating on leaves. Improving airflow and managing moisture are key steps.",
        'possible_reasons': [
            "☁️ High humidity combined with moderate temperatures",
            "💨 Poor air circulation within the tree canopy",
            "🌿 Dense foliage creating a humid microclimate"
        ],
        'tips': [
            "✂️ Prune to open up the tree canopy and improve air movement",
            "🍂 Remove and dispose of heavily infected leaves if practical (don't compost!)",
            "💧 Avoid overhead watering; water the soil directly, preferably in the morning",
            "☀️ Ensure the tree gets adequate sunlight",
            "🛡️ Consider fungicides (like sulfur or potassium bicarbonate based) if the infection is severe, following label directions"
        ]
    },
    'Cherry___healthy': {
        'summary': "Your cherry tree looks cheerful and healthy! 🍒 Keep nurturing it for those delicious future harvests.",
        'possible_reasons': [
            "✅ Consistent and appropriate care",
            "☀️ Good site selection (sunlight, drainage)",
            "🚫 Absence of significant pest or disease issues"
        ],
        'tips': [
            "💧 Water regularly, especially when young or during dry spells",
            "🍽️ Fertilize lightly in early spring if needed (avoid over-fertilizing)",
            "✂️ Prune during dormancy to establish a strong structure and remove dead/crossing branches",
            "👀 Watch for common cherry pests (like aphids or birds) and diseases (like brown rot or bacterial canker)",
            "❄️ Protect young trees from winter sunscald if applicable"
        ]
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'summary': "Seeing rectangular, gray spots on your corn? 🌽 That's likely Gray Leaf Spot (Cercospora). Management focuses on reducing residue and choosing resistant varieties.",
        'possible_reasons': [
            "🍂 Fungal spores surviving in old corn debris left in the field",
            "🌧️ Warm, humid, and overcast weather conditions",
            "🌽 Planting susceptible corn hybrids"
        ],
        'tips': [
            "♻️ Rotate crops – avoid planting corn in the same spot year after year",
            "🚜 Till under or remove corn residue after harvest to reduce fungal survival",
            "🧬 Choose resistant corn hybrids if available for your area",
            "🛡️ Fungicides can be effective but need timely application based on scouting and disease pressure",
            "💨 Ensure good field drainage and avoid overly dense planting if possible"
        ]
    },
    'Corn___Common_rust': {
        'summary': "Those reddish-brown pustules on your corn look like Common Rust! 🌽 Usually, it's more cosmetic than yield-damaging, especially if it appears late.",
        'possible_reasons': [
            "💨 Spores blown in from other areas (often southern regions)",
            "🌡️ Moderate temperatures and high humidity/dew",
            "🌽 Planting susceptible corn varieties"
        ],
        'tips': [
            "🧬 Plant resistant or tolerant corn hybrids – this is the best defense!",
            "🗓️ Early planting can sometimes help the corn outgrow the peak infection period",
            "🛡️ Fungicides are rarely economical for common rust alone, but monitor if other diseases are present",
            "♻️ Crop rotation helps with many diseases, though rust spores travel far"
        ]
    },
    'Corn___Northern_Leaf_Blight': {
        'summary': "Those long, cigar-shaped, grayish lesions suggest Northern Leaf Blight on your corn. 🌽 Reducing residue and resistant hybrids are key management tools.",
        'possible_reasons': [
            "🍂 Fungus surviving in infected corn debris from the previous season",
            "🌧️ Moderate temperatures and prolonged periods of leaf wetness (rain, dew)",
            "🌽 Planting susceptible corn hybrids"
        ],
        'tips': [
            "♻️ Rotate crops away from corn for at least one year",
            "🚜 Manage residue: Tillage can help bury infected debris",
            "🧬 Select resistant corn hybrids specifically rated for NLB tolerance",
            "🛡️ Fungicides can be effective if applied early, based on scouting and disease risk",
            "💨 Avoid overly dense planting to promote faster leaf drying"
        ]
    },
    'Corn___healthy': {
        'summary': "Your corn looks amazing! 🌽 Strong stalks and healthy leaves. Keep providing what it needs for a great harvest.",
        'possible_reasons': [
            "✅ Excellent care and management",
            "☀️ Optimal growing conditions (sun, water, nutrients)",
            "🧬 Planting healthy, vigorous hybrids",
            "🚫 Low pest and disease pressure"
        ],
        'tips': [
            "💧 Ensure consistent watering, especially during pollination and grain fill stages",
            "🍽️ Provide adequate nitrogen and other nutrients based on soil tests or plant needs",
            "🐜 Monitor for pests like corn earworm, aphids, or borers",
            "💨 Keep the area weed-free to reduce competition",
            "🌽 Harvest at the 'milk stage' for sweet corn for peak sweetness!"
        ]
    },
    'Grape___Black_rot': {
        'summary': "Uh oh, looks like Grape Black Rot! ⚫ This fungus can affect leaves, stems, and especially the fruit, turning them into hard 'mummies'. Sanitation and fungicide sprays are crucial.",
        'possible_reasons': [
            "🍇 Infected fruit mummies left on the vine or ground from last year",
            "🍂 Infected canes or leaves providing overwintering spots",
            "🌧️ Warm, wet weather during the growing season"
        ],
        'tips': [
            "🧹 Remove ALL mummified fruit from the vines and the ground during dormancy – this is critical!",
            "✂️ Prune out infected canes and remove fallen leaves",
            "💨 Improve air circulation through proper pruning and canopy management",
            "🛡️ Apply fungicides preventatively, starting before bloom and continuing through early fruit development, following label instructions",
            "💧 Avoid overhead irrigation if possible"
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'summary': "Seeing spots on leaves and berries ('measles') or sudden dieback? 🍇 This could be Esca, a complex fungal trunk disease. Management is difficult, focusing on prevention and vine health.",
        'possible_reasons': [
            "🪵 Fungi entering through large pruning wounds",
            "⏳ Often affects older, stressed vines",
            "❓ Complex interaction of several fungi"
        ],
        'tips': [
            "🚫 Avoid making large pruning cuts during wet winter weather when infection risk is highest",
            "✨ Consider delayed pruning (late winter/early spring) or double pruning",
            "🩹 Protect large pruning wounds with a sealant (effectiveness debated, but may help)",
            "🌳 Remove and destroy dead or suddenly collapsed vines promptly",
            "💪 Maintain overall vine health through good water and nutrient management"
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'summary': "Those angular brown spots with yellow halos look like Grape Leaf Blight (Isariopsis). 🍇 While often less severe than other diseases, it can defoliate vines if bad. Good sanitation helps.",
        'possible_reasons': [
            "🍂 Fungus overwintering in fallen leaves",
            "🌧️ Wet, humid conditions promoting spore spread",
            "🌿 Dense canopy limiting air flow"
        ],
        'tips': [
            "🍂 Rake and remove (or compost) fallen leaves in autumn or winter",
            "✂️ Prune for good air circulation to help leaves dry faster",
            "🛡️ Fungicides used for other grape diseases (like Black Rot or Downy Mildew) often control Leaf Blight too",
            "💪 Ensure good vine nutrition and health"
        ]
    },
    'Grape___healthy': {
        'summary': "Your grapevine looks vibrant and healthy! 🍇 Ready to reach for the sun. Keep up the good practices for a bountiful harvest.",
        'possible_reasons': [
            "✅ Attentive care and proper training/pruning",
            "☀️ Good site selection with plenty of sun and drainage",
            "🚫 Effective disease and pest management (or just lucky!)"
        ],
        'tips': [
            "✂️ Prune annually during dormancy based on your chosen training system (e.g., Cane or Spur pruning)",
            "💧 Water deeply but infrequently, especially during establishment and dry spells",
            "🌿 Manage the canopy during summer (leaf pulling, shoot positioning) for sun exposure and airflow",
            "👀 Monitor regularly for common pests (like Japanese beetles, grape berry moth) and diseases (like Powdery Mildew, Downy Mildew, Black Rot)",
            "🍽️ Fertilize judiciously based on soil tests or vine appearance"
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'summary': "Oh dear, this looks like symptoms of Huanglongbing (HLB) or Citrus Greening. 😥 This is a very serious bacterial disease spread by psyllids, with no known cure. Focus is on prevention and removal.",
        'possible_reasons': [
            "🐞 Spread by the Asian Citrus Psyllid insect carrying the bacteria",
            "🌳 Infected citrus trees nearby serving as a source",
            "🌱 Planting infected nursery stock (use certified disease-free trees!)"
        ],
        'tips': [
            "❗ Confirm the diagnosis with your local extension office or agricultural authority",
            "🚫 Remove and destroy infected trees immediately to prevent further spread",
            "🐞 Control Asian Citrus Psyllids using recommended insecticides and management practices",
            "🌱 Plant only certified disease-free nursery trees",
            "💪 Support overall tree health with proper nutrition and watering, but this won't cure HLB"
        ]
    },
    'Peach___Bacterial_spot': {
        'summary': "Those dark, angular spots on leaves and fruit look like Bacterial Spot on your peach tree. 🍑 This bacterial disease thrives in warm, wet weather. Management involves sanitation and protective sprays.",
        'possible_reasons': [
            "🦠 Bacteria overwintering in twig cankers or leaf buds",
            "🌧️ Wind-driven rain splashing bacteria onto leaves and fruit",
            "🌡️ Warm, humid, and wet conditions during spring and summer"
        ],
        'tips': [
            "✂️ Prune out infected twigs and cankers during dormancy",
            "🍂 Remove fallen leaves from around the tree base",
            "💧 Avoid overhead watering; water the soil directly",
            "🛡️ Apply copper-based bactericides during dormancy and potentially early season sprays (check local recommendations)",
            "🍑 Choose resistant peach varieties if available and suitable for your climate"
        ]
    },
    'Peach___healthy': {
        'summary': "Your peach tree is looking peachy keen! ✨ Healthy growth is the first step towards delicious fruit.",
        'possible_reasons': [
            "✅ Providing excellent care!",
            "☀️ Good growing location with plenty of sun and well-drained soil",
            "🚫 Currently free from major disease or pest outbreaks"
        ],
        'tips': [
            "💧 Water regularly, especially during establishment and fruit development (deep watering is best)",
            "🍽️ Fertilize in early spring based on soil needs – avoid late-season nitrogen",
            "✂️ Prune annually during dormancy (late winter/early spring) to an open center shape for light penetration and airflow",
            "🍑 Thin fruit heavily when they are small (thumbnail size) for larger, higher-quality peaches",
            "👀 Monitor for common peach pests (like plum curculio, oriental fruit moth) and diseases (like peach leaf curl, brown rot)"
        ]
    },
    'Pepper,_bell___Bacterial_spot': {
        'summary': "Seeing water-soaked spots turning brown/black on your bell pepper leaves or fruit? 🌶️ It's likely Bacterial Spot. This disease loves warm, wet conditions. Sanitation is key!",
        'possible_reasons': [
            "🦠 Bacteria surviving on seeds or infected plant debris",
            "💦 Splashing water (rain, overhead irrigation) spreading bacteria",
            "👐 Handling wet plants can spread the bacteria"
        ],
        'tips': [
            "🌱 Use certified disease-free seeds or transplants",
            "🚫 Remove and destroy infected plants or plant parts promptly",
            "💧 Water at the base of the plant, avoid wetting foliage",
            "👐 Avoid working with plants when they are wet",
            "♻️ Rotate crops: Don't plant peppers, tomatoes, or eggplants in the same spot for 3-4 years",
            "🛡️ Copper-based sprays can help suppress spread if applied early and regularly, but won't cure infected plants"
        ]
    },
    'Pepper,_bell___healthy': {
        'summary': "Your bell pepper plant looks healthy and ready to produce! 🌶️ Keep providing good care for a tasty harvest.",
        'possible_reasons': [
            "✅ Great gardening practices!",
            "☀️ Plenty of sunshine, warm temperatures, and well-drained soil",
            "💧 Consistent watering and appropriate feeding",
            "🚫 Absence of significant pests or diseases"
        ],
        'tips': [
            "💧 Water consistently, aiming for moist but not waterlogged soil",
            " mulch around the base to retain moisture and suppress weeds",
            "🍽️ Fertilize moderately; peppers need nutrients but too much nitrogen can favor leaves over fruit",
            "☀️ Ensure at least 6-8 hours of direct sunlight daily",
            "🐜 Monitor for common pests like aphids, flea beetles, or hornworms",
            "🧑‍🤝‍🧑 Provide support (stakes or cages) if plants become heavy with fruit"
        ]
    },
    'Potato___Early_blight': {
        'summary': "Seeing dark spots with 'target' rings, especially on lower potato leaves? 🎯 That sounds like Early Blight. It's a common fungal issue, often starting as plants mature.",
        'possible_reasons': [
            "🍂 Fungus surviving in infected plant debris or soil",
            "🌧️ Wet leaves from rain, dew, or overhead watering",
            " stresed plants (e.g., low fertility, other diseases)"
        ],
        'tips': [
            "🌱 Use certified disease-free seed potatoes",
            "♻️ Rotate crops: Avoid planting potatoes, tomatoes, or eggplants in the same spot for 3-4 years",
            "🍂 Remove and destroy infected lower leaves if practical",
            "💧 Water at the soil level, avoid wetting foliage, water early in the day",
            "💪 Maintain good plant health with proper fertilization and watering",
            "🛡️ Fungicides can help prevent spread if applied preventatively or at first sign, follow labels"
        ]
    },
    'Potato___Late_blight': {
        'summary': "Watch out! Those dark, water-soaked spots on leaves/stems could be Late Blight. 😨 This is a very destructive disease (infamous from the Irish Potato Famine!) and spreads rapidly in cool, wet weather.",
        'possible_reasons': [
            "🥔 Spores surviving on infected seed potatoes or volunteer plants",
            "☁️ Cool, moist conditions (nights 50-60°F, days 60-70°F with rain/fog/dew)",
            "💨 Spores spreading rapidly via wind and rain"
        ],
        'tips': [
            "‼️ Immediately remove and destroy infected plants (bag them or burn them, don't compost!)",
            "🌱 Plant only certified disease-free seed potatoes",
            "♻️ Practice crop rotation (3-4 years away from potatoes/tomatoes)",
            "🥔 Destroy volunteer potato plants that sprout up",
            "💧 Avoid overhead watering; ensure good air circulation",
            "🛡️ Fungicides are crucial for management in high-risk conditions; apply preventatively based on forecasts or at first sign"
        ]
    },
    'Potato___healthy': {
        'summary': "Your potato plants look vigorous and healthy! 🥔 Keep them happy, and they'll reward you with plenty of spuds.",
        'possible_reasons': [
            "✅ Good soil preparation and planting practices",
            "💧 Consistent moisture and nutrient supply",
            "🌱 Using healthy seed potatoes",
            "🚫 Low pressure from major pests or diseases"
        ],
        'tips': [
            "⛰️ 'Hill' soil up around the base of the stems as they grow to protect developing tubers from sunlight (which turns them green)",
            "💧 Water consistently, providing about 1-2 inches per week, especially during tuber formation",
            "🍽️ Ensure adequate soil fertility, but avoid excess nitrogen late in the season",
            "🐜 Monitor for pests like Colorado potato beetles, flea beetles, and aphids",
            "🍂 Watch for signs of early or late blight, especially during humid or wet weather"
        ]
    },
    'Raspberry___healthy': {
        'summary': "Your raspberry patch looks great! 🌱 Healthy canes are key to delicious berries.",
        'possible_reasons': [
            "✅ Proper pruning and maintenance",
            "☀️ Good site selection (sun, drainage, air circulation)",
            "💧 Adequate water and nutrients",
            "🚫 Absence of serious pests or diseases"
        ],
        'tips': [
            "✂️ Prune correctly based on type: remove fruited canes on summer-bearers after harvest; manage primocanes on everbearers",
            "💪 Provide support (trellis or wires) to keep canes upright",
            " mulch around plants to conserve moisture, suppress weeds, and keep roots cool",
            "💧 Water regularly, especially during fruiting",
            "👀 Monitor for common issues like cane borers, Japanese beetles, fungal diseases (like anthracnose or spur blight)",
            "😋 Harvest berries regularly when fully ripe!"
        ]
    },
    'Soybean___healthy': {
        'summary': "Your soybean plants look healthy and thriving! 🌱 Good stand establishment and green leaves are great signs.",
        'possible_reasons': [
            "✅ Good planting practices (seed quality, timing, depth)",
            "☀️ Favorable weather conditions (temperature, rainfall)",
            "🌱 Adequate soil fertility and pH",
            "🚫 Low levels of yield-limiting pests, diseases, or weeds"
        ],
        'tips': [
            "💧 Ensure adequate moisture, especially during flowering and pod fill stages (critical periods)",
            "🍽️ Soybeans fix nitrogen, but ensure other nutrients (Phosphorus, Potassium) are sufficient based on soil tests",
            "🐜 Scout regularly for key pests like aphids, bean leaf beetles, or stink bugs, and treat based on economic thresholds",
            "🧐 Monitor for diseases like frogeye leaf spot, septoria brown spot, or sudden death syndrome, especially during conducive weather",
            "🌿 Manage weeds effectively to reduce competition"
        ]
    },
    'Squash___Powdery_mildew': {
        'summary': "Seeing that white, powdery coating on your squash leaves? That's Powdery Mildew! 🍄 A very common fungal issue, especially later in the season. Good airflow helps!",
        'possible_reasons': [
            "☁️ High humidity (though it doesn't need free water on leaves like Downy Mildew)",
            "💨 Poor air circulation around plants",
            "🌿 Dense plant growth, shaded lower leaves"
        ],
        'tips': [
            "✂️ Prune some older or heavily infected leaves to improve airflow",
            "☀️ Plant in a sunny location with good spacing between plants",
            "💧 Water the soil, not the leaves, preferably in the morning",
            "✨ Choose powdery mildew-resistant squash varieties when possible",
            "🧪 Apply fungicides (like potassium bicarbonate, neem oil, or sulfur-based) preventatively or at first sign, covering leaves well"
        ]
    },
    'Strawberry___Leaf_scorch': {
        'summary': "Those irregular purplish spots turning brown and dry on your strawberry leaves look like Leaf Scorch. 🔥 It's a fungal disease that can weaken plants.",
        'possible_reasons': [
            "🍂 Fungus overwintering on infected dead leaves",
            "💦 Spores spread by splashing water (rain, irrigation)",
            "🌡️ Favored by warm, wet conditions"
        ],
        'tips': [
            "🧹 Remove and destroy infected leaves and plant debris after harvest and in late fall",
            " mulch with straw to keep leaves off the soil and reduce splashing",
            "💧 Avoid overhead watering; use drip irrigation or water early in the day",
            "🍓 Plant resistant strawberry varieties if available",
            "🛡️ Fungicides may be needed in severe cases, applied according to local recommendations"
        ]
    },
    'Strawberry___healthy': {
        'summary': "Your strawberry plants look happy and healthy! 🍓 Ready to produce sweet treats.",
        'possible_reasons': [
            "✅ Good planting system and care",
            "☀️ Sunny location with well-drained, fertile soil",
            "💧 Consistent watering, especially during fruiting",
            "🚫 Effective pest and disease management"
        ],
        'tips': [
            " mulch with straw or other material to keep fruit clean, conserve moisture, and suppress weeds",
            "💧 Water regularly, ensuring about 1 inch per week, especially from flowering through harvest",
            "🍽️ Fertilize after renovation (for June-bearers) or lightly throughout season (for day-neutrals/everbearers)",
            "✂️ Renovate June-bearing beds after harvest (mow leaves, narrow rows, fertilize)",
            "👀 Monitor for common pests (slugs, spittlebugs, tarnished plant bugs) and diseases (gray mold, leaf spots)",
            "❄️ Provide winter protection (like straw mulch) in colder climates"
        ]
    },
    'Tomato___Bacterial_spot': {
        'summary': "Seeing small, dark, sometimes water-soaked spots on tomato leaves and fruit? 🍅 Could be Bacterial Spot. This bacteria loves warm, wet conditions. Sanitation is key!",
        'possible_reasons': [
            "🦠 Bacteria surviving on seeds or infected plant debris",
            "💦 Splashing water (rain, overhead irrigation) spreading bacteria",
            "👐 Handling wet plants transferring bacteria"
        ],
        'tips': [
            "🌱 Use certified disease-free seeds or transplants",
            "🚫 Remove and destroy infected plants or severely affected leaves promptly",
            "💧 Water at the base of the plant, avoid wetting foliage",
            "👐 Avoid working with plants when they are wet",
            "♻️ Rotate crops: Don't plant tomatoes, peppers, or eggplants in the same spot for 3-4 years",
            "🛡️ Copper-based sprays can help suppress spread if applied early and regularly, but won't cure infected plants"
        ]
    },
    'Tomato___Early_blight': {
        'summary': "Those dark spots with 'target' or 'bullseye' rings, starting on lower tomato leaves? 🎯 Classic Early Blight. A common fungal issue, especially as plants age or get stressed.",
        'possible_reasons': [
            "🍂 Fungus surviving in infected plant debris or soil",
            "🌧️ Wet leaves from rain, dew, or overhead watering",
            " stresed plants (low fertility, other issues)"
        ],
        'tips': [
            "🌱 Use certified disease-free seed/transplants",
            "♻️ Rotate crops (3-4 years away from tomatoes/potatoes/eggplants)",
            "🍂 Remove and destroy infected lower leaves promptly",
            " mulch around plants to reduce soil splashing onto leaves",
            "💧 Water at the soil level, avoid wetting foliage, water early",
            "💪 Maintain good plant health and fertility",
            "🛡️ Apply fungicides preventatively or at first sign, especially during wet/humid weather"
        ]
    },
    'Tomato___Late_blight': {
        'summary': "Watch out! Dark, greasy-looking, water-soaked spots on tomato leaves/stems? 😨 This could be Late Blight, a highly destructive disease that spreads fast in cool, wet weather.",
        'possible_reasons': [
            "🥔 Spores from infected potatoes (seed, volunteers) or nearby infected tomatoes",
            "☁️ Cool, moist conditions (nights 50-60°F, days 60-70°F with rain/fog/dew)",
            "💨 Spores spreading rapidly via wind and rain"
        ],
        'tips': [
            "‼️ Immediately remove and destroy infected plants (bag or burn, don't compost!) to prevent spread",
            "🌱 Plant certified disease-free stock",
            "♻️ Practice crop rotation (3-4 years)",
            "🥔 Control volunteer potato and tomato plants",
            "💧 Avoid overhead watering; ensure good air circulation (pruning, spacing)",
            "🛡️ Fungicides are essential for management in high-risk conditions; apply preventatively based on forecasts or at first sign"
        ]
    },
    'Tomato___Leaf_Mold': {
        'summary': "Seeing yellowish spots on top of tomato leaves, with fuzzy olive-green/brown mold underneath? ☁️ That's Leaf Mold, a fungus that loves high humidity.",
        'possible_reasons': [
            "💧 High relative humidity (often >85%)",
            "💨 Poor air circulation, especially in greenhouses or dense plantings",
            "🌿 Fungus surviving on plant debris or structures"
        ],
        'tips': [
            "💨 Increase air circulation: prune lower leaves, space plants well, use fans in greenhouses",
            "💧 Water at the base, avoid wetting leaves, water early in the day",
            "🌡️ In greenhouses, manage humidity through venting and heating if possible",
            "🍂 Remove and destroy infected leaves promptly",
            "✨ Choose resistant tomato varieties if Leaf Mold is a persistent problem",
            "🛡️ Fungicides can help, especially if applied preventatively"
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'summary': "Lots of small, circular spots with dark borders and tiny black dots (pycnidia) in the center? 🐞 Sounds like Septoria Leaf Spot on your tomato. It starts low and moves up.",
        'possible_reasons': [
            "🍂 Fungus overwintering in infected plant debris or on weeds",
            "💦 Spores splashed by rain or overhead watering",
            "🌡️ Moderate temperatures and high humidity/rainfall"
        ],
        'tips': [
            "🍂 Remove and destroy infected lower leaves as soon as they appear",
            " mulch heavily to create a barrier between soil and leaves",
            "💧 Water at the base of the plant, avoid wetting foliage",
            "♻️ Rotate crops (3-4 years away from tomatoes/potatoes/eggplants)",
            "🌿 Control susceptible weeds nearby",
            "💨 Improve air circulation through pruning and spacing",
            "🛡️ Fungicides can help slow the spread if applied early and regularly"
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'summary': "Seeing tiny yellow dots (stippling), fine webbing, or bronzing on tomato leaves? 🕷️ You might have Spider Mites! These tiny pests suck plant juices and love hot, dry conditions.",
        'possible_reasons': [
            "☀️ Hot, dry, dusty conditions favouring mite reproduction",
            "🌬️ Mites blown in by wind or carried on tools/clothing",
            "🚫 Lack of natural predators (sometimes due to broad-spectrum insecticide use)"
        ],
        'tips': [
            "🚿 Spray plants forcefully with water, especially undersides of leaves, to dislodge mites (do early in day)",
            "💧 Increase humidity around plants if possible (less favourable for mites)",
            "🐞 Encourage beneficial insects (predatory mites, ladybugs) that feed on spider mites",
            "✂️ Remove heavily infested leaves or plants",
            "🧪 Use insecticidal soaps or horticultural oils, ensuring thorough coverage of leaf undersides. Miticides may be needed for severe infestations."
        ]
    },
    'Tomato___Target_Spot': {
        'summary': "Are those spots on your tomato leaves small and dark, maybe with concentric rings like Early Blight, but often smaller? 🎯 Could be Target Spot (Corynespora). Thrives in humidity.",
        'possible_reasons': [
            "🍂 Fungus surviving in plant debris",
            "🌧️ High humidity and warm temperatures",
            "💦 Splashing water spreading spores"
        ],
        'tips': [
            "💨 Improve air circulation through pruning and proper spacing",
            "💧 Water at the base, avoid wetting leaves",
            " mulch to reduce soil splash",
            "🍂 Remove and destroy infected leaves and clean up debris at season's end",
            "♻️ Practice crop rotation",
            "🛡️ Fungicides used for other tomato leaf spots may also help control Target Spot"
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'summary': "Seeing upward curling, yellowing, and stunting on new tomato growth? 😥 This could be Tomato Yellow Leaf Curl Virus (TYLCV). It's spread by whiteflies and can severely impact yield.",
        'possible_reasons': [
            "🦟 Virus transmitted by infected Silverleaf Whiteflies",
            "🌱 Infected plants nearby serving as a source",
            "☀️ Often worse in warm climates where whiteflies thrive"
        ],
        'tips': [
            "🦟 Control whiteflies early and consistently (sticky traps, insecticidal soap, neem oil, targeted insecticides if necessary)",
            "🚫 Remove and destroy infected plants immediately to reduce spread",
            "🌱 Plant TYLCV-resistant tomato varieties if available for your region",
            "🌿 Control weeds in and around the garden, as they can host whiteflies",
            " reflective mulch may help deter whiteflies"
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'summary': "Mottled light/dark green patterns (mosaic), yellowing, and distorted or stunted growth on your tomato? 🦠 Sounds like Tomato Mosaic Virus (ToMV) or a related virus.",
        'possible_reasons': [
            "🌱 Infected seeds or transplants",
            "👐 Virus spread mechanically by handling plants (especially after using tobacco products, as TMV is related)",
            "🌿 Surviving on infected plant debris or some weeds"
        ],
        'tips': [
            "🚫 Remove and destroy infected plants immediately",
            "👐 Wash hands thoroughly with soap and water before handling plants, especially if you use tobacco",
            "✨ Disinfect tools (pruners, stakes) regularly, especially between plants",
            "🌱 Use certified virus-free seeds/transplants",
            "🍓 Choose resistant tomato varieties (check labels for ToMV resistance)",
            "🐜 Control sucking insects like aphids, as they can sometimes transmit other viruses (though ToMV is mostly mechanical)"
        ]
    },
    'Tomato___healthy': {
        'summary': "Your tomato plant looks fantastic! 🍅 Strong, green, and ready to produce delicious fruit. Keep up the excellent care!",
        'possible_reasons': [
            "✅ You're doing a great job!",
            "☀️ Plenty of sunshine (6-8+ hours/day)",
            "💧 Consistent watering and good soil drainage",
            "🍽️ Balanced nutrition",
            "🚫 Low pest and disease pressure"
        ],
        'tips': [
            "💧 Water deeply and consistently at the base, aiming for moist soil",
            " mulch to conserve moisture, suppress weeds, and reduce soil splash",
            "🍽️ Fertilize regularly based on plant stage (more Nitrogen early, more P & K during fruiting)",
            "🧑‍🤝‍🧑 Provide support (stakes, cages, trellis) as the plant grows",
            "✂️ Prune suckers as desired (especially for indeterminate types) to manage growth and improve airflow",
            "👀 Monitor regularly for pests (hornworms, aphids, stink bugs) and diseases"
        ]
    }
}

# Example of accessing a recommendation
# print(PLANT_GENERAL_RECOMMENDATIONS['Tomato___Early_blight']['summary'])
# print(PLANT_GENERAL_RECOMMENDATIONS['Apple___healthy']['tips'])