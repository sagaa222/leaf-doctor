"""Crop disease knowledge base: typical VISUAL signs used for differential diagnosis.

The small vision model is much more accurate when it compares what it sees against a short list of
known problems for the crop, instead of guessing a disease name from memory.
Format per entry: (name, cause, category, key visible signs)
"""

CROPS = {
    "maize": [
        ("Common rust", "Puccinia sorghi", "fungal",
         "small oval/elongated cinnamon-brown to orange RAISED powdery pustules scattered on BOTH leaf surfaces; "
         "rubbing leaves gives rusty powder; no long lesions"),
        ("Southern rust", "Puccinia polysora", "fungal",
         "very small ROUND light-orange pustules densely clustered mainly on the UPPER surface; spreads fast in hot humid weather"),
        ("Northern corn leaf blight (Turcicum)", "Exserohilum turcicum", "fungal",
         "LONG cigar/boat-shaped grey-green to tan lesions 2.5–15 cm, starting on lower leaves"),
        ("Maydis / Southern corn leaf blight", "Bipolaris maydis", "fungal",
         "tan oval to rectangular lesions between veins, 1–4 cm, buff centre with brown/purple margin, NOT raised"),
        ("Gray leaf spot", "Cercospora zeae-maydis", "fungal",
         "narrow rectangular grey-tan lesions strictly bounded by veins, running parallel like matchsticks"),
        ("Downy mildew", "Peronosclerospora sorghi", "fungal",
         "pale yellow/white stripes along leaves, white downy growth on underside in morning, stunted plants"),
        ("Fall armyworm", "Spodoptera frugiperda", "pest",
         "ragged holes and 'windowpane' scraping on leaves, wet sawdust-like frass in the whorl, larvae with inverted Y on head"),
        ("Corn leaf aphid", "Rhopalosiphum maidis", "pest",
         "colonies of small blue-green insects in whorl/tassel, sticky honeydew, black sooty mould"),
        ("Zinc deficiency", "low zinc", "nutrient deficiency",
         "broad white/pale-yellow band on both sides of the midrib near leaf base on YOUNG leaves, short internodes"),
        ("Nitrogen deficiency", "low nitrogen", "nutrient deficiency",
         "V-shaped yellowing starting at leaf TIP and moving down the midrib on OLDER lower leaves"),
        ("Potassium deficiency", "low potassium", "nutrient deficiency",
         "yellow then brown scorched leaf MARGINS on older leaves, midrib stays green"),
    ],
    "rice": [
        ("Rice blast", "Magnaporthe oryzae", "fungal",
         "diamond/spindle-shaped spots with grey-white centre and brown margin; neck turns black and breaks"),
        ("Brown spot", "Bipolaris oryzae", "fungal",
         "many oval brown spots like sesame seeds, grey centre, yellow halo; often in poor soil"),
        ("Bacterial leaf blight", "Xanthomonas oryzae pv. oryzae", "bacterial",
         "yellow to straw-white stripes starting at leaf TIP/edges moving down with wavy margin; leaves dry"),
        ("Sheath blight", "Rhizoctonia solani", "fungal",
         "oval greenish-grey lesions with brown border on leaf SHEATH near water line, snake-skin pattern"),
        ("Rice tungro", "tungro virus (green leafhopper)", "viral",
         "yellow-orange discoloration from leaf tip, stunted plants, fewer tillers, patchy in field"),
        ("False smut", "Ustilaginoidea virens", "fungal",
         "individual grains turn into orange then green-black velvety balls"),
        ("Leaf folder", "Cnaphalocrocis medinalis", "pest",
         "leaves folded lengthwise with white scraped streaks inside"),
        ("Stem borer", "Scirpophaga incertulas", "pest",
         "central shoot dries ('dead heart') or white empty panicle ('white ear'), small hole on stem"),
    ],
    "tomato": [
        ("Early blight", "Alternaria solani", "fungal",
         "brown spots with CONCENTRIC rings (target board), yellow halo, starts on OLDER lower leaves; dark ring on fruit stem end"),
        ("Late blight", "Phytophthora infestans", "fungal",
         "large irregular water-soaked greasy dark-brown patches, white mould on underside in humid weather; firm brown fruit rot"),
        ("Septoria leaf spot", "Septoria lycopersici", "fungal",
         "many SMALL round spots (2–3 mm) with dark border, grey centre with tiny black dots, lower leaves"),
        ("Bacterial spot", "Xanthomonas spp.", "bacterial",
         "small dark greasy angular spots, sometimes yellow halo; raised scabby spots on fruit"),
        ("Tomato leaf curl virus", "begomovirus (whitefly)", "viral",
         "leaves curl UPWARD, small crinkled yellow-edged leaves, bushy stunted plant, few fruits"),
        ("Fusarium / bacterial wilt", "Fusarium oxysporum / Ralstonia solanacearum", "fungal",
         "whole plant or one side wilts and yellows, brown vascular tissue inside stem"),
        ("Leaf miner", "Liriomyza spp. / Tuta absoluta", "pest",
         "white-silvery winding SERPENTINE tunnels (Liriomyza) or papery blotch mines (Tuta) inside the leaf"),
        ("Blossom end rot", "calcium deficiency / irregular watering", "nutrient deficiency",
         "dark sunken leathery patch at the BOTTOM (blossom end) of fruit"),
        ("Powdery mildew", "Leveillula taurica", "fungal",
         "yellow patches on upper surface with white powdery growth below"),
    ],
    "potato": [
        ("Early blight", "Alternaria solani", "fungal", "brown target-like spots with concentric rings on older leaves"),
        ("Late blight", "Phytophthora infestans", "fungal",
         "water-soaked dark patches spreading fast, white mould on underside, brown rot in tubers"),
        ("Common scab", "Streptomyces scabies", "bacterial", "rough corky scabs on tuber skin"),
    ],
    "chilli": [
        ("Anthracnose / fruit rot", "Colletotrichum spp.", "fungal",
         "sunken circular spots on FRUIT with concentric rings and black/orange dots; twig die-back"),
        ("Chilli leaf curl", "begomovirus (whitefly) / thrips / mites", "viral",
         "leaves curl up or down, puckered, small; thrips cause upward curl, mites downward curl"),
        ("Cercospora leaf spot", "Cercospora capsici", "fungal",
         "round spots with light grey centre and dark brown margin ('frog-eye'), leaves drop"),
        ("Powdery mildew", "Leveillula taurica", "fungal",
         "white powdery growth on leaf underside, yellow patches on top, leaf fall"),
        ("Bacterial leaf spot", "Xanthomonas spp.", "bacterial", "small water-soaked spots turning brown with yellow halo"),
    ],
    "brinjal": [
        ("Shoot and fruit borer", "Leucinodes orbonalis", "pest",
         "drooping/wilted shoot tips, holes in fruit plugged with frass"),
        ("Little leaf", "phytoplasma (leafhopper)", "viral", "very small narrow leaves, bushy plant, no fruit"),
        ("Phomopsis blight", "Phomopsis vexans", "fungal",
         "round brown leaf spots with light centre; large sunken soft spots on fruit with black dots"),
        ("Epilachna beetle", "Henosepilachna vigintioctopunctata", "pest",
         "leaves scraped into skeleton/ladder pattern, orange spotted beetles and spiny yellow larvae"),
    ],
    "okra": [
        ("Yellow vein mosaic", "begomovirus (whitefly)", "viral",
         "bright yellow network of VEINS, later whole leaf yellow, small fibrous fruits"),
        ("Powdery mildew", "Erysiphe cichoracearum", "fungal", "white powder on leaves"),
        ("Shoot and fruit borer", "Earias vittella", "pest", "drooping shoots, bored fruits with holes"),
    ],
    "banana": [
        ("Sigatoka leaf spot", "Mycosphaerella spp.", "fungal",
         "small yellow streaks parallel to veins becoming dark brown elongated spots with grey centre, leaf drying"),
        ("Panama wilt", "Fusarium oxysporum f. sp. cubense", "fungal",
         "yellowing of older leaves from margin, leaves hang around stem like skirt, split pseudostem"),
        ("Bunchy top", "banana bunchy top virus (aphid)", "viral",
         "dark green dot-dash streaks on veins/petiole, narrow upright bunched leaves at top"),
    ],
    "cotton": [
        ("Bacterial blight", "Xanthomonas citri pv. malvacearum", "bacterial",
         "angular water-soaked spots bounded by veins turning brown-black, black arm on stems"),
        ("Alternaria leaf spot", "Alternaria macrospora", "fungal", "brown spots with concentric rings"),
        ("Cotton leaf curl", "begomovirus (whitefly)", "viral", "upward/downward curling, thick dark veins, leaf-like outgrowths"),
        ("Jassid (leafhopper) damage", "Amrasca biguttula", "pest",
         "leaf edges turn yellow then reddish and curl DOWN ('hopper burn')"),
        ("Bollworm", "Helicoverpa / pink bollworm", "pest", "holes in bolls and squares, damaged lint"),
    ],
    "groundnut": [
        ("Early leaf spot", "Cercospora arachidicola", "fungal",
         "light-brown round spots with YELLOW HALO mainly on upper surface"),
        ("Late leaf spot (tikka)", "Nothopassalora personata", "fungal",
         "dark brown/black round spots mainly on lower surface, no clear halo"),
        ("Rust", "Puccinia arachidis", "fungal", "orange-brown raised pustules on leaf UNDERSIDE that rupture to powder"),
        ("Leaf miner", "Aproaerema modicella", "pest", "blotch mines and folded webbed leaflets"),
    ],
    "coconut": [
        ("Bud rot", "Phytophthora palmivora", "fungal", "youngest central leaf yellows, wilts and pulls out easily, foul smell"),
        ("Rhinoceros beetle", "Oryctes rhinoceros", "pest", "V-shaped or triangular cuts on opened fronds"),
        ("Eriophyid mite", "Aceria guerreronis", "pest", "triangular pale then brown cracked patches on nuts under the cap"),
        ("Grey leaf blight", "Pestalotiopsis palmarum", "fungal", "yellow spots turning grey with dark borders on older leaflets"),
    ],
    "mango": [
        ("Anthracnose", "Colletotrichum gloeosporioides", "fungal",
         "black irregular spots on leaves, flowers and fruit; black sunken patches on ripe fruit"),
        ("Powdery mildew", "Oidium mangiferae", "fungal", "white powder on flowers, young leaves and small fruits; flowers drop"),
        ("Sooty mould", "Capnodium spp. (after hoppers/scales)", "fungal", "black papery coating on leaves that peels off"),
    ],
    "onion": [
        ("Purple blotch", "Alternaria porri", "fungal", "small white sunken spots becoming PURPLE lesions with yellow margin"),
        ("Stemphylium blight", "Stemphylium vesicarium", "fungal", "yellow-orange elongated spots, leaf tips dry"),
        ("Thrips", "Thrips tabaci", "pest", "silvery-white streaks and specks on leaves, tips curl and dry"),
    ],
    "citrus": [
        ("Citrus canker", "Xanthomonas citri", "bacterial",
         "raised corky brown spots with YELLOW HALO on both leaf sides, fruit and twigs"),
        ("Citrus greening (HLB)", "Candidatus Liberibacter (psyllid)", "bacterial",
         "blotchy asymmetrical yellow mottling across veins, small lopsided bitter fruit"),
        ("Citrus leaf miner", "Phyllocnistis citrella", "pest", "silvery serpentine mines on young leaves, leaves curl"),
    ],
    "cucurbits": [
        ("Downy mildew", "Pseudoperonospora cubensis", "fungal",
         "ANGULAR yellow patches bounded by veins on top, grey-purple fuzz underneath"),
        ("Powdery mildew", "Podosphaera xanthii", "fungal", "white flour-like powder on upper leaf surface"),
        ("Mosaic virus", "CMV / ZYMV (aphids)", "viral", "light/dark green mosaic, distorted leaves, warty fruits"),
        ("Fruit fly", "Bactrocera cucurbitae", "pest", "punctures on fruit, fruit rots with maggots inside"),
    ],
    "sugarcane": [
        ("Red rot", "Colletotrichum falcatum", "fungal", "leaves dry from top, split stalk is red with white patches, sour smell"),
        ("Smut", "Sporisorium scitamineum", "fungal", "long black whip-like structure from the top of the cane"),
    ],
}

ALIASES = {
    "maize": ["corn", "maize", "zea mays", "makka", "makkacholam"],
    "rice": ["rice", "paddy", "oryza"],
    "tomato": ["tomato", "solanum lycopersicum", "thakkali"],
    "potato": ["potato", "solanum tuberosum"],
    "chilli": ["chilli", "chili", "pepper", "capsicum", "milagai"],
    "brinjal": ["brinjal", "eggplant", "aubergine", "solanum melongena", "kathiri"],
    "okra": ["okra", "bhindi", "bhendi", "lady finger", "ladies finger", "abelmoschus", "vendakkai"],
    "banana": ["banana", "plantain", "musa", "vazhai"],
    "cotton": ["cotton", "gossypium", "paruthi"],
    "groundnut": ["groundnut", "peanut", "arachis", "nilakadalai"],
    "coconut": ["coconut", "cocos", "thennai"],
    "mango": ["mango", "mangifera", "maa"],
    "onion": ["onion", "allium", "vengayam"],
    "citrus": ["citrus", "lemon", "lime", "orange", "mandarin", "sweet orange", "elumichai"],
    "cucurbits": ["cucumber", "pumpkin", "gourd", "melon", "squash", "cucurbit", "watermelon", "bitter gourd"],
    "sugarcane": ["sugarcane", "saccharum", "karumbu"],
}

GENERIC = [
    ("Rust", "rust fungus", "fungal", "small raised orange/brown powdery pustules"),
    ("Powdery mildew", "powdery mildew fungus", "fungal", "white flour-like powder on leaf surface"),
    ("Downy mildew", "oomycete", "fungal", "angular yellow patches on top, grey/purple fuzz underneath"),
    ("Fungal leaf spot", "fungal pathogen", "fungal", "round brown spots, often with darker margin or rings"),
    ("Anthracnose", "Colletotrichum spp.", "fungal", "dark sunken spots on leaves/fruit, sometimes with pink/orange spore masses"),
    ("Bacterial leaf spot / blight", "bacterial pathogen", "bacterial", "water-soaked angular spots, yellow halo, may ooze"),
    ("Mosaic / leaf curl virus", "virus spread by insects", "viral", "mottled light-dark green or yellow, curled, distorted leaves"),
    ("Leaf miner", "fly or moth larvae", "pest", "white winding tunnels or blotches inside the leaf"),
    ("Sucking pests (aphids, whitefly, mites, thrips)", "sap-sucking insects", "pest",
     "tiny insects on underside, sticky honeydew, sooty mould, stippling or silvering, curling"),
    ("Caterpillar / beetle damage", "chewing insects", "pest", "holes, ragged edges, skeletonised leaves, frass"),
    ("Nitrogen deficiency", "low nitrogen", "nutrient deficiency", "uniform yellowing of older lower leaves"),
    ("Iron / magnesium deficiency", "low iron or magnesium", "nutrient deficiency",
     "yellowing between green veins (interveinal) on young (Fe) or old (Mg) leaves"),
    ("Sunscald / chemical injury", "environmental", "environmental",
     "bleached or scorched patches on exposed side, no spreading spots, no fungal growth"),
]


def lookup(plant_name: str, scientific_name: str = ""):
    """Returns (crop_key or None, list of entries for that crop)."""
    text = f"{plant_name} {scientific_name}".lower()
    for key, names in ALIASES.items():
        if any(n in text for n in names):
            return key, CROPS[key]
    return None, []


def as_text(entries):
    return "\n".join(f"- {n} ({c}, {cat}): {signs}" for n, c, cat, signs in entries)


def signs_for(name: str, entries):
    name = (name or "").lower()
    for n, c, cat, signs in entries:
        if n.lower() in name or name in n.lower():
            return signs
    return ""
