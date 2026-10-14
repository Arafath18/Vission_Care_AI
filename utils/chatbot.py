"""Local RAG chatbot engine for VisionCare clinical assistant."""

import re
import urllib.parse
from typing import Any, Optional

# Multi-lingual Retinal Disease Knowledge Base
KNOWLEDGE_BASE = {
    "DME": {
        "name": "Diabetic Macular Edema (DME)",
        "tamil_name": "நீரிழிவு விழித்திரை மேக்குலா நீர்வீக்கம் (DME)",
        "keywords": ["dme", "diabetic", "macular", "edema", "sugar", "neerveekkam", "diabetes", "நீர்வீக்கம்"],
        "general": {
            "english": "Diabetic Macular Edema (DME) is a serious complication of diabetes where fluid accumulates in the macula (the central part of the retina), causing swelling and blurred central vision.",
            "tamil": "நீரிழிவு விழித்திரை மேக்குலா நீர்வீக்கம் (DME) என்பது நீரிழிவு நோயின் ஒரு தீவிரமான சிக்கலாகும். விழித்திரையின் மத்திய பகுதியான மேக்குலாவில் திரவம் சேர்வதால் நீர்வீக்கம் ஏற்பட்டு மத்திய பார்வை மங்கலாகிறது.",
            "tanglish": "DME enpathu sugar (diabetes) irukuravangaluku varum eye disease. Kannula ulla macula paguthiyil neer (fluid) seruvathaal veekkam yerpadu paarvai mangalaagum."
        },
        "symptoms": {
            "english": "Key symptoms include blurry or wavy central vision, colors appearing washed out or faded, difficulty reading or driving, and a blind spot near the center of vision.",
            "tamil": "முக்கிய அறிகுறிகள்: மங்கலான அல்லது அலை அலையான பார்வை, மங்கிப்போன நிறங்கள், படிப்பதில் அல்லது ஓட்டுவதில் சிரமம், மற்றும் மத்திய பார்வையில் இருண்ட புள்ளி.",
            "tanglish": "DME symptoms: Paarvai mangalaga (blurry) iruppathu, color faded-ah theriyuvathu, padikumbothu ezhuthukkal valainthu (wavy) theriyuvathu, centre-la black spot viluvathu."
        },
        "treatment": {
            "english": "Treatments include Anti-VEGF injections (e.g. Lucentis, Eylea), focal/grid laser photocoagulation, steroid implants, and maintaining strict control of blood glucose, blood pressure, and cholesterol levels.",
            "tamil": "சிகிச்சைகள்: கண்ணுக்குள் செலுத்தப்படும் இன்ஜெக்ஷன்கள் (Anti-VEGF), லேசர் சிகிச்சை, ஸ்டீராய்டு உள்வைப்புகள், மற்றும் இரத்த சர்க்கரை, இரத்த அழுத்தம், கொழுப்பை மிகக் கட்டுப்பாட்டில் வைத்தல்.",
            "tanglish": "DME treatment: Kannula Anti-VEGF injection poduvannga, laser treatment panni leak aagura blood vessels-ai seal pannalaam, steroid capsules iruku. Aana blood sugar-ai strict-ah control pannanum."
        },
        "prevention": {
            "english": "Prevention requires annual dilated eye exams, keeping hemoglobin A1c levels below 7%, managing hypertension and lipid levels, eating a healthy diet, and exercising regularly.",
            "tamil": "தடுப்பு முறைகள்: ஆண்டுக்கு ஒருமுறை கண் பரிசோதனை, HbA1c அளவை 7% க்குக் கீழ் வைத்தல், இரத்த அழுத்தம் மற்றும் கொழுப்பை பராமரித்தல், ஆரோக்கியமான உணவு மற்றும் உடற்பயிற்சி.",
            "tanglish": "DME prevent panna: Varushathuku oru murai eye screening pannanum. HbA1c 7%-ku keezha irukanum. Walking matrum healthy food eduthukanum."
        }
    },
    "CNV": {
        "name": "Choroidal Neovascularization (CNV)",
        "tamil_name": "விழித்திரைக்கீழ் புதிய இரத்தக்குழாய் உருவாக்கம் (CNV)",
        "keywords": ["cnv", "choroidal", "neovascularization", "new vessels", "rathakuzhal", "இரத்தக்குழாய்"],
        "general": {
            "english": "Choroidal Neovascularization (CNV) involves the growth of abnormal, leaky blood vessels underneath the retina, often associated with wet age-related macular degeneration (wet AMD) or high myopia.",
            "tamil": "விழித்திரைக்கீழ் புதிய இரத்தக்குழாய் உருவாக்கம் (CNV) என்பது விழித்திரையின் அடியில் அசாதாரணமான, எளிதில் கசியும் இரத்தக்குழாய்கள் வளர்வதாகும். இது பொதுவாக வயது முதிர்வு சார்ந்த மேக்குலா சிதைவுடன் தொடர்புடையது.",
            "tanglish": "CNV enbathu retina-vukku keezha abnormal-ah ratha kuzhalgal (blood vessels) valaruvathu. Ithu seekkiram leak aagi ratham matrum neer vazhindhu paarvaiyai kedukkum."
        },
        "symptoms": {
            "english": "Symptoms include a sudden painless distortion of vision (metamorphopsia) where straight lines appear bent, rapid decline in central vision acuity, and central dark patches.",
            "tamil": "அறிகுறிகள்: வலி இல்லாத திடீர் பார்வை சிதைவு (நேர்கோடுகள் வளைந்து தெரிவது), மத்திய பார்வை விரைவாகக் குறைதல், மற்றும் மத்திய பகுதியில் இருண்ட புள்ளிகள் தோன்றுதல்.",
            "tanglish": "CNV symptoms: Vali illamal sudden-ah paarvai valaiyiradhu (straight lines bent-ah theriyum), centre vision vegamaa kuraiyadhu, naduvula karuppu pulligal theriyum."
        },
        "treatment": {
            "english": "First-line therapy is intravitreal Anti-VEGF therapy to inhibit abnormal vessel growth. Photodynamic therapy (PDT) or thermal laser may also be utilized in select cases.",
            "tamil": "முதன்மை சிகிச்சை: அசாதாரண இரத்தக்குழாய் வளர்ச்சியைத் தடுக்க கண்ணுக்குள் போடப்படும் Anti-VEGF இன்ஜெக்ஷன். குறிப்பிட்ட சில வழக்குகளுக்கு ஃபோட்டோடைனமிக் சிகிச்சை (PDT) மற்றும் லேசர் பயன்படுத்தப்படுகிறது.",
            "tanglish": "CNV treatment: Intravitreal Anti-VEGF injections poduvaanga abnormal blood vessels-ai surunga vekka. Photodynamic therapy (PDT) or thermal laser treatment-um use pannuvanga."
        },
        "prevention": {
            "english": "Monitor vision weekly using an Amsler grid. Stop smoking, manage cardiovascular risks, protect eyes from UV rays, and maintain a diet rich in leafy green vegetables and omega-3 fatty acids.",
            "tamil": "தடுப்பு முறைகள்: வாரந்தோறும் ஆம்ஸ்லர் கிரிட் (Amsler Grid) மூலம் பார்வையைச் சோதித்தல், புகைபிடித்தலை நிறுத்துதல், இதய நோய் ஆபத்தைக் குறைத்தல், பசுமையான காய்கறிகள் மற்றும் ஒமேகா-3 உணவுகளை உட்கொள்ளுதல்.",
            "tanglish": "CNV management: Varathuku oru murai Amsler Grid chart vechu eye-ai check pannanum. Cigarette smoking-ai niruthanum, lipid/heart risk-ai control pannanum, green vegetables sapdanum."
        }
    },
    "DRUSEN": {
        "name": "Drusen",
        "tamil_name": "விழித்திரை கொழுப்பு படிவுகள் (Drusen)",
        "keywords": ["drusen", "drusens", "deposits", "yellow spots", "padivugal", "படிவுகள்", "spots"],
        "general": {
            "english": "Drusen are tiny yellow or white extracellular deposits of lipids and proteins that accumulate beneath the retina. They are a hallmark clinical sign of dry Age-Related Macular Degeneration (AMD).",
            "tamil": "விழித்திரை கொழுப்பு படிவுகள் (Drusen) என்பவை விழித்திரைக்கு அடியில் சேரும் சிறிய மஞ்சள் அல்லது வெள்ளை நிற கொழுப்பு மற்றும் புரதப் படிவுகளாகும். இவை உலர் வயது முதிர்வு சார்ந்த மேக்குலா சிதைவின் முக்கிய அடையாளமாகும்.",
            "tanglish": "Drusen enrathu retina-vuku keezha tiny yellow or white fatty deposits (lipids, proteins) form aaguradhu. Ithu dry AMD disease-oda primary start-nu solluvanga."
        },
        "symptoms": {
            "english": "Small drusen are usually asymptomatic. Larger or soft drusen cause mild blurred vision, difficulty seeing in dim light, and distortion of text.",
            "tamil": "சிறிய படிவுகள் எந்த அறிகுறியையும் காட்டுவதில்லை. பெரிய படிவுகள் லேசான மங்கலான பார்வை, மங்கலான வெளிச்சத்தில் பார்ப்பதில் சிரமம், மற்றும் எழுத்துக்கள் சிதைந்து தெரிவதை உண்டாக்கும்.",
            "tanglish": "Drusen symptoms: Chinna drusen-a irundha symptoms theriyadhu. Aana perusa irundha, dim light-la paarvai theriyathu, read panna kashtama irukum, mild blurring irukum."
        },
        "treatment": {
            "english": "There is no direct medical treatment for early dry AMD with drusen. Progression is managed through dietary supplementation (AREDS2 formula containing vitamins C, E, zinc, lutein, and zeaxanthin).",
            "tamil": "ஆரம்ப உலர் AMD-க்கு நேரடி மருந்து சிகிச்சை இல்லை. வைட்டமின் சி, ஈ, துத்தநாகம், லுடீன் மற்றும் ஜியாக்சாண்டின் அடங்கிய AREDS2 வைட்டமின் சப்ளிமெண்ட்டுகள் மூலம் நோய் தீவிரமடைவதைத் தடுக்கலாம்.",
            "tanglish": "Drusen-ku direct treatment illa. Disease progression-ai thadukka AREDS2 multivitamins capsule (Vitamins C, E, zinc, lutein, zeaxanthin) specialist recommend pannuvaanga."
        },
        "prevention": {
            "english": "Preventative steps include visual monitoring with an Amsler grid, quitting smoking, eating antioxidant-rich foods, and wearing UV-blocking sunglasses.",
            "tamil": "தடுப்பு முறைகள்: ஆம்ஸ்லர் கிரிட் சுயபரிசோதனை, புகைபிடிப்பதைத் தவிர்த்தல், ஆன்டிஆக்ஸிடன்ட்கள் நிறைந்த உணவுகளை உண்ணுதல், சூரிய ஒளியிலிருந்து கண்களைப் பாதுகாக்க UV கண்ணாடி அணிதல்.",
            "tanglish": "Drusen prevent panna: Smoker-ah irundha stop pannanum. Anti-oxidants rich fruit matrum omega-3 meen vagai unavil eduthukka vendum. UV protective glasses use pannanum."
        }
    },
    "NORMAL": {
        "name": "Normal Retina",
        "tamil_name": "இயல்பான விழித்திரை",
        "keywords": ["normal", "healthy", "clear", "no disease", "iyalbana", "இயல்பான"],
        "general": {
            "english": "A normal retina scan shows healthy retinal layers with intact foveal depression, no fluid leakage, no abnormal vessel growth, and no cellular deposits.",
            "tamil": "இயல்பான விழித்திரை ஸ்கேன் என்பது ஆரோக்கியமான விழித்திரை அடுக்குகளைக் காட்டும். இதில் மேக்குலா குழிவு சரியாக இருக்கும், திரவக் கசிவு அல்லது கொழுப்புப் படிவுகள் போன்ற எந்தவொரு அசாதாரணமும் இருக்காது.",
            "tanglish": "Normal retina scan-la ella layers-um correct structure-la irukum. Macular contour perfect-ah irukum. Liquid leakage, blood vessel abnormal growth or drusen pads ethuvumee irukaathu. Eye fully healthy."
        },
        "symptoms": {
            "english": "Normal healthy eyes experience clear, sharp central and peripheral vision, accurate color perception, and no visual distortions.",
            "tamil": "இயல்பான ஆரோக்கியமான கண்கள் தெளிவான மற்றும் கூர்மையான பார்வையைப் பெறும். நிறங்கள் சரியாகத் தெரியும், பார்வை சிதைவு எதுவும் இருக்காது.",
            "tanglish": "Healthy eye symptoms: Paarvai sharp-ah clear-ah irukum. Distortions, wavy lines or dark spots ethuvume irukaadhu. Normal color vision irukum."
        },
        "treatment": {
            "english": "No medical treatment is required. Regular annual checkups are advised to maintain current ocular health.",
            "tamil": "எந்த மருத்துவ சிகிச்சையும் தேவையில்லை. கண் ஆரோக்கியத்தைத் தொடர்ந்து பராமரிக்க ஆண்டுக்கு ஒருமுறை வழக்கமான கண் பரிசோதனை போதுமானது.",
            "tanglish": "Treatment ethuvume thevaiyillai. Current condition-ai maintain panna annual routine checkup mattum panna podhum."
        },
        "prevention": {
            "english": "Maintain eye health by consuming a balanced diet, limiting screen time (20-20-20 rule), staying hydrated, and using protective eyewear during intense activities.",
            "tamil": "தடுப்பு முறைகள்: சமச்சீரான உணவு, கணினி திரையைப் பார்க்கும் நேரத்தைக் குறைத்தல் (20-20-20 விதி), போதுமான அளவு தண்ணீர் குடித்தல், பாதுகாப்பு கண்ணாடிகளை அணிதல்.",
            "tanglish": "Health maintain panna: Nutrition diet sapdanum, screen time limit pannanum (20-20-20 rule follow pannanum - every 20 mins, 20 feet thoorathai 20 seconds paarka vendum)."
        }
    },
    "AMD": {
        "name": "Age-Related Macular Degeneration (AMD)",
        "tamil_name": "வயது முதிர்வு சார்ந்த மேக்குலா சிதைவு (AMD)",
        "keywords": ["amd", "macular degeneration", "dry amd", "wet amd", "ageing", "முதிர்வு", "சிதைவு"],
        "general": {
            "english": "Age-Related Macular Degeneration (AMD) is a leading cause of vision loss in people over 50. It gradually destroys the macula, interfering with sharp, straight-ahead vision.",
            "tamil": "வயது முதிர்வு சார்ந்த மேக்குலா சிதைவு (AMD) என்பது 50 வயதுக்கு மேற்பட்டவர்களுக்கு ஏற்படும் பார்வை இழப்புக்கான முக்கிய காரணியாகும். இது மேக்குலாவை மெதுவாக சிதைத்து, நேரடி பார்வையை பாதிக்கிறது.",
            "tanglish": "AMD (Age-Related Macular Degeneration) enbathu 50 vayadhuku melulla periyavargaluku varum disease. Ithil dry AMD (drusen yerpaduvathu) matrum wet AMD (rathakkuzhal valarndhu leak aaguvathu) ena irandu vagaigal undu."
        },
        "symptoms": {
            "english": "Gradual or sudden loss of central vision, distortion of straight lines, difficulty recognizing faces, and needing brighter light to perform close tasks.",
            "tamil": "அறிகுறிகள்: மத்திய பார்வை மெதுவாக அல்லது திடீரென மறைதல், நேர்க்கோடுகள் வளைந்திருப்பது போல் தோன்றுதல், மனித முகங்களை அடையாளம் காண்பதில் சிரமம், மற்றும் பிரகாசமான வெளிச்சத்தின் தேவை.",
            "tanglish": "AMD symptoms: Central vision kurayum. Amsler grid-la straight lines wavy-ah theriyum. Muka-saadaigal (faces) clear-ah identify panna mudiyathu. Velicham adhigama thevai padum."
        },
        "treatment": {
            "english": "Dry AMD is managed with AREDS2 dietary supplements. Wet AMD is treated with regular intravitreal Anti-VEGF injections to arrest abnormal blood vessel growth.",
            "tamil": "உலர் AMD-க்கு AREDS2 வைட்டமின் சப்ளிமெண்ட்டுகள். ஈரமான AMD-க்கு அசாதாரண இரத்தக்குழாய்களைச் சுருங்கச் செய்யும் தொடர்ச்சியான Anti-VEGF இன்ஜெக்ஷன்கள்.",
            "tanglish": "Dry AMD-ku vitamin AREDS2 supplements thaan treatment. Wet AMD-a irundha regular-ah eye injection (Anti-VEGF) panni blood leakage block pannanum."
        },
        "prevention": {
            "english": "Quit smoking, maintain healthy blood pressure and cholesterol, wear sunglasses, and consume a diet rich in antioxidants, zinc, and green vegetables.",
            "tamil": "தடுப்பு முறைகள்: புகைபிடிப்பதை விடுதல், இரத்த அழுத்தம் மற்றும் கொழுப்பைக் கட்டுப்படுத்தல், சன் கிளாஸ் அணிதல், மற்றும் ஆன்டிஆக்ஸிடன்ட்கள், துத்தநாகம் கொண்ட உணவுகளை உண்ணுதல்.",
            "tanglish": "AMD management: Cigarette poga thadukanum, weight control pannanum, blood pressure management, healthy cholesterol, and green leafy greens sapdanum."
        }
    },
    "CSCR": {
        "name": "Central Serous Chorioretinopathy (CSCR)",
        "tamil_name": "மைய விழித்திரை திரவ சேகரிப்பு (CSCR)",
        "keywords": ["cscr", "central", "serous", "chorioretinopathy", "fluid leakage stress", "திரவச் சேகரிப்பு"],
        "general": {
            "english": "Central Serous Chorioretinopathy (CSCR) is a condition where fluid builds up under the retina, specifically in the macula, causing temporary or permanent visual impairment. It is heavily linked to high stress levels and corticosteroid use.",
            "tamil": "மைய விழித்திரை திரவ சேகரிப்பு (CSCR) என்பது விழித்திரைக்கு (குறிப்பாக மேக்குலாவிற்கு) அடியில் திரவம் சேர்வதாகும். இது தற்காலிக அல்லது நிரந்தர பார்வை குறைபாட்டை ஏற்படுத்தும். இது மன அழுத்தம் மற்றும் ஸ்டீராய்டு மருந்துகளுடன் தொடர்புடையது.",
            "tanglish": "CSCR na retina-vuku keezha fluid (liquid) accumulate aaguradhu. Ithu high stress (mental pressure) matrum steroid tablets/skin creams use pannuvathaal varum. Niraiya cases-la ithu target young & middle-aged men."
        },
        "symptoms": {
            "english": "Symptoms include dim or distorted central vision, objects appearing smaller (micropsia) or farther away than they are, and a dark circular spot in the center of vision.",
            "tamil": "அறிகுறிகள்: மத்திய பார்வை மங்கலாக அல்லது சிதைந்து தெரிவது, பொருட்கள் சிறியதாகவோ அல்லது தூரத்தில் இருப்பது போலவோ தோன்றுவது, மற்றும் பார்வையின் மையத்தில் ஒரு இருண்ட வட்டமான புள்ளி ஏற்படுவது.",
            "tanglish": "CSCR symptoms: Centre paarvai karuppa circular shadow madhiri theriyum. Porutkal chinna size-ah (micropsia) theriyum, thoorathila irukura madhiri irukum."
        },
        "treatment": {
            "english": "Many CSCR cases resolve spontaneously within 3 to 6 months. Active options include stress management, discontinuing corticosteroids, photodynamic therapy (PDT), or micro-pulse laser therapy.",
            "tamil": "சிகிச்சைகள்: பல சமயங்களில் இது 3 முதல் 6 மாதங்களில் தானாகவே சரியாகிவிடும். தீவிர சிகிச்சைகளாக மன அழுத்தத்தைக் குறைத்தல், ஸ்டீராய்டு பயன்பாட்டை நிறுத்துதல், PDT லேசர் சிகிச்சை ஆகியவை உள்ளன.",
            "tanglish": "CSCR treatment: Nariya cases 3-6 months-la thanave resolve aagidum. Aana stress-ai kuraikkanum, steroid injections/creams-ai stop pannanum. Chronic aana PDT laser pannanum."
        },
        "prevention": {
            "english": "Reduce psychological stress, avoid any unnecessary self-prescribed corticosteroids (nasal sprays, skin creams, inhalers), and maintain healthy sleep hygiene.",
            "tamil": "தடுப்பு முறைகள்: மன அழுத்தத்தைக் குறைத்தல், மருத்துவர் பரிந்துரைக்காத ஸ்டீராய்டு மருந்துகளை (நாசி தெளிப்பான்கள், தோல் கிரீம்கள்) தவிர்த்தல், மற்றும் சரியான தூக்கத்தைப் பேணுதல்.",
            "tanglish": "CSCR prevent panna: Mental stress illama irukanum. Steroid nasal sprays, skin creams or inhalers use pannrathai doc approval illama panna koodathu. 8 hours thoongunum."
        }
    },
    "GLAUCOMA": {
        "name": "Glaucoma",
        "tamil_name": "கண் அழுத்த நோய் (Glaucoma)",
        "keywords": ["glaucoma", "glacoma", "pressure", "optic nerve", "eye pressure", "அழுத்தம்", "கண் அழுத்தம்"],
        "general": {
            "english": "Glaucoma is a group of eye conditions that damage the optic nerve, the health of which is vital for good vision. This damage is often caused by an abnormally high pressure in your eye (intraocular pressure).",
            "tamil": "கண் அழுத்த நோய் (Glaucoma) என்பது விழி நரம்பை (Optic Nerve) பாதிக்கும் கண் நோய்களின் தொகுப்பாகும். இது பொதுவாக கண் உள்ளழுத்தம் (Intraocular Pressure) அதிகரிப்பதால் ஏற்படுகிறது. இதனை 'பார்வையின் அமைதியான திருடன்' என்றும் அழைப்பர்.",
            "tanglish": "Glaucoma enrathu kannil ulla optic nerve (paarvai narambu) pathikapadum disease. Ithil kannin intraocular pressure (IOP) adhigamaagurathal narambu pathikka pattu paarvai permanent-ah pogalaam."
        },
        "symptoms": {
            "english": "In early stages, open-angle glaucoma has no symptoms. Later, it leads to gradual loss of peripheral (side) vision, resulting in tunnel vision. Acute angle-closure glaucoma is a medical emergency with severe eye pain, headache, nausea, and blurry vision.",
            "tamil": "அறிகுறிகள்: ஆரம்பத்தில் எந்த அறிகுறியும் இருக்காது. பின்னர், பக்கவாட்டு பார்வை மெதுவாக மறைந்து நேர்க்கோட்டுப் பார்வை (Tunnel Vision) மட்டுமே இருக்கும். தீவிர கண் அழுத்த நோயின் போது கண் வலி, தலைவலி, வாந்தி இருக்கும்.",
            "tanglish": "Glaucoma symptoms: Open-angle-la aarambathil symptoms theriyathu. Silent-ah side vision (peripheral) kurayum, kadaisiyila tunnel vision (narrow straight view) aagidum. Sudden severe eye pain, headache vandha acute emergency."
        },
        "treatment": {
            "english": "Treatment aims to lower intraocular pressure using prescription eye drops (prostaglandin analogs, beta-blockers), selective laser trabeculoplasty (SLT), or filtration surgery (trabeculectomy).",
            "tamil": "சிகிச்சைகள்: கண் அழுத்தத்தைக் குறைக்கும் சொட்டு மருந்துகள், லேசர் சிகிச்சை (SLT) அல்லது கண் நீர் வடிகட்டுதலை மேம்படுத்தும் அறுவை சிகிச்சை (Trabeculectomy).",
            "tanglish": "Glaucoma treatment: Optic nerve damage-ai thadukka eye drops regular-ah podanum. Drops control pannalana SLT Laser check-up or trabeculectomy surgery panni pressure-ai kuraipaanga."
        },
        "prevention": {
            "english": "Early detection is key. Have routine eye exams with pressure tests (tonometry) and optic nerve evaluations, especially if you have a family history or are over 40.",
            "tamil": "தடுப்பு முறைகள்: ஆரம்ப கட்டக் கண்டறிதல் மிகவும் முக்கியம். குறிப்பாக 40 வயதுக்கு மேற்பட்டவர்கள் அல்லது குடும்ப வரலாறு உள்ளவர்கள் கண் அழுத்த சோதனையை (Tonometry) வழக்கமாக செய்ய வேண்டும்.",
            "tanglish": "Glaucoma prevent/manage panna: 40 vayadhuku mel eye pressure test (tonometry) and optic nerve screening regular-ah check pannanum, especially family history irundha screening must."
        }
    },
    "RD": {
        "name": "Retinal Detachment (RD)",
        "tamil_name": "விழித்திரை விலகல் / பிரிதல் (RD)",
        "keywords": ["rd", "detachment", "retinal detachment", "detached", "curtain", "flashes", "floaters", "விலகல்", "பிரிதல்"],
        "general": {
            "english": "Retinal Detachment (RD) is a medical emergency where the thin layer of tissue (the retina) at the back of the eye pulls away from its normal position, separating from underlying blood vessels that provide oxygen and nourishment.",
            "tamil": "விழித்திரை விலகல் அல்லது பிரிதல் (RD) என்பது ஒரு அவசர மருத்துவ நிலையாகும். இதில் விழித்திரை கண் சுவற்றில் இருந்து பிரிந்து விடுகிறது. இதனால் விழித்திரைக்கு இரத்த ஓட்டமும் ஆக்சிஜனும் தடைபட்டு உடனடியாக சிகிச்சை அளிக்காவிட்டால் பார்வை நிரந்தரமாகப் போய்விடும்.",
            "tanglish": "Retinal Detachment (RD) enpathu oru severe medical emergency. Kannoda pinnal ulla retina layer kan suvattril irundhu pirindhu (detach aagi) thaniyaaga vandhuvidum. Oxygen and blood circulation stop aagurathaal blindness varalaam."
        },
        "symptoms": {
            "english": "Classic warning signs include the sudden appearance of many floaters (tiny specks), flashes of light in one or both eyes, and a shadow or curtain-like block spreading across the visual field.",
            "tamil": "அறிகுறிகள்: திடீரென பார்வையில் ஏராளமான மிதப்பான்கள் (Floaters) மற்றும் மின்னல்கள் (Flashes) தோன்றுவது, மற்றும் பார்வைப்புலத்தில் ஒரு நிழல் அல்லது திரை போன்று பார்வையை மறைப்பது.",
            "tanglish": "RD warning symptoms: Sudden-ah niraiya karuppu boochigal/nooldhumbu paarvayil parappathu (floaters), light flashes adippathu (photopsia), visual field-la black curtain or shadow cover aaguradhu."
        },
        "treatment": {
            "english": "Surgical repair is required immediately. Options include pneumatic retinopexy (gas injection), scleral buckle, or vitrectomy (removal of vitreous gel and reattachment using laser/freeze treatment).",
            "tamil": "சிகிச்சைகள்: அவசர அறுவை சிகிச்சை அவசியம். நியூமேடிக் ரெட்டினோபெக்ஸி (கேஸ் இன்ஜெக்ஷன்), ஸ்கிளீரல் பக்கிள் அல்லது விட்ரெக்டமி (விழித்திரை ஒட்டுதல் அறுவை சிகிச்சை) போன்ற சிகிச்சைகள் அளிக்கப்படும்.",
            "tanglish": "RD treatment: Immediate-ah surgery pannanum. Pneumatic retinopexy (gas bubble inject pannuvathu), scleral buckling or vitrectomy surgery panni laser/cryopexy moolam retina-vai thirumba ottuvaanga."
        },
        "prevention": {
            "english": "Promptly seek medical attention if you experience flashes of light or new floaters. Wear protective eyewear to avoid ocular trauma, and undergo regular retinal exams if you are highly myopic.",
            "tamil": "தடுப்பு முறைகள்: மின்னல் கீற்றுகள் அல்லது புதிய மிதப்பான்கள் தெரிந்தால் உடனே கண் மருத்துவரை அணுக வேண்டும். காயம் ஏற்படாமல் கண்களைப் பாதுகாக்க வேண்டும். அதிக கிட்டப்பார்வை (high myopia) உள்ளவர்கள் வழக்கமான விழித்திரை பரிசோதனை செய்ய வேண்டும்.",
            "tanglish": "RD management: Light flashes or sudden floaters vandha delay pannama doc-ai paarkanum. High myopia (minus power eye glasses) irukuravanga retina-vai safe-ah check pannanum. Eye injury varama pathukanum."
        }
    },
    "RP": {
        "name": "Retinitis Pigmentosa (RP)",
        "tamil_name": "விழித்திரை நிறமி அழற்சி நோய் (RP)",
        "keywords": ["rp", "retinitis", "pigmentosa", "night blindness", "hereditary", "genetic", "மாலக்கண்", "நிறமி"],
        "general": {
            "english": "Retinitis Pigmentosa (RP) is a group of rare genetic disorders that involve a breakdown and loss of cells in the retina (rod cells deteriorate first, followed by cone cells). It is a progressive disease that leads to severe vision loss.",
            "tamil": "விழித்திரை நிறமி அழற்சி நோய் (RP) என்பது ஒரு அரிய மரபணு ரீதியான கண் நோயாகும். இதில் விழித்திரையில் உள்ள செல்கள் (முதலில் தடிச் செல்கள், பின்னர் கூம்புச் செல்கள்) படிப்படியாக சிதைவடைந்து பார்வை இழப்பை உண்டாக்குகின்றன.",
            "tanglish": "Retinitis Pigmentosa (RP) enpathu parambarai parambaraiyaga varum oru genetic eye disease. Kannil ulla rod cells and cone cells padipadiyaga saavadhaaal paarvai padipadiyaga kuraiyum."
        },
        "symptoms": {
            "english": "Initial symptoms include night blindness (difficulty seeing in the dark) and a progressive loss of peripheral vision, eventually leading to tunnel vision. Central vision is often preserved until late stages.",
            "tamil": "அறிகுறிகள்: ஆரம்பத்தில் மாலைக்கண் நோய் (இருட்டில் பார்ப்பதில் தீவிர சிரமம்), பக்கவாட்டு பார்வை படிப்படியாக குறைந்து நடுப்பார்வை மட்டுமே மிஞ்சுதல் (Tunnel vision).",
            "tanglish": "RP symptoms: Maalakkan noi (night blindness) - iruttil paarvai theriyathu. Pinpu sides-la ulla view kuraiyum. Naduvula mattum circular-ah view theriyum (tunnel vision)."
        },
        "treatment": {
            "english": "There is no cure for RP. Progression may be slowed by high-dose Vitamin A palmitate, low-vision aids, Argus II retinal prosthesis (bionic eye) for advanced stages, or gene therapy (Luxturna) for specific gene mutations.",
            "tamil": "சிகிச்சைகள்: இதற்கு நிரந்தர தீர்வு இல்லை. இருப்பினும், வைட்டமின் ஏ பால்மிட்டேட் மாத்திரைகள், பார்வைத்திறன் உதவிகள், குறிப்பிட்ட சில மரபணு மாற்றங்களுக்கு ஜீன் தெரபி (Luxturna) ஆகியவை பயன்படுத்தப்படுகின்றன.",
            "tanglish": "RP-ku permanent cure illa. High dose Vitamin A palmitate capsules slow panna tharuvaanga. Low-vision aids (magnifiers) use pannanum. Specific gene mutation-ku Gene Therapy (Luxturna) surgery or Argus II bionic eye use panlaam."
        },
        "prevention": {
            "english": "Since RP is inherited, genetic counseling is recommended. Wear sunglasses to protect the retina from light damage and maintain regular ophthalmic checkups.",
            "tamil": "தடுப்பு முறைகள்: இது பரம்பரை நோய் என்பதால் மரபணு ஆலோசனை (Genetic Counseling) பெறுதல் நல்லது. வெளிச்சத்தால் ஏற்படும் விழித்திரை பாதிப்பைத் தடுக்க சன் கிளாஸ் அணியவும்.",
            "tanglish": "RP management: Genetic counseling eduthukanum. Sunshine ultraviolet rays-la irundhu retina-vai protect panna dark sunglasses wear pannanum. Health checkup regular-ah pannanum."
        }
    },
    "AMD_CSCR_ERM_MH_GLAUCOMA": {
        "name": "Other Retinal Conditions",
        "tamil_name": "இதர விழித்திரை நோய்கள்",
        "keywords": ["erm", "mh", "macular hole", "epiretinal membrane", "membrane", "hole", "துளை", "சவ்வு"],
        "general": {
            "english": "Epiretinal Membrane (ERM) is a thin sheet of fibrous tissue that develops on the surface of the macula, causing distortion. Macular Hole (MH) is a small break in the macula that causes central vision blurring.",
            "tamil": "விழித்திரை மேல் சவ்வு (ERM) என்பது மேக்குலாவின் மேல் வளரும் ஒரு மெல்லிய சவ்வு ஆகும். மேக்குலா துளை (MH) என்பது மத்திய பார்வையை பாதிக்கும் மேக்குலாவில் ஏற்படும் ஒரு சிறிய துளையாகும்.",
            "tanglish": "Epiretinal Membrane (ERM) na macula mela thin paper-madhiri fibrous sheet valarundhu distortion tharum. Macular Hole (MH) na central retina padala-thula chinna hole vilundhu centre view blurring tharum."
        },
        "symptoms": {
            "english": "ERM symptoms include mild visual distortion (straight lines look wavy). Macular Hole (MH) symptoms include a central gray spot, severe blurring, or a blind spot in central vision.",
            "tamil": "ERM: லேசான பார்வை கோணல். மேக்குலா துளை: மத்திய பார்வையில் சாம்பல் நிற புள்ளி, தீவிர மங்கல் அல்லது முழுமையான பார்வையின்மை.",
            "tanglish": "ERM signs: Lines wavy-ah theriyum. Macular Hole signs: Centre point gray/black circular patch form aaguradhu, detail read panna mudiyamal blur aaguradhu."
        },
        "treatment": {
            "english": "Both ERM and Macular Hole are treated surgically via vitrectomy to peel the membrane or closure of the hole using a gas bubble. Normal drops do not cure them.",
            "tamil": "இரண்டிற்கும் விட்ரெக்டமி (Vitrectomy) அறுவை சிகிச்சை மூலம் சவ்வை நீக்குதல் அல்லது வாயு குமிழ் மூலம் துளையை அடைத்தல் ஆகிய சிகிச்சைகளே தீர்வாகும். சாதாரண சொட்டு மருந்துகள் குணப்படுத்தாது.",
            "tanglish": "Drops thirumba help pannathu. ERM membranes-ai surgery (vitrectomy) panni peel pannanum. Macular Hole-a irundha vitrectomy panni gas bubble inject panni face down position maintain pannanum."
        },
        "prevention": {
            "english": "Preventative steps are limited. Avoid eye trauma, manage systemic vascular disease, and get regular dilated exams.",
            "tamil": "தடுப்பு முறைகள்: கண் காயங்களைத் தவிர்த்தல், இரத்த அழுத்தம் மற்றும் சர்க்கரையைக் கட்டுக்குள் வைத்தல், விழித்திரை பரிசோதனை செய்தல்.",
            "tanglish": "Management: Trauma kaniyil padama padhukanum, eye checkup regular-ah maintain pannanum."
        }
    }
}

# Conversational phrases and greetings
CONVERSATIONAL = {
    "english": {
        "greetings": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"],
        "greeting_response": "Hello! I am your VisionCare AI Clinical Assistant. How can I help you today? You can ask me about retinal diseases, symptoms, diagnosis, and treatments.",
        "how_are_you": ["how are you", "how's it going", "how are you doing", "how do you do"],
        "how_are_you_response": "I am doing well, thank you for asking! I am ready to answer any questions about retinal health and OCT imaging. What condition would you like to discuss?",
        "thank_you": ["thanks", "thank you", "thank", "appreciate it", "awesome", "perfect"],
        "thank_you_response": "You're very welcome! If you have any other questions about retinal health, symptoms, or OCT analysis, feel free to ask. Stay healthy!",
        "not_found": "I understand you are asking a medical question. However, I specialize in retinal diseases and eye-related conditions. I can provide detailed information about Diabetic Macular Edema (DME), Choroidal Neovascularization (CNV), Drusen, Age-Related Macular Degeneration (AMD), Central Serous Chorioretinopathy (CSCR), Glaucoma, Retinal Detachment (RD), and Retinitis Pigmentosa (RP). Which of these conditions would you like to discuss?"
    },
    "tamil": {
        "greetings": ["வணக்கம்", "நலம்", "காலை வணக்கம்", "மாலை வணக்கம்"],
        "greeting_response": "வணக்கம்! நான் உங்கள் விஷன் கேர் AI மருத்துவ உதவியாளர். இன்று உங்களுக்கு நான் எவ்வாறு உதவ முடியும்? விழித்திரை நோய்கள், அறிகுறிகள் மற்றும் சிகிச்சைகள் பற்றி நீங்கள் என்னிடம் கேட்கலாம்.",
        "how_are_you": ["எப்படி இருக்கிறீர்கள்", "நலமா", "எப்படி இருக்கீங்க"],
        "how_are_you_response": "நான் நன்றாக இருக்கிறேன், கேட்டதற்கு நன்றி! விழித்திரை ஆரோக்கியம் மற்றும் OCT பகுப்பாய்வு பற்றிய கேள்விகளுக்கு பதிலளிக்க நான் தயாராக உள்ளேன். நீங்கள் எதைப் பற்றி அறிய விரும்புகிறீர்கள்?",
        "thank_you": ["நன்றி", "மிகவும் நன்றி", "நன்றிகள்"],
        "thank_you_response": "மிக்க நன்றி! விழித்திரை ஆரோக்கியம் அல்லது OCT பகுப்பாய்வு பற்றி வேறு ஏதேனும் கேள்விகள் இருந்தால் தயங்காமல் கேளுங்கள். ஆரோக்கியமாக இருங்கள்!",
        "not_found": "நான் உங்கள் மருத்துவக் கேள்வியைப் புரிந்து கொள்கிறேன். இருப்பினும், நான் விழித்திரை நோய்கள் மற்றும் கண் சார்ந்த குறைபாடுகளில் மட்டுமே நிபுணத்துவம் பெற்றுள்ளேன். என்னால் நீரிழிவு விழித்திரை நோய் (DME), விழித்திரைக்கீழ் புதிய இரத்தக்குழாய் உருவாக்கம் (CNV), விழித்திரை படிவுகள் (Drusen), கண் அழுத்த நோய் (Glaucoma), விழித்திரை விலகல் (RD) அல்லது மாலைக்கண் நோய் (RP) பற்றி விரிவான தகவல்களை வழங்க முடியும். இவற்றில் எதைப் பற்றி அறிய விரும்புகிறீர்கள்?"
    },
    "tanglish": {
        "greetings": ["vanakkam", "hello", "hi", "hey", "epdi irukinga", "nalama", "epadi irukinga", "good morning", "vanakam"],
        "greeting_response": "Vanakkam! Naan unga VisionCare AI Assistant. Iniku ungaluku epdi help pannanum? Retinal diseases, symptoms, testing, treatments pathi ketkaalam.",
        "how_are_you": ["how are you", "epdi irukinga", "nalama", "epadi irukinga", "how you doing"],
        "how_are_you_response": "Naan nalla iruken, thanks! Retina health matrum OCT scans pathi unga questions-ku answer panna ready-ah iruken. Enna condition pathi ketka poreenga?",
        "thank_you": ["nandri", "thanks", "thank you", "romba nandri", "nandrigal"],
        "thank_you_response": "Romba nandri! Retina related vera ethavathu queries irundha dharalama ketkaalam. Take care!",
        "not_found": "Puriyudhu, neenga medical question ketkuriga. Aana naan retinal diseases matrum eye conditions-la thaan specialize panni iruken. Ennala DME, CNV, Drusen, AMD, CSCR, Glaucoma, Retinal Detachment (RD), and Retinitis Pigmentosa (RP) pathi detailed information thara mudiyum. Idhula ungaluku enna disease pathi theriyanum?"
    }
}

DECLINE_MESSAGES = {
    "english": "Sorry, I am a Medical AI Assistant and can only answer questions related to medical topics, retinal diseases, eye health, diagnosis, treatments, and related healthcare information. Please ask a medical-related question.",
    "tamil": "மன்னிக்கவும், நான் ஒரு மருத்துவ AI உதவியாளர். என்னால் மருத்துவ தலைப்புகள், விழித்திரை நோய்கள், கண் ஆரோக்கியம், கண்டறிதல், சிகிச்சைகள் மற்றும் தொடர்புடைய சுகாதார தகவல்கள் பற்றிய கேள்விகளுக்கு மட்டுமே பதிலளிக்க முடியும். தயவுசெய்து மருத்துவம் தொடர்பான கேள்வியைக் கேளுங்கள்.",
    "tanglish": "Sorry, naan oru Medical AI Assistant. Ennala medical topics, retinal diseases, eye health, diagnosis, treatments, matrum healthcare related questions-ku mattum thaan answer panna mudiyum. Dayavu seidhu medical related-ah kelpinga."
}

def detect_language(text: str) -> str:
    """Detect whether text is Tamil, Tanglish, or English."""
    cleaned = text.strip().lower()
    
    # Check for Tamil Unicode character range
    if any('\u0b80' <= char <= '\u0bff' for char in text):
        return "tamil"
    
    # Tanglish vocabulary check
    tanglish_words = {
        "enna", "epdi", "epadi", "iruku", "irukinga", "ku", "oda", "pathu", "patha", "panna", "pannalaam", 
        "marundhu", "sikichai", "kan", "paarvai", "pathipu", "varum", "vazhi", "vali", "paarkanum", "unna",
        "theriyum", "theriyala", "nalla", "nalama", "vanakkam", "vanakam", "iniku", "kannula", "ratha", "kuzhal",
        "neer", "veekkam", "karuppu", "pulli", "mangalaga", "valainthu", "muthal", "kadaisi", "illama", "koodathu"
    }
    
    words = re.findall(r'\b[a-z]+\b', cleaned)
    if not words:
        return "english"
        
    tanglish_count = sum(1 for w in words if w in tanglish_words)
    # If a good portion of words are Tanglish indicators, classify as Tanglish
    if tanglish_count > 0 or (len(words) < 4 and any(w in ["epdi", "enna", "iruku", "ok", "nandri"] for w in words)):
        return "tanglish"
        
    return "english"

def is_medical_query(query: str) -> bool:
    """Check if the search query is related to medical or eye-related topics."""
    cleaned = query.strip().lower()
    
    medical_keywords = {
        # English eye terms
        "eye", "eyes", "retina", "retinal", "macula", "macular", "fovea", "vision", "blindness", "sight", "optic",
        "nerve", "cornea", "lens", "cataract", "pupil", "iris", "photoreceptor", "rods", "cones", "oct", "fundus",
        "ophthalmologist", "doctor", "clinical", "scan", "image", "visual", "glasses", "spectacles", "myopia", 
        "hyperopia", "astigmatism", "photopsia", "floaters", "flashes", "blind", "glaucoma", "dme", "cnv", "drusen",
        "amd", "cscr", "erm", "mh", "rd", "rp", "edema", "degeneration", "detachment", "pigmentosa", "membrane",
        "diabetic", "choroidal", "neovascularization", "hole",
        # English general medical terms
        "disease", "illness", "symptom", "symptoms", "treatment", "treatments", "cure", "medicine", "medicines",
        "drug", "drugs", "pill", "pills", "injection", "injections", "surgery", "surgeries", "laser", "therapy",
        "prevention", "prevent", "cause", "causes", "diagnose", "diagnosis", "hospital", "clinic", "patient",
        "health", "healthy", "healthcare", "medical", "care", "body", "pain", "blood", "sugar", "diabetes", 
        "pressure", "hypertension", "cardiovascular", "stress", "steroid", "steroids", "infection", "infectious",
        "genetic", "inherited", "gene", "mutation", "specialist", "cortex", "vitreous", "aqueous", "intraocular",
        "iop", "tonometry", "amsler", "grid", "nutrition", "diet", "lifestyle", "ill", "ache",
        # Tamil eye/medical terms
        "விழித்திரை", "கண்", "பார்வை", "நீர்வீக்கம்", "இரத்தக்குழாய்", "படிவுகள்", "முதிர்வு", "சிதைவு", "அழுத்தம்",
        "விலகல்", "பிரிதல்", "நிறமி", "அழற்சி", "மாலக்கண்", "சவ்வு", "துளை", "அறிகுறிகள்", "சிகிச்சை", "சிகிச்சைகள்",
        "மருந்து", "மாத்திரை", "ஊசி", "அறுவை", "தடுப்பு", "நோய்", "நோய்கள்", "உடல்", "இரத்தம்", "சர்க்கரை", "வலி",
        "மருத்துவ", "சுகாதார", "நோயாளி",
        # Tanglish terms
        "kannu", "kan", "retina", "paarvai", "pathipu", "varum", "vazhi", "vali", "paarkanum", "marundhu", 
        "sikichai", "neer", "veekkam", "karuppu", "pulli", "mangalaga", "valainthu", "rathakkuzhal", "rathakuzhal",
        "sugar", "diabetes", "diabetic", "pressure", "injection", "laser", "surgery", "doctor", "doc", "prevent",
        "symptoms", "symptom", "treatment", "cure", "medicine", "pill", "steroid", "stress", "diet", "food",
        "walking", "excercise", "checkup", "screening", "scanning", "scan", "floaters", "flashes"
    }
    
    # Split query into words
    words = re.findall(r'\b[a-z0-9_-]+\b', cleaned)
    tamil_words = re.findall(r'[\u0b80-\u0bff]+', cleaned)
    all_query_words = set(words + tamil_words)
    
    for kw in medical_keywords:
        # For Tamil terms, perform a simple substring match
        if any('\u0b80' <= char <= '\u0bff' for char in kw):
            if kw in cleaned:
                return True
        else:
            # For English/Tanglish, check if it's a word prefix or exact match in query words
            for qw in all_query_words:
                if qw.startswith(kw) or (len(kw) > 3 and kw in qw):
                    return True
                    
    # Also check basic healthcare context words
    extra_keywords = ["health", "care", "lifestyle", "ill", "ache", "pain", "hospital", "clinic", "medical", "disease", "treatment", "doctor", "doc"]
    if any(ek in cleaned for ek in extra_keywords):
        return True
        
    return False

def process_chat_query(query: str) -> dict[str, str]:
    """Retrieve medical context and generate response in user's detected language."""
    detected_lang = detect_language(query)
    cleaned_query = query.strip().lower()
    
    # Check for simple conversational responses
    conv_data = CONVERSATIONAL[detected_lang]
    
    # Check greetings
    for greeting in conv_data["greetings"]:
        if greeting in cleaned_query and len(cleaned_query.split()) <= 3:
            return {"response": conv_data["greeting_response"], "language": detected_lang}
            
    # Check how are you
    for hay in conv_data["how_are_you"]:
        if hay in cleaned_query:
            return {"response": conv_data["how_are_you_response"], "language": detected_lang}
            
    # Check thank you
    for ty in conv_data["thank_you"]:
        if ty in cleaned_query:
            return {"response": conv_data["thank_you_response"], "language": detected_lang}

    # Restrict response scope to medical queries only (exclude conversational greetings checked above)
    if not is_medical_query(cleaned_query):
        return {"response": DECLINE_MESSAGES[detected_lang], "language": detected_lang}

    # RAG Search Engine
    # Target intents
    intents = {
        "symptoms": ["symptom", "signs", "symptoms", "effects", "feel", "warning", "அறிகுறிகள்", "அறிகுறி", "epdi irukum", "signs enna", "symptoms enna", "epdi theriyum"],
        "treatment": ["treatment", "cure", "medicine", "injection", "laser", "surgery", "treat", "சிகிச்சை", "மருந்து", "ஊசி", "அறுவை", "therapy", "injection podalaama", "marundhu enna", "treatment enna", "epdi treat", "cure panna"],
        "prevention": ["prevent", "avoid", "prevention", "protect", "diet", "lifestyle", "exercise", "Amsler", "தடுப்பு", "தடுக்க", "management", "epdi prevent", "prevent panna", "diet enna", "food enna", "lifestyle tips"]
    }
    
    detected_intent = "general" # fallback default
    for intent_name, keywords in intents.items():
        if any(keyword in cleaned_query for keyword in keywords):
            detected_intent = intent_name
            break
            
    # Target disease
    matched_disease = None
    max_score = 0
    
    for disease_key, info in KNOWLEDGE_BASE.items():
        score = 0
        # Match disease key/code
        if disease_key.lower() in cleaned_query:
            score += 10
            
        # Match disease keywords
        for keyword in info["keywords"]:
            if keyword.lower() in cleaned_query:
                score += 5
                
        # Match full/extended labels
        if disease_key == "AMD" and ("macular degeneration" in cleaned_query or "வயது முதிர்வு" in cleaned_query):
            score += 8
        if disease_key == "RD" and ("detachment" in cleaned_query or "விலகல்" in cleaned_query):
            score += 8
        if disease_key == "RP" and ("retinitis" in cleaned_query or "pigmentosa" in cleaned_query or "மாலைக்கண்" in cleaned_query):
            score += 8
            
        if score > max_score:
            max_score = score
            matched_disease = disease_key
            
    # Generate RAG response
    if matched_disease and max_score >= 3:
        disease_info = KNOWLEDGE_BASE[matched_disease]
        content = disease_info[detected_intent][detected_lang]
        
        disease_name = disease_info["tamil_name"] if detected_lang == "tamil" else disease_info["name"]
        
        # Structure Response
        if detected_lang == "english":
            response = f"### **{disease_name} - {detected_intent.upper()} INFO**\n\n"
            response += f"{content}\n\n"
            response += f"*Disclaimer: This clinical information is retrieved automatically for reference. Always consult a retinal specialist for formal diagnosis.*"
        elif detected_lang == "tamil":
            intent_tamil = {"general": "பொதுவான தகவல்", "symptoms": "அறிகுறிகள்", "treatment": "சிகிச்சை முறைகள்", "prevention": "தடுப்பு மற்றும் மேலாண்மை"}[detected_intent]
            response = f"### **{disease_name} - {intent_tamil}**\n\n"
            response += f"{content}\n\n"
            response += f"*எச்சரிக்கை: இந்த மருத்துவ தகவல் வழிகாட்டுதலுக்காக மட்டுமே வழங்கப்படுகிறது. முறையான பரிசோதனைக்கு கண் மருத்துவரை அணுகவும்.*"
        else: # tanglish
            intent_tanglish = {"general": "General Details", "symptoms": "Symptoms & Signs", "treatment": "Treatment Details", "prevention": "Prevention & Care"}[detected_intent]
            response = f"### **{disease_name} - {intent_tanglish}**\n\n"
            response += f"{content}\n\n"
            response += f"*Note: Indha information retrieval assistant moolam tharapattadhu. Accurate check-up-ku oru retinal specialist doctor-ai compulsory paarkanum.*"
            
        return {"response": response, "language": detected_lang}
        
    # Check general retina health query
    if "retina" in cleaned_query or "eye" in cleaned_query or "விழித்திரை" in cleaned_query or "கண்" in cleaned_query:
        if detected_lang == "english":
            resp = "### **General Retinal Health & Care**\n\nTo keep your retina healthy:\n1. Consome leafy green vegetables, citrus fruits, and omega-3 rich fish.\n2. Schedule an annual dilated eye screening if you have diabetes or high blood pressure.\n3. Wear UV-protection sunglasses in bright sun.\n4. Control blood glucose and cholesterol levels."
        elif detected_lang == "tamil":
            resp = "### **விழித்திரை ஆரோக்கியம்**\n\nவிழித்திரையைப் பாதுகாக்க:\n1. கீரைகள், சிட்ரஸ் பழங்கள் மற்றும் ஒமேகா-3 மீன் வகைகளை உணவில் சேர்க்கவும்.\n2. சர்க்கரை அல்லது இரத்த அழுத்தம் இருப்பின் ஆண்டுக்கு ஒருமுறை கண் பரிசோதனை செய்யவும்.\n3. வெயிலில் செல்லும்போது UV சன்கிளாஸ் அணியவும்.\n4. இரத்த சர்க்கரையை சீராக வைக்கவும்."
        else: # tanglish
            resp = "### **Retina Health & Care Tips**\n\nRetina healthy-ah irukka:\n1. Green vegetables, fruits matrum fish sapdanum.\n2. Diabetes/BP irundha annual screening check up pannanum.\n3. Sun-la UV protective glass pottukanum.\n4. Sugar control panni proper-ah diet maintain pannanum."
        return {"response": resp, "language": detected_lang}
        
    # Return conversational not found fallback
    return {"response": conv_data["not_found"], "language": detected_lang}
