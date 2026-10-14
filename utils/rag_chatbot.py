"""AI-powered Retinal Healthcare RAG Chatbot Assistant.
Integrates LangChain, FAISS, and Llama 3 simulations with Sentence Transformers.
"""

from __future__ import annotations
import os
import re
import urllib.parse
import numpy as np
from typing import Any, Optional, Dict, List
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.prompts import PromptTemplate

# Trusted Medical Knowledge Base
RAG_KNOWLEDGE_BASE = {
    "DR": {
        "name": "Diabetic Retinopathy (DR) & Diabetic Macular Edema (DME)",
        "tamil_name": "நீரிழிவு விழித்திரை நோய் மற்றும் மேக்குலா நீர்வீக்கம் (DR/DME)",
        "keywords": ["dr", "dme", "diabetic retinopathy", "diabetic macular edema", "sugar retina", "fluid leak", "neerveekkam", "நீர்வீக்கம்", "சர்க்கரை", "இரத்தக்கசிவு"],
        "causes": {
            "english": "Diabetic Retinopathy and DME are caused by chronic high blood sugar from diabetes, which damages the tiny blood vessels (capillaries) in the retina. These damaged vessels leak fluid, lipids, and blood into the retina and macula, causing swelling and ischemia.",
            "tamil": "நீரிழிவு நோயினால் ஏற்படும் நாள்பட்ட உயர் இரத்த சர்க்கரை விழித்திரையில் உள்ள சிறிய இரத்தக்குழாய்களை சேதப்படுத்துகிறது. இதனால் திரவங்கள் மற்றும் கொழுப்பு கசிந்து மேக்குலா பகுதியில் நீர்வீக்கம் மற்றும் இரத்தக்கசிவு ஏற்படுகிறது.",
            "tanglish": "Sugar level romba naal control illama irukardhala, kannula ulla tiny blood vessels damage aagum. Adhula irundhu fluid matrum blood leak aagi macula-la swelling (neerveekkam) yerpaduthum."
        },
        "symptoms": {
            "english": "Symptoms include blurry or wavy central vision, floaters (dark moving spots), faded colors, difficulty reading, fluctuating vision, and blind spots in the center of sight.",
            "tamil": "மத்திய பார்வை மங்கலாகவோ அல்லது வளைந்த கோடுகளாகவோ தெரிவது, பார்வையில் மிதக்கும் கரும்புள்ளிகள் (Floaters), நிறங்கள் மங்கித் தோன்றுவது, படிப்பதில் சிரமம் மற்றும் பார்வை இழப்பு ஆகியவை இதன் அறிகுறிகளாகும்.",
            "tanglish": "Paarvai mangalaga theriyum, straight lines valainju (wavy) theriyum. Kannil karuppu pulligal (floaters) parrakuradhu madhiri irukum. Colors light-ah faded-ah theriyum."
        },
        "risk_factors": {
            "english": "Uncontrolled blood glucose (high HbA1c), long duration of diabetes, hypertension (high blood pressure), high cholesterol (hyperlipidemia), and pregnancy.",
            "tamil": "கட்டுப்படுத்தப்படாத இரத்த சர்க்கரை, நீண்ட கால நீரிழிவு நோய், உயர் இரத்த அழுத்தம், அதிக கொழுப்பு மற்றும் கர்ப்ப காலம்.",
            "tanglish": "Blood sugar control illama irupathu, diabetes vandhu nariya varusham aagiyirupathu, high BP, cholesterol levels, matrum pregnancy."
        },
        "diagnosis": {
            "english": "Optical Coherence Tomography (OCT) scans measure retinal thickness and fluid pockets. Fundus photography and Fluorescein Angiography (FA) map blood vessel leakage.",
            "tamil": "விழித்திரை தடிமனை அளவிட ஆப்டிகல் கோஹரன்ஸ் டோமோகிராபி (OCT) ஸ்கேன் செய்யப்படுகிறது. கண் வண்ண புகைப்படம் மற்றும் இரத்தக்குழாய் கசிவை அறிய FA பரிசோதனை செய்யப்படுகிறது.",
            "tanglish": "OCT scan vechu retina thickness-ai accurate-ah check pannalaam. Fundus photo matrum blood vessel leakage check panna Fluorescein Angiography (FA) test pannuvaanga."
        },
        "prevention": {
            "english": "Maintain HbA1c below 7%, manage hypertension and cholesterol, follow a diabetic diet, exercise regularly, and undergo annual dilated eye exams.",
            "tamil": "HbA1c சர்க்கரை அளவை 7%-க்கு கீழ் வைத்திருங்கள், இரத்த அழுத்தம் மற்றும் கொழுப்பைக் கட்டுப்படுத்துங்கள், சமச்சீரான உணவு, வழக்கமான உடற்பயிற்சி மற்றும் ஆண்டுக்கொருமுறை கண் பரிசோதனை செய்யுங்கள்.",
            "tanglish": "HbA1c levels 7%-ku kulla vachukanum. BP matrum cholesterol control pannanum. Healthy diet, regular walking, sugar check and yearly eye screening must."
        },
        "treatment": {
            "english": "First-line treatments include intravitreal Anti-VEGF injections (e.g., Lucentis, Eylea) to stop vessel leakage, panretinal photocoagulation (PRP laser) to shrink abnormal vessels, and vitrectomy surgery for severe bleeding.",
            "tamil": "கசிவைத் தடுக்க கண்ணுக்குள் போடப்படும் Anti-VEGF இன்ஜெக்ஷன்கள் முதன்மை சிகிச்சையாகும். லேசர் சிகிச்சை (PRP) மற்றும் தீவிர இரத்தக்கசிவுக்கு விட்ரெக்டமி அறுவை சிகிச்சை ஆகியவை செய்யப்படுகின்றன.",
            "tanglish": "Kannula Anti-VEGF injections poduvaanga fluid leakage stop panna. New blood vessels shrinkage-ku PRP Laser checkup or severe bleeding condition-ku Vitrectomy surgery recommended."
        },
        "research_insights": {
            "english": "Clinical trials (such as DRCR.net Protocol T) show that Anti-VEGF agents are highly effective, with earlier treatment initiation yielding significantly better visual acuity outcomes.",
            "tamil": "சமீபத்திய மருத்துவ ஆய்வுகள் (DRCR.net போன்றவை) ஆரம்ப கட்டத்திலேயே Anti-VEGF சிகிச்சையைத் தொடங்குவது நோயாளியின் பார்வைத்திறனை மிகச் சிறந்த முறையில் மேம்படுத்துகிறது எனக் காட்டுகின்றன.",
            "tanglish": "Recent trials (DRCR.net Protocol T) solrathu enna na, Anti-VEGF injection treatment-ai seekiramave arambicha paarvai nalla improve aagum."
        },
        "faqs": [
            {"q": "Can DME lead to permanent blindness?", "a": "If left untreated, chronic macular swelling can cause irreversible damage to photoreceptors, leading to severe permanent central vision loss."},
            {"q": "How often do I need injections?", "a": "Initially, injections are often given monthly, and then spaced out based on how well the macula responds to treatment."}
        ]
    },
    "GLAUCOMA": {
        "name": "Glaucoma",
        "tamil_name": "கண் அழுத்த நோய் (Glaucoma)",
        "keywords": ["glaucoma", "optic nerve", "eye pressure", "iop", "அழுத்தம்", "கண் அழுத்தம்", "பக்கவாட்டு பார்வை"],
        "causes": {
            "english": "Glaucoma occurs when the normal fluid drainage in the eye is restricted, leading to an increase in intraocular pressure (IOP). This elevated pressure damages the sensitive fibers of the optic nerve, leading to peripheral vision loss.",
            "tamil": "கண்ணில் உள்ள திரவ வடிகால் தடைபடும் போது கண் உள்ளழுத்தம் (IOP) அதிகரிக்கிறது. இந்த அதிக அழுத்தம் விழி நரம்பை (Optic Nerve) சேதப்படுத்தி பக்கவாட்டு பார்வையை குறைக்கிறது.",
            "tanglish": "Kannil ulla fluid drainage proper-ah illana, eye-kulla pressure (IOP) adhigamaagum. Indha pressure paarvai narambai (optic nerve) padithu damage pannum."
        },
        "symptoms": {
            "english": "Open-angle glaucoma is a 'silent thief of sight' with no early symptoms. Eventually, it causes gradual loss of peripheral (side) vision, leading to tunnel vision. Acute angle-closure glaucoma causes sudden eye pain, headaches, halos, and nausea.",
            "tamil": "ஆரம்பத்தில் எந்த அறிகுறியும் இருக்காது. பின்னர் பக்கவாட்டு பார்வை படிப்படியாகக் குறைந்து நேர்க்கோட்டுப் பார்வை (Tunnel Vision) மட்டுமே மிஞ்சும். தீவிர கண் அழுத்த நோயின் போது திடீர் கண் வலி மற்றும் தலைவலி ஏற்படும்.",
            "tanglish": "Silent disease-nu solluvanga, start-la symptoms theriyadhu. Sides-la view padipadiya kurayum (tunnel vision). Sudden severe pain matrum headache acute glaucoma emergency sign."
        },
        "risk_factors": {
            "english": "Age over 40, family history of glaucoma, high intraocular pressure, diabetes, extreme nearsightedness (myopia), and prolonged steroid use.",
            "tamil": "40 வயதுக்கு மேற்பட்டவர்கள், குடும்பத்தில் கண் அழுத்த நோய் வரலாறு உள்ளவர்கள், அதிக கண் அழுத்தம், நீரிழிவு நோய் மற்றும் கிட்டப்பார்வை உள்ளவர்கள்.",
            "tanglish": "40 vayadhuku mel irupavargal, family history, high eye pressure, severe myopia, matrum long-term steroid drops use."
        },
        "diagnosis": {
            "english": "Diagnosed via tonometry (measuring IOP), ophthalmoscopy (optic nerve evaluation), visual field testing (perimetry), and OCT scanning of the optic nerve head (RNFL analysis).",
            "tamil": "கண் அழுத்தம் அளவிடுதல் (Tonometry), விழி நரம்பு பரிசோதனை, பார்வைப்புல சோதனை (Perimetry) மற்றும் விழி நரம்பின் OCT ஸ்கேன் மூலம் கண்டறியப்படுகிறது.",
            "tanglish": "Tonometry test vechu eye pressure measure pannalaam. Optic nerve shape check pannanum matrum visual field test (perimetry) vechu side vision-ai verify pannuvaanga."
        },
        "prevention": {
            "english": "Vision loss from glaucoma is irreversible, so early detection is vital. Undergo comprehensive dilated eye exams every 1 to 2 years, especially if risk factors are present.",
            "tamil": "கண் அழுத்த நோயால் ஏற்படும் பார்வை இழப்பை மீட்டெடுக்க முடியாது. எனவே 40 வயதுக்கு மேற்பட்டவர்கள் மற்றும் ஆபத்துக் காரணிகள் உள்ளவர்கள் தவறாமல் கண் பரிசோதனை செய்ய வேண்டும்.",
            "tanglish": "Glaucoma-vukku permanent cure kidayathu, so primary checkup is important. 40 vayadhuku mel eye BP test regular-ah pannanum."
        },
        "treatment": {
            "english": "Treatments include prescription eye drops to lower IOP (prostaglandin analogs, beta-blockers), selective laser trabeculoplasty (SLT), or surgical drainage procedures like trabeculectomy.",
            "tamil": "கண் அழுத்தத்தைக் குறைக்கும் பிரத்யேக கண் சொட்டு மருந்துகள், லேசர் சிகிச்சை (SLT) அல்லது கண் நீரை வடிக்கச் செய்யும் அறுவை சிகிச்சைகள் (Trabeculectomy) அளிக்கப்படுகின்றன.",
            "tanglish": "Pressure-ai kuraika daily eye drops regular-ah podanum. Pressure drops-la control aagalana SLT laser treatment or Trabeculectomy surgery pannuvaanga."
        },
        "research_insights": {
            "english": "Studies indicate that early medical or laser intervention (like the LiGHT trial recommending SLT laser as first-line therapy) leads to better long-term preservation of the visual field.",
            "tamil": "ஆரம்பத்திலேயே லேசர் சிகிச்சை (SLT) செய்வது நீண்ட காலத்திற்குப் பார்வைப்புலம் சிதைவடையாமல் பாதுகாக்க உதவுகிறது என மருத்துவ ஆய்வுகள் காட்டுகின்றன.",
            "tanglish": "Aarambathileye SLT Laser therapy panna eye vision field-ai romba varushathuku safely maintain panna mudiyum-nu research solradhu."
        },
        "faqs": [
            {"q": "Can glaucoma be cured?", "a": "No, glaucoma cannot be cured, and lost vision cannot be restored. However, treatment can effectively halt or slow further progression."},
            {"q": "What is normal eye pressure?", "a": "Normal intraocular pressure ranges between 10 and 21 mmHg. Pressure consistently above 21 mmHg is a key risk factor."}
        ]
    },
    "AMD": {
        "name": "Age-Related Macular Degeneration (AMD)",
        "tamil_name": "வயது முதிர்வு சார்ந்த மேக்குலா சிதைவு (AMD)",
        "keywords": ["amd", "macular degeneration", "dry amd", "wet amd", "cnv", "drusen", "முதிர்வு", "சிதைவு", "மேக்குலா சிதைவு"],
        "causes": {
            "english": "AMD is caused by the aging and breakdown of the macula, the central part of the retina. Dry AMD involves accumulation of yellow fatty deposits called Drusen, while Wet AMD (Choroidal Neovascularization - CNV) involves the growth of abnormal, leaking blood vessels underneath the retina.",
            "tamil": "வயது முதிர்வதால் விழித்திரையின் மையப் பகுதியான மேக்குலா சிதைவடைகிறது. உலர் AMD-யில் கொழுப்புப் படிவுகளும் (Drusen), ஈரமான AMD-யில் கசியும் அசாதாரண புதிய இரத்தக்குழாய்களும் (CNV) தோன்றும்.",
            "tanglish": "Retina-vin center portion (macula) age-aagurathaal wear and tear aagum. Dry AMD-la drusen yellow spots form aagum, Wet AMD-la abnormal-ah blood vessels form aagi blood leak aagum."
        },
        "symptoms": {
            "english": "Gradual or sudden loss of central vision, distortion of straight lines (appearing wavy), blind spots in the center of vision, and difficulty recognizing faces.",
            "tamil": "மத்திய பார்வை படிப்படியாகவோ அல்லது திடீரெனவோ குறைதல், நேர்க்கோடுகள் வளைந்து தெரிவது, முகங்களை அடையாளம் காண்பதில் சிரமம்.",
            "tanglish": "Nadu paarvai (central vision) kurayum, straight lines wavy-ah theriyum, mooka saadaigal facial structures clear-ah theriyadhu."
        },
        "risk_factors": {
            "english": "Age over 50, smoking (doubles the risk), genetics/family history, cardiovascular disease, obesity, and prolonged UV light exposure.",
            "tamil": "50 வயதுக்கு மேல் இருத்தல், புகைபிடித்தல் (அபாயத்தை இருமடங்காக்குகிறது), குடும்ப வரலாறு, இதய நோய் மற்றும் சூரிய ஒளியின் பாதிப்பு.",
            "tanglish": "50 vayadhuku mel irupathu, smoking habit (risk double pannum), genetics, cardiovascular problems, high weight, and UV rays exposure."
        },
        "diagnosis": {
            "english": "Diagnosed using visual acuity tests, Amsler grid tests, dilated fundus exam, OCT scanning (to check for fluid/drusen), and fluorescein angiography for wet AMD.",
            "tamil": "ஆம்ஸ்லர் கிரிட் பரிசோதனை, கண் விரிவாக்கப் பரிசோதனை, OCT ஸ்கேன் மற்றும் கசிவை அறிய FA சோதனைகள் செய்யப்படுகின்றன.",
            "tanglish": "Amsler Grid charts, Dilated Eye checkup, OCT scans (fluid checking), matrum fundus camera check-up panni and double-check diagnostics pannuvaanga."
        },
        "prevention": {
            "english": "Stop smoking, maintain a healthy diet rich in leafy greens and omega-3, protect eyes from UV rays, and use an Amsler grid weekly for self-monitoring.",
            "tamil": "புகைபிடித்தலை நிறுத்துங்கள், ஆன்டிஆக்ஸிடன்ட்கள் மற்றும் கீரைகளை அதிகம் உண்ணுங்கள், ஆம்ஸ்லர் கிரிட் அட்டை மூலம் வாரமொருமுறை சுயபரிசோதனை செய்யுங்கள்.",
            "tanglish": "Smoking-ai stop pannanum. Green vegetables, fruits, fish regular unavil eduthukka vendum. Amsler Grid vechu weekly check pannanum."
        },
        "treatment": {
            "english": "Dry AMD is managed with specialized AREDS2 vitamin supplements to slow progression. Wet AMD requires regular intravitreal Anti-VEGF injections to stop leakage and shrink abnormal vessels.",
            "tamil": "உலர் AMD-க்கு AREDS2 வைட்டமின் சப்ளிமெண்ட்டுகள் வழங்கப்படுகின்றன. ஈரமான AMD-க்கு அசாதாரண இரத்தக்குழாய்களைச் சுருங்கச் செய்ய Anti-VEGF இன்ஜெக்ஷன்கள் வழங்கப்படுகின்றன.",
            "tanglish": "Dry AMD condition-ku AREDS2 capsules recommendation mathiri tharuvaanga. Wet AMD condition-ku regular intravitreal injection standard treatment."
        },
        "research_insights": {
            "english": "Large-scale clinical trials (AREDS and AREDS2) established that high-dose vitamins C, E, zinc, lutein, and zeaxanthin reduce the risk of progression to advanced AMD by 25% in high-risk patients.",
            "tamil": "AREDS2 ஆய்வுகளின்படி வைட்டமின்கள் C, E, துத்தநாகம், லுடீன் மற்றும் ஜியாக்சாண்டின் அடங்கிய மாத்திரைகள் நோய் தீவிரமடைவதை 25% வரை குறைக்கின்றன என்று கண்டறியப்பட்டுள்ளது.",
            "tanglish": "Research-la AREDS2 formulation vitamins intake dry AMD progression-ai 25% kurakuthunu verify panni irukannga."
        },
        "faqs": [
            {"q": "What is the difference between Dry and Wet AMD?", "a": "Dry AMD is slow and characterized by drusen deposits, while Wet AMD is rapid, dangerous, and caused by leaky abnormal blood vessels under the retina."},
            {"q": "Is AMD inherited?", "a": "Yes, genetics play a significant role. If you have a family member with AMD, your risk of developing it is substantially higher."}
        ]
    },
    "RVO": {
        "name": "Retinal Vein Occlusion (RVO)",
        "tamil_name": "விழித்திரை நரம்பு அடைப்பு (RVO)",
        "keywords": ["rvo", "retinal vein occlusion", "vein block", "adaipu", "அடைப்பு", "நரம்பு அடைப்பு", "கண் பக்கவாதம்"],
        "causes": {
            "english": "RVO is caused by a blockage or clot in the veins that carry blood away from the retina. This causes backup pressure, leading to hemorrhages and macular edema. It is often referred to as an 'eye stroke'.",
            "tamil": "விழித்திரையிலிருந்து இரத்தத்தை வெளியேற்றும் நரம்புகளில் அடைப்பு அல்லது இரத்தக்கட்டி ஏற்படுவதால் RVO உண்டாகிறது. இது 'கண் பக்கவாதம்' என்றும் அழைக்கப்படும்.",
            "tanglish": "Retina-la irundhu blood veliya kondu pora vein blood vessel-la blockage or clot yerpadum. Itha eye stroke-nu solluvanga."
        },
        "symptoms": {
            "english": "Sudden, painless blurring or loss of vision in one eye. Vision may fluctuate and worsen over hours or days.",
            "tamil": "வலியில்லாத திடீர் பார்வை மங்கல் அல்லது ஒரு கண்ணில் பார்வை இழப்பு. சில மணிநேரங்களில் அல்லது நாட்களில் இது தீவிரமடையும்.",
            "tanglish": "Vali illamal sudden-ah oru eye-la blur aagum or complete visual loss yerpadalaam. Warning signs illamal sudden-ah varalaam."
        },
        "risk_factors": {
            "english": "High blood pressure, high cholesterol, glaucoma, diabetes, smoking, and age over 60.",
            "tamil": "உயர் இரத்த அழுத்தம், அதிக கொழுப்பு, கண் அழுத்த நோய், நீரிழிவு, புகைபிடித்தல் மற்றும் 60 வயதுக்கு மேல் இருத்தல்.",
            "tanglish": "High BP, high fat (cholesterol), glaucoma, diabetes, smoking matrum age over 60."
        },
        "diagnosis": {
            "english": "Dilated fundus examination reveals extensive retinal hemorrhages (often described as 'blood and thunder' appearance). OCT shows macular edema, and FA identifies non-perfused capillary zones.",
            "tamil": "கண் விரிவாக்கப் பரிசோதனையில் அதிக இரத்தக்கசிவு தெரியும். OCT மூலம் நீர்வீக்கத்தைக் கண்டறிந்து FA மூலம் அடைப்பு நிலை மதிப்பிடப்படுகிறது.",
            "tanglish": "Eye dilution check-up panna rathakasiyu clear-ah theriyum ('blood and thunder'). OCT scan vechu edema measure pannuvaanga matrum non-perfusion verify panna FA test pannanum."
        },
        "prevention": {
            "english": "Control cardiovascular risks: maintain optimal blood pressure, reduce cholesterol, manage diabetes, and quit smoking.",
            "tamil": "இரத்த அழுத்தம், சர்க்கரை மற்றும் கொழுப்பைக் கட்டுக்குள் வைத்திருங்கள். புகைப்பிடிப்பதை நிறுத்துங்கள்.",
            "tanglish": "Cardiovascular risks control pannanum: proper BP management, lipid levels reduction, sugar levels checks, and smoking avoidance."
        },
        "treatment": {
            "english": "First-line therapy is intravitreal Anti-VEGF injections or corticosteroid implants to treat macular edema. Sector laser photocoagulation is used if abnormal new vessels grow.",
            "tamil": "நீர்வீக்கத்தைக் குறைக்க Anti-VEGF இன்ஜெக்ஷன்கள் அல்லது ஸ்டீராய்டு உள்வைப்புகள். புதிய நரம்புகள் வளர்ந்தால் லேசர் சிகிச்சை அளிக்கப்படுகிறது.",
            "tanglish": "Macular edema-vukku anti-VEGF injections poduvaanga. Pudha abnormal blood vessels grow aana segment laser photocoagulation pannalaam."
        },
        "research_insights": {
            "english": "Clinical trials (such as the BRAVO and CRUISE studies) proved that anti-VEGF therapy provides rapid, substantial improvements in vision and reduces macular swelling in RVO patients.",
            "tamil": "BRAVO மற்றும் CRUISE ஆய்வுகள், Anti-VEGF சிகிச்சைகள் RVO நோயாளிகளின் பார்வையை விரைவாக மேம்படுத்தி வீக்கத்தைக் குறைக்கின்றன என்று நிரூபித்துள்ளன.",
            "tanglish": "Clinical studies (BRAVO, CRUISE) confirm panniyiruku anti-VEGF treatment RVO-la visual swelling and blurring-ai rapid-ah control panna best-nu."
        },
        "faqs": [
            {"q": "Can vision recover after RVO?", "a": "Yes, many patients recover significant vision, though it depends on the extent of the blockage, macular swelling control, and how quickly treatment is initiated."},
            {"q": "Is RVO related to a stroke in the brain?", "a": "RVO shares similar vascular risk factors (high BP, atherosclerosis) as a brain stroke, but a blockage in the eye does not directly cause a brain stroke."}
        ]
    },
    "CATARACT_RETRAD": {
        "name": "Cataract-Related Retinal Complications",
        "tamil_name": "கண்புரை விழித்திரை சிக்கல்கள்",
        "keywords": ["cataract retina", "irvine-gass", "cme post surgery", "புரை விழித்திரை", "கண்புரை", "அறுவை சிகிச்சை"],
        "causes": {
            "english": "Surgical trauma during cataract extraction can trigger inflammatory pathways, leading to cystoid macular edema (CME), also known as Irvine-Gass syndrome. Severe cataracts can also block clean visualization of the retina, masking underlying diseases.",
            "tamil": "கண்புரை அறுவை சிகிச்சையின் போது ஏற்படும் அழற்சியால் மேக்குலாவில் நீர்வீக்கம் (Irvine-Gass syndrome) உருவாகலாம். மேலும் தடிமனான புரை விழித்திரையை மறைக்கக் கூடும்.",
            "tanglish": "Cataract surgery trauma-val eye-kulla inflammation yerpattu, central retina portion (macula)-la fluid swelling form aagalaam (Irvine-Gass syndrome)."
        },
        "symptoms": {
            "english": "Blurred or decreased central vision weeks after cataract surgery, wavy lines, and mild redness or sensitivity to light.",
            "tamil": "அறுவை சிகிச்சைக்கு சில வாரங்களுக்குப் பிறகு மத்திய பார்வை மங்கலாகுதல், நேர்க்கோடுகள் வளைந்திருப்பது போல் தோன்றுதல்.",
            "tanglish": "Cataract surgery mudinju chinna gap-ku aprom nadu paarvai blur aagum, straight lines wavy-ah theriyum."
        },
        "risk_factors": {
            "english": "History of uveitis, diabetic retinopathy, retinal vein occlusion, complex cataract surgery, and history of CME in the other eye.",
            "tamil": "விழித்திரை நரம்பு அடைப்பு, நீரிழிவு விழித்திரை நோய், கண்ணில் ஏற்படும் ஒவ்வாமை (Uveitis) மற்றும் சிக்கலான புரை அறுவை சிகிச்சை.",
            "tanglish": "Diabetic Retinopathy history, past uveitis, complex cataract surgery duration, matrum other eye-la CME history irupathu."
        },
        "diagnosis": {
            "english": "Diagnosed via visual acuity testing, dilated fundus exam, and OCT scan which displays classic cystic spaces (macular thickening) in retinal layers.",
            "tamil": "OCT ஸ்கேன் மூலம் விழித்திரை அடுக்குகளில் உள்ள நீர் வட்ட வடிவக் குமிழ்கள் (cystic spaces) மூலம் கண்டறியப்படுகிறது.",
            "tanglish": "OCT scan-la retinal layers-kulla bubble/cystic spaces theriyum, idhai check panni diagnose pannuvaanga."
        },
        "prevention": {
            "english": "Prophylactic use of NSAID and corticosteroid eye drops before and after cataract surgery, especially for high-risk patients like diabetics.",
            "tamil": "அறுவை சிகிச்சைக்கு முன்னும் பின்னும் தகுந்த அழற்சி எதிர்ப்பு சொட்டு மருந்துகளை (NSAID drops) மருத்துவர் பரிந்துரைப்படி பயன்படுத்துதல்.",
            "tanglish": "Surgery-ku munaadiyum pinnaadiyum doc thara anti-inflammatory eye drops (NSAID) proper-ah use pannanum, especially for sugar patients."
        },
        "treatment": {
            "english": "Treated with topical NSAID and steroid eye drops. For persistent cases, intravitreal steroid or anti-VEGF injections may be required.",
            "tamil": "அழற்சி எதிர்ப்பு சொட்டு மருந்துகள் வழங்கப்படுகின்றன. நீண்ட கால பாதிப்புகளுக்கு கண்ணுக்குள் போடும் இன்ஜெக்ஷன்கள் தேவைப்படலாம்.",
            "tanglish": "Prophylactic eye drops (INSAIDs & steroids) eduthukalam. Swelling romba naal control aagalana eye steroid injection poduvaanga."
        },
        "research_insights": {
            "english": "Studies demonstrate that combining topical NSAIDs with corticosteroids yields significantly lower rates of pseudophakic CME compared to using steroids alone.",
            "tamil": "ஸ்டீராய்டு மருந்துகளுடன் NSAID சொட்டு மருந்துகளைச் சேர்த்துப் பயன்படுத்துவது கண்புரை அறுவை சிகிச்சைக்குப் பின் ஏற்படும் நீர்வீக்கத்தை பெருமளவு குறைக்கிறது என்று ஆய்வுகள் காட்டுகின்றன.",
            "tanglish": "Steroid drops kooda NSAID drops serthu poduvathu pseudophakic CME risk-ai massive-ah kuraikuthunu clinical studies indicate pannuthu."
        },
        "faqs": [
            {"q": "What is Irvine-Gass syndrome?", "a": "It is a complication of cataract surgery characterized by fluid accumulation in the macula (cystoid macular edema), typically peaking 4-6 weeks post-surgery."},
            {"q": "Does Irvine-Gass syndrome resolve?", "a": "Yes, the majority of cases resolve spontaneously or with targeted anti-inflammatory eye drops within a few months."}
        ]
    }
}

# Glossary of medical terminology
MEDICAL_GLOSSARY = {
    "macula": {"simple": "the center part of the retina responsible for sharp, detailed reading vision", "tamil": "விழித்திரையின் மையப் பகுதி (பார்வையின் கூர்மைக்கு உகந்தது)", "tanglish": "retina-vin nadu paguthi, ezhuthukalai details-ah paarka udhavuvadhu"},
    "retina": {"simple": "the light-sensitive layer of tissue lining the back of the eye", "tamil": "ஒளி உணர் விழித்திரை அடுக்கு (கண்ணின் பின்புறச் சுவர்)", "tanglish": "kannin pinpuram ulla camera film madhiri ஒளி உணரும் பகுதி"},
    "edema": {"simple": "swelling caused by fluid accumulation", "tamil": "திரவம் கசிந்து சேர்வதால் ஏற்படும் நீர்வீக்கம்", "tanglish": "fluid (neer) seruvathaal yerpadum veekkam"},
    "neovascularization": {"simple": "growth of abnormal, fragile new blood vessels", "tamil": "அசாதாரண பலவீனமான புதிய இரத்தக்குழாய்கள் முளைப்பது", "tanglish": "fragile-ah leaky pudhiya ratha kuzhalgal valaruvadhu"},
    "drusen": {"simple": "tiny yellow fatty waste deposits beneath the retina", "tamil": "விழித்திரைக்கு அடியில் சேரும் சிறிய கொழுப்புப் படிவுகள்", "tanglish": "retina layer-ku keezhe form aagum chinna yellow fat pads"},
    "intraocular pressure": {"simple": "the fluid pressure inside the eyeball", "tamil": "கண்ணின் உட்புறத் திரவ அழுத்தம்", "tanglish": "kannoda eyeball-kulla ulla neer alutham"},
    "hemorrhage": {"simple": "bleeding or leaking of blood from blood vessels", "tamil": "இரத்தக்குழாய்களிலிருந்து ஏற்படும் இரத்தக்கசிவு", "tanglish": "blood vessel laye-la irundhu yerpadum blood leakage"},
    "pathophysiology": {"simple": "the abnormal physical processes associated with a disease", "tamil": "ஒரு நோயின் செயல்பாட்டு மற்றும் உடல் சார்ந்த அசாதாரண பாதிப்பு முறை", "tanglish": "disease-al udambula/kannula yerpadum abnormal changes"},
    "exudates": {"simple": "leaked lipid (fatty) fluid that leaves yellow spots on the retina", "tamil": "இரத்தக் கசிவின் போது விழித்திரையில் படியும் கொழுப்புப் படிவு", "tanglish": "fluid leak aagi retina mela padigira fatty material"}
}

# Custom Memory Class implementing LangChain memory structure
class ConversationBufferMemory:
    def __init__(self, memory_key: str = "history", return_messages: bool = False):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.buffer: List[Dict[str, str]] = []

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        user_msg = list(inputs.values())[0]
        bot_msg = list(outputs.values())[0]
        self.buffer.append({"user": user_msg, "bot": bot_msg})

    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        memory_str = ""
        for turn in self.buffer[-4:]: # Keep last 4 turns for context
            memory_str += f"User: {turn['user']}\nAI: {turn['bot']}\n"
        return {self.memory_key: memory_str}

    def clear(self):
        self.buffer = []

# Real RAG FAISS Vector Store
class RetinalFAISSIndex:
    def __init__(self):
        print("[FAISS] Loading SentenceTransformer('BAAI/bge-small-en-v1.5')...")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self._build_chunks()
        self._build_index()

    def _build_chunks(self):
        for key, data in RAG_KNOWLEDGE_BASE.items():
            name = data["name"]
            tamil_name = data["tamil_name"]
            
            # 1. English Chunks
            for sec in ["causes", "symptoms", "risk_factors", "diagnosis", "prevention", "treatment", "research_insights"]:
                txt = data[sec]["english"]
                chunk = f"Disease: {name}. Category: {sec}. Info: {txt}"
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    "disease": key,
                    "category": sec,
                    "language": "english",
                    "text": txt,
                    "source": f"{name} Guidelines - {sec.replace('_', ' ').title()}"
                })
                
            # 2. Tamil Chunks
            for sec in ["causes", "symptoms", "risk_factors", "diagnosis", "prevention", "treatment", "research_insights"]:
                txt = data[sec]["tamil"]
                chunk = f"விழித்திரை நோய்: {tamil_name}. பிரிவு: {sec}. விவரங்கள்: {txt}"
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    "disease": key,
                    "category": sec,
                    "language": "tamil",
                    "text": txt,
                    "source": f"{tamil_name} மருத்துவ வழிகாட்டி"
                })

            # 3. Tanglish Chunks
            for sec in ["causes", "symptoms", "risk_factors", "diagnosis", "prevention", "treatment", "research_insights"]:
                txt = data[sec]["tanglish"]
                chunk = f"Eye disease: {name}. Category: {sec} in Tanglish. Explanation: {txt}"
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    "disease": key,
                    "category": sec,
                    "language": "tanglish",
                    "text": txt,
                    "source": f"{name} Tanglish Guidance"
                })
                
            # 4. FAQs
            for faq in data.get("faqs", []):
                self.chunks.append(f"Retinal Disease: {name}. FAQ Question: {faq['q']}. Answer: {faq['a']}")
                self.chunk_metadata.append({
                    "disease": key,
                    "category": "faq",
                    "language": "english",
                    "text": f"Question: {faq['q']}\nAnswer: {faq['a']}",
                    "source": f"{name} FAQ Reference"
                })

    def _build_index(self):
        print(f"[FAISS] Encoding {len(self.chunks)} text chunks with SentenceTransformer...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print("[FAISS] Real FAISS L2 vector store initialized successfully.")

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.model.encode([query], show_progress_bar=False)
        query_vector = np.array(query_vector).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "metadata": self.chunk_metadata[idx],
                    "source": self.chunk_metadata[idx]["source"],
                    "distance": float(distances[0][i])
                })
        return results

# Prompt Template definition (LangChain Style)
RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "history", "query", "language", "level", "scan_info"],
    template="""You are an expert AI Retinal Healthcare Assistant.
Your task is to provide accurate, helpful, and empathetic retinal health information.
Ensure you align with the user's preferred language and explanation level.

Context from trusted guidelines:
{context}

Active Patient Scan Data:
{scan_info}

Conversation History:
{history}

User Query: {query}
Preferred Language: {language}
Explanation Level: {level}

Provide a detailed, patient-focused response below:"""
)

# Llama 3 Inference Simulation Class
class Llama3InferenceEngine:
    """Simulates Llama 3 inference matching clinical parameters and RAG contexts."""
    
    def _match_clinical_trials(self, scan_data: Optional[Dict], lang: str) -> str:
        registry = [
            {
                "id": "NCT05984321",
                "title": "A Phase III Study of Anti-VEGF for Diabetic Macular Edema (DME)",
                "condition": "DME",
                "min_stage": "Moderate",
                "min_affected_pct": 15.0,
                "description": "Evaluating the efficacy of next-generation anti-VEGF injection intervals for patients diagnosed with moderate to severe DME.",
                "site": "Metro Retinal Center, Boston",
                "status": "Recruiting"
            },
            {
                "id": "NCT04875932",
                "title": "Dry AMD Photobiomodulation and Drusen Reduction Trial",
                "condition": "DRUSEN",
                "min_stage": "Mild",
                "min_affected_pct": 5.0,
                "description": "Testing non-invasive light therapy (photobiomodulation) for reduction of macular drusen deposits in early to moderate dry AMD.",
                "site": "Ocular Research Institute, SF",
                "status": "Recruiting"
            },
            {
                "id": "NCT06214590",
                "title": "Gene Therapy Trials for Advanced Neovascular AMD (CNV)",
                "condition": "CNV",
                "min_stage": "Severe",
                "min_affected_pct": 30.0,
                "description": "Evaluating one-time subretinal gene therapy injection for long-term suppression of choroidal neovascularization.",
                "site": "Global Eye Hospital, NY",
                "status": "Active, not recruiting"
            },
            {
                "id": "NCT05312064",
                "title": "Intravitreal Steroid Implant vs Anti-VEGF for Retinal Vein Occlusion (RVO)",
                "condition": "RVO",
                "min_stage": "Moderate",
                "min_affected_pct": 10.0,
                "description": "Head-to-head comparison of long-acting dexamethasone implant vs regular anti-VEGF for macular edema secondary to RVO.",
                "site": "Southern Retina Clinic, Atlanta",
                "status": "Recruiting"
            }
        ]
        
        if not scan_data:
            if lang == "tamil":
                return "உறுதியான நோயறிதல் சுருக்கம் ஏதும் இந்த அமர்வில் இல்லை. தகுதியான மருத்துவ சோதனைகளை ஒப்பிட முதலில் ஒரு நோயாளியின் கண் ஸ்கேனை பதிவேற்றவும்."
            elif lang == "tanglish":
                return "Session la scanning details/report edhuvum illa. Model check panna mudiyadhu. Please analyze a scan first before matching trials."
            else:
                return "No active patient scan record found in session. Please upload/analyze an OCT image first to evaluate eligibility for active clinical trials."

        label = scan_data.get("label", "").upper()
        stage = scan_data.get("stage", scan_data.get("severity", "Normal"))
        affected_pct = scan_data.get("total_affected_pct", scan_data.get("retina_affected_pct", 0.0))
        pname = scan_data.get("patient_name", "Patient")
        
        stages_map = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
        patient_stage_val = stages_map.get(stage, 0)
        
        matched = []
        for trial in registry:
            trial_cond = trial["condition"]
            trial_min_stage = trial["min_stage"]
            trial_min_stage_val = stages_map.get(trial_min_stage, 0)
            trial_min_pct = trial["min_affected_pct"]
            
            # Match condition
            cond_match = False
            if label == trial_cond:
                cond_match = True
            elif label == "CNV" and trial_cond == "CNV":
                cond_match = True
            elif label == "DRUSEN" and trial_cond == "DRUSEN":
                cond_match = True
            elif label == "DME" and trial_cond == "DME":
                cond_match = True
                
            if cond_match:
                # Check eligibility
                if patient_stage_val >= trial_min_stage_val:
                    if affected_pct >= trial_min_pct:
                        matched.append(trial)

        # Generate response in target language
        if lang == "tamil":
            resp = f"### **மருத்துவ பரிசோதனை பொருத்தங்கள் (Clinical Trial Matching)**\n\n"
            resp += f"நோயாளி: **{pname}** | கண்டறியப்பட்ட பாதிப்பு: **{label}** ({stage})\n\n"
            if not matched:
                resp += f"நோயாளிக்கு தகுந்த மருத்துவ சோதனைகள் ஏதும் தற்போதைய பதிவேட்டில் பொருந்தவில்லை.\n\n*தகுதி வரம்பு: நோயின் தீவிரம் மற்றும் பாதிக்கப்பட்ட பகுதி குறைந்தபட்ச தேவையை விட குறைவாக இருக்கலாம்.*"
            else:
                resp += f"பின்வரும் **{len(matched)}** மருத்துவ சோதனைகள் நோயாளியின் தற்போதைய மருத்துவ அறிக்கைக்கு பொருந்துகின்றன:\n\n"
                for idx, t in enumerate(matched, 1):
                    resp += f"**{idx}. {t['title']}**\n"
                    resp += f"* **சோதனை எண்:** {t['id']}\n"
                    resp += f"* **நிலை:** {t['status']}\n"
                    resp += f"* **ஆராய்ச்சி மையம்:** {t['site']}\n"
                    resp += f"* **விளக்கம்:** {t['description']}\n\n"
                resp += f"*குறிப்பு: இந்த பொருத்தங்கள் பூர்வாங்க மதிப்பீட்டின் அடிப்படையில் வழங்கப்படுகின்றன. விரிவான விவரங்களுக்கு உங்கள் மருத்துவரிடம் ஆலோசிக்கவும்.*"
            return resp

        elif lang == "tanglish":
            resp = f"### **Clinical Trial Match Result (Tanglish)**\n\n"
            resp += f"Patient Name: **{pname}** | Diagnosis: **{label}** ({stage})\n\n"
            if not matched:
                resp += f"Current condition-ku matched trials edhuvum registry-la illa.\n\n*Eligibility criteria: Stage details and affected % metrics trial requirements-oda verify panna pothu matching range-la illa.*"
            else:
                resp += f"Total **{len(matched)}** matched clinical trials check aagiyiruku:\n\n"
                for idx, t in enumerate(matched, 1):
                    resp += f"**{idx}. {t['title']}**\n"
                    resp += f"* **Trial ID:** {t['id']}\n"
                    resp += f"* **Status:** {t['status']}\n"
                    resp += f"* **Location:** {t['site']}\n"
                    resp += f"* **Details:** {t['description']}\n\n"
                resp += f"*Disclaimer: Doctor instructions and review strictly required before clinical enrollment.*"
            return resp

        else: # english
            resp = f"### **Automated Clinical Trial Matches**\n\n"
            resp += f"Patient Name: **{pname}** | Diagnostics: **{label}** ({stage}, Affected: {affected_pct:.1f}%)\n\n"
            if not matched:
                resp += f"No active clinical trials in the local registry match this patient's current diagnostic profile.\n\n*Reason: The patient's condition severity ({stage}) or affected area ({affected_pct:.1f}%) is below the minimum threshold required for active trials.*"
            else:
                resp += f"Based on the patient's diagnostic profile, they are eligible for the following **{len(matched)}** active clinical trials:\n\n"
                for idx, t in enumerate(matched, 1):
                    resp += f"**{idx}. {t['title']}**\n"
                    resp += f"* **Registry ID:** {t['id']} (ClinicalTrials.gov)\n"
                    resp += f"* **Status:** {t['status']}\n"
                    resp += f"* **Research Facility:** {t['site']}\n"
                    resp += f"* **Trial Description:** {t['description']}\n\n"
                resp += f"*Disclaimer: Patient matching is based on local mock registry metrics. Clinicians must verify biological eligibility before enrollment.*"
            return resp

    def generate_response(self, prompt: str, user_query: str, retrieved_docs: List[Dict[str, Any]], scan_data: Optional[Dict], session: Dict) -> str:
        print("[Llama 3] Model prompted with context size 4096 tokens...")
        print("[Llama 3] Running inference (temperature=0.2 for medical precision)...")
        
        cleaned = user_query.lower()
        lang = session.get("language", "english")
        
        # Check for attached file in session first
        attached_file = session.get("attached_file")
        if attached_file:
            return self._generate_attached_file_response(attached_file, user_query, lang)

        # If not a medical query and no attachment, trigger general response simulation
        if not is_medical_query(user_query):
            return self._generate_general_response(user_query, lang)
            
        # Clinical Trial Matching check
        if any(w in cleaned for w in ["clinical trial", "trial match", "eligible trial", "match trial", "trials"]):
            return self._match_clinical_trials(scan_data, lang)
        
        # Determine explanation level
        level = "beginner" # default
        if any(w in cleaned for w in ["expert", "professional", "technical", "scientific", "researcher", "clinician", "physician", "pathophysiology"]):
            level = "expert"
        elif any(w in cleaned for w in ["intermediate", "detailed", "moderate", "clinical summary", "doctor explanation"]):
            level = "intermediate"
            
        # 1. Advanced Report Interpretation Module (Very Important)
        pasted_report = self._parse_pasted_report(user_query)
        if pasted_report:
            return self._interpret_report(pasted_report, lang, level)
            
        # 2. Prediction Result / Scan Data Explanation
        if scan_data and any(w in cleaned for w in ["prediction", "result", "my report", "my scan", "diagnose", "explain my", "diagnosis", "why was this", "confidence"]):
            return self._explain_prediction(scan_data, lang, level)
            
        # 3. Symptom-Based Query Support
        symptom_match = self._check_symptoms(cleaned)
        if symptom_match:
            return self._generate_symptom_response(symptom_match, lang)
            
        # 4. Research Insights
        if any(w in cleaned for w in ["research", "study", "clinical trial", "paper", "insights", "findings"]):
            return self._generate_research_response(cleaned, lang)

        # 5. Structured Report Generation Request
        if any(w in cleaned for w in ["generate report", "create summary", "report summary", "print report"]):
            return self._generate_chat_report(scan_data, lang)

        # 6. FAQs
        if any(w in cleaned for w in ["faq", "frequently", "question", "answers"]):
            faq_resp = self._get_faq_response(cleaned, lang)
            if faq_resp:
                return faq_resp

        # 7. Check if we retrieved specific docs via FAISS
        if retrieved_docs:
            # Group by matched disease
            best_match = retrieved_docs[0]["metadata"]
            d_key = best_match["disease_key"] if "disease_key" in best_match else best_match["disease"]
            doc = RAG_KNOWLEDGE_BASE[d_key]
            return self._explain_disease(doc, d_key, lang, level)

        # Fallback to general assistance message
        return self._fallback_response(lang)

    def _generate_attached_file_response(self, attached_file: Dict[str, Any], query: str, lang: str) -> str:
        filename = attached_file.get("filename", "attached_file")
        
        # Scenario A: It is a parsed medical image analysis from our ML model
        if attached_file.get("is_medical_scan"):
            label_full = attached_file.get("label_full", "Unknown")
            conf = attached_file.get("confidence", 0.0)
            stage = attached_file.get("stage", "Unknown")
            total_affected = attached_file.get("total_affected_pct", 0.0)
            fluid = attached_file.get("fluid_area_pct", 0.0)
            lesion = attached_file.get("lesion_area_pct", 0.0)
            
            if lang == "tamil":
                summary = f"""அறிக்கையின்படி, உங்களுடைய விழித்திரை ஸ்கேன் படம் ({filename}) பகுப்பாய்வு செய்யப்பட்டது:
- **கண்டறியப்பட்ட பாதிப்பு:** {label_full}
- **AI உறுதிப்பாடு:** {conf:.1f}%
- **பாதிப்பின் தீவிரம்:** {stage}
- **விழித்திரை பாதிப்பு அளவு:** {total_affected:.2f}%
- **நீர் கோர்த்த பகுதி (Fluid):** {fluid:.2f}%
- **சிதைவு பகுதி (Lesion):** {lesion:.2f}%

*விளக்கம்:* இந்த ஸ்கேன் படத்தின் அடிப்படையில், உங்களுக்கு {label_full} அறிகுறிகள் தெரிகின்றன. உடனடியாக கண் மருத்துவரை அணுகி விரிவான விழித்திரை சோதனை செய்து கொள்ள பரிந்துரைக்கிறோம்."""
            elif lang == "tanglish":
                summary = f"""Ungaloda retinal scan image ({filename}) analysis results:
- **Condition:** {label_full}
- **AI Confidence:** {conf:.1f}%
- **Severity Stage:** {stage}
- **Retina Affected area:** {total_affected:.2f}%
- **Fluid (edema) area:** {fluid:.2f}%
- **Lesion area:** {lesion:.2f}%

*Explanation:* Scan check-up panni paathaadhula {label_full} signs theriyudhu. Immediate-ah eye specialist (Ophthalmologist) kitta report screen panni consultation panna recommended."""
            else:
                summary = f"""Based on the attached retinal scan ({filename}):
- **Condition Detected:** {label_full}
- **AI Confidence Score:** {conf:.1f}%
- **Severity Classification:** {stage}
- **Total Affected Retina Area:** {total_affected:.2f}%
- **Fluid Area (Edema):** {fluid:.2f}%
- **Lesion Area:** {lesion:.2f}%

*Clinical Interpretation:* The scan exhibits features highly consistent with {label_full}. We strongly recommend scheduling a comprehensive dilated eye exam with a retinal specialist for formal clinical correlation."""
            return summary

        # Scenario B: It is a general image file that was not classified as a valid retinal scan
        elif attached_file.get("type", "").startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            if lang == "tamil":
                return f"உங்களது **{filename}** என்ற படம் இணைக்கப்பட்டுள்ளது. ஆனால் இது ஒரு செல்லுபடியாகும் விழித்திரை (OCT/Fundus) ஸ்கேன் படமாகத் தெரியவில்லை. தயவுசெய்து சரியான விழித்திரை படத்தை பதிவேற்றவும்."
            elif lang == "tanglish":
                return f"Ungaloda image **{filename}** attach aagiyiruku, aana idhu valid-ana retinal (OCT/Fundus) scan images madhiri illa. Correct scan image-ai upload pannavum."
            else:
                return f"Your image **{filename}** has been attached. However, it does not appear to be a valid retinal (OCT or Fundus) scan image. Please ensure you upload a valid clinical scan for automated analysis."

        # Scenario C: It is a document file (PDF, DOCX, TXT, MD, etc.)
        file_text = attached_file.get("text", "").strip()
        if not file_text:
            if lang == "tamil":
                return f"இணைக்கப்பட்ட கோப்பு **{filename}** காலியாக உள்ளது அல்லது அதிலிருந்து உரையைப் பிரித்தெடுக்க முடியவில்லை."
            elif lang == "tanglish":
                return f"Attached file **{filename}** empty-ah iruku, or text extract panna mudiyala."
            else:
                return f"The attached file **{filename}** appears to be empty, or text content could not be extracted from it."
                
        cleaned_query = query.lower()
        
        # Heuristics for file Q&A
        if any(w in cleaned_query for w in ["summarize", "summary", "brief", "what is this", "explain this", "சுருக்கம்", "vilakku", "explain panna"]):
            sentences = [s.strip() for s in re.split(r'\.|\n', file_text) if len(s.strip()) > 10]
            summary_lines = sentences[:6]
            summary_text = "\n- ".join(summary_lines)
            
            if lang == "tamil":
                return f"இணைக்கப்பட்ட **{filename}** கோப்பின் சுருக்கம் இதோ:\n\n- {summary_text}"
            elif lang == "tanglish":
                return f"Attached file **{filename}**-oda summary details idho:\n\n- {summary_text}"
            else:
                return f"Here is a summary of the attached document **{filename}**:\n\n- {summary_text}"
                
        # Word search matching
        sentences = [s.strip() for s in re.split(r'\.|\n', file_text) if len(s.strip()) > 5]
        query_words = [w for w in re.findall(r'\b\w+\b', cleaned_query) if len(w) > 2 and w not in ["what", "how", "why", "who", "when", "where", "the", "and", "for", "with", "this", "that", "from", "sollunga", "enna", "epdi"]]
        
        matched_sentences = []
        for s in sentences:
            s_lower = s.lower()
            if any(qw in s_lower for qw in query_words):
                if s not in matched_sentences:
                    matched_sentences.append(s)
                    
        if not matched_sentences:
            matched_sentences = sentences[:5]
            
        result_text = "\n- ".join(matched_sentences[:6])
        if lang == "tamil":
            return f"இணைக்கப்பட்ட **{filename}** கோப்பின் அடிப்படையில் பெறப்பட்ட பொருத்தமான விவரங்கள்:\n\n- {result_text}"
        elif lang == "tanglish":
            return f"Attached file **{filename}**-la irundhu ungaloda kelviku matched answers:\n\n- {result_text}"
        else:
            return f"Based on the attached file **{filename}**, here are the relevant findings:\n\n- {result_text}"

    def _generate_general_response(self, query: str, lang: str) -> str:
        cleaned = query.lower().strip()
        
        # 1. Greetings
        if any(w in cleaned for w in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "வணக்கம்", "vanakkam"]):
            if lang == "tamil":
                return "வணக்கம்! நான் உங்கள் விழித்திரை மருத்துவ AI உதவியாளர். உங்களுக்கு இன்று நான் எவ்வாறு உதவ முடியும்? கண் நோய்கள், அறிகுறிகள், சிகிச்சைகள் அல்லது பொதுவான தலைப்புகள் குறித்து என்னிடம் தாராளமாகக் கேட்கலாம்."
            elif lang == "tanglish":
                return "Hello! Naan ungaloda AI clinical assistant. Iniku ungaluku enna details venum? Retinal scans pathiyo or general topics pathiyo edhu venalum kelunga, clear-ah explain panren!"
            else:
                return "Hello! I am your AI clinical assistant. How can I assist you today? You can ask me questions about retinal health, analyze attached medical reports/images, or ask general questions."

        # 2. Love query
        if any(w in cleaned for w in ["love", "what is love", "love na enna", "அன்பு என்றால்"]):
            if lang == "tamil":
                return """### **அன்பு (Love) என்றால் என்ன?**

அன்பு என்பது மனித உணர்வுகளில் மிகவும் ஆழமான மற்றும் சக்திவாய்ந்த ஒன்றாகும். இது பாசம், அரவணைப்பு, பாதுகாப்பு, மற்றும் பரஸ்பர மதிப்பை அடிப்படையாகக் கொண்டது.

**அறிவியல் விளக்கம்:**
1. **நரம்பியல் இரசாயனம்:** மூளையில் ஆக்சிடோசின் (Oxytocin), டோபமைன் (Dopamine), மற்றும் செரோடோனின் (Serotonin) போன்ற மகிழ்ச்சியைத் தரும் ஹார்மோன்களின் வெளியீட்டோடு இது தொடர்புடையது.
2. **உளவியல் ஆரோக்கியம்:** அன்பு மற்றும் ஆதரவைப் பெறுவது மனித மன அழுத்தத்தைக் குறைத்து, ஒட்டுமொத்த உடல் ஆரோக்கியத்தையும் மேம்படுத்துகிறது.

*மருத்துவ AI ஆலோசனையாக இருந்தாலும், மனித உறவுகளின் அன்பு மற்றும் பாசம் வாழ்வின் மிக முக்கிய அம்சமாகும்!*"""
            elif lang == "tanglish":
                return """### **What is Love? (Tanglish)**

Love-nu சொல்றது humans-oda romba primary and deep feeling/emotion. Adhu affection, care, respect, matrum emotional bonding-ai base panniyiruku.

**Scientific Details:**
1. **Brain chemicals:** Namma brain-la oxytocin, dopamine, and serotonin hormones release aaguradhaala dhaan indha feeling and bonding create aagudhu.
2. **Mental health:** Direct attachment, love, and care namma stress levels-ai romba reduce panni, healthy body parameters maintenance-ku support pannum.

*Healthcare point-of-view-la, pure mental peace matrum caring dhaan eye recovery details check panna kooda helpful-ah irukum!*"""
            else:
                return """### **Understanding Love**

Love is a profound and complex set of emotions, behaviors, and beliefs associated with strong feelings of affection, protectiveness, warmth, and respect for another person, object, or concept.

### **Scientific & Physiological Aspects:**
1. **Neurochemical Release:** When experiencing love, the brain releases neurotransmitters such as **oxytocin** (the bonding hormone), **dopamine** (the reward transmitter), and **serotonin** (the mood stabilizer).
2. **Psychosomatic Health:** Strong social attachments and healthy relationships are shown to lower cortisol (stress hormone) levels, reduce cardiovascular risks, and strengthen overall immunity.

*As a medical AI assistant, I evaluate scan data, but emotional support and empathy are critical components of patient care.*"""

        # 3. Who are you
        if any(w in cleaned for w in ["who are you", "your name", "what are you", "நீ யார்", "nee yaar"]):
            if lang == "tamil":
                return "நான் **விழித்திரை மருத்துவ AI உதவியாளர் (VisionCare Clinical Assistant)**. நான் விழித்திரை ஸ்கேன்களை பகுப்பாய்வு செய்து, கண் நோய் கண்டறிதல் மற்றும் சிகிச்சைகள் குறித்த தகவல்களை வழங்க வடிவமைக்கப்பட்டுள்ளேன்."
            elif lang == "tanglish":
                return "Naan ungaloda **VisionCare AI Assistant**. Retinal scans-ai check panni, eye problems details find panni, clarify panna design pannapata healthcare assistant."
            else:
                return "I am the **VisionCare AI Clinical Assistant**, a RAG-powered deep learning medical companion. My primary goal is to help you analyze retinal OCT/Fundus scans, interpret clinical findings, and answer eye health queries."

        # 4. How are you
        if any(w in cleaned for w in ["how are you", "how is it going", "how are you doing", "நலம்", "nalama"]):
            if lang == "tamil":
                return "நான் நன்றாக இருக்கிறேன், நன்றி! உங்கள் கண் ஆரோக்கியம் மற்றும் விழித்திரை ஸ்கேன்கள் குறித்து உங்களுக்கு உதவ நான் தயாராக உள்ளேன். நான் எவ்வாறு உங்களுக்கு உதவ முடியும்?"
            elif lang == "tanglish":
                return "Naan romba nalla iruken, thank you! Iniku ungaloda scanning reports verify panni, clarified decisions eduka support panna ready-ah iruken. Start pannalama?"
            else:
                return "I am functioning at peak optimal capacity, thank you for asking! I am ready to analyze retinal scans and clarify any ophthalmology-related questions you have. How can I help you?"

        # 5. Thank you
        if any(w in cleaned for w in ["thank you", "thanks", "நன்றி", "nanri"]):
            if lang == "tamil":
                return "மிக்க மகிழ்ச்சி! உங்களுக்கு உதவ முடிந்ததில் நான் மகிழ்ச்சி அடைகிறேன். கண் ஆரோக்கியம் அல்லது வேறு ஏதேனும் சந்தேகம் இருந்தால் எப்போது வேண்டுமானாலும் கேட்கலாம்."
            elif lang == "tanglish":
                return "Romba thanks! Ungaluku clear definitions kuduthadhula happy. Scan comparisons pathiyo or general symptoms pathiyo edhunaalum check pannalaam!"
            else:
                return "You are very welcome! I am glad to assist you. Let me know if you have any further questions about retinal scans, treatment protocols, or eye care guidelines."

        # 6. Jokes, Poems, and Stories
        if "joke" in cleaned or "நகைச்சுவை" in cleaned:
            jokes = [
                ("Why are eye doctors so smart? Because they have many pupils!",
                 "கண் மருத்துவர்கள் ஏன் இவ்வளவு புத்திசாலிகள்? ஏனென்றால் அவர்களிடம் பல 'pupils' (மாணவர்கள்/கண் மணிகள்) உள்ளனர்!",
                 "Why eye docs romba smart-ah irukaanga? Because avanga kitta nariya 'pupils' (students/eye pupils) irukaanga!"),
                ("What did the retina say to the lens? I see you!",
                 "விழித்திரை லென்ஸிடம் என்ன சொன்னது? 'நான் உன்னை பார்க்கிறேன்!'",
                 "Retina lens-kitta enna soluchu? 'I see you!' (Naan unna paakuren!)"),
                ("Why did the eye doctor go to school? To improve their vision!",
                 "கண் மருத்துவர் ஏன் மீண்டும் பள்ளிக்குச் சென்றார்? தனது பார்வையை மேம்படுத்த!",
                 "Eye doctor yen school-ku ponaaru? Avaroda vision-ai improve panna!")
            ]
            selected_joke = jokes[hash(cleaned) % len(jokes)]
            if lang == "tamil":
                return f"### **மருத்துவ நகைச்சுவை (Ophthalmology Joke):**\n\n{selected_joke[1]}"
            elif lang == "tanglish":
                return f"### **Funny Eye Joke:**\n\n{selected_joke[2]}"
            else:
                return f"### **Ophthalmology Joke:**\n\n{selected_joke[0]}"

        if "poem" in cleaned or "கவிதை" in cleaned:
            if lang == "tamil":
                return """### **விழித்திரை கவிதை (Poem about Eyes):**

மண்ணில் வாழும் மனிதக் கண்ணே,
விண்ணின் ஒளியை வாங்கும் பெண்ணே!
விழித்திரை என்னும் திரையினில் வந்து,
உலகைக் காட்டும் ஒளி விளக்கு!

நோயில்லாது உன்னைப் பேண,
அறிவியல் காட்டும் வழிகள் நூறு.
நலம் பெறவே என்றும் நீயும்,
விழிப்பாய் இருப்பாய் கண்ணே நீயும்!"""
            elif lang == "tanglish":
                return """### **Retina Poem (Tanglish):**

Kannaana kanne, window to the soul,
Keeping namma vision healthy and whole.
Light ray shadow-ah retina-la vizhum,
Brain-kitta signal send panni, images malarum.

Oru tiny cell veekkam stress-naala aagalaam,
Ophthalmology analysis pathi seekiram paakalam.
Take care of eyes daily, clear-ah irukanum,
AI tools help-oda paarvai minnanum!"""
            else:
                return """### **Ode to the Eye**

Windows of the soul, so bright and clear,
Capturing the memories we hold dear.
With rods and cones to catch the morning ray,
Mapping the colors of a brand-new day.

A perfect lens that focuses the stream,
From silent shadow to a golden dream.
Protect these spheres of light and vision true,
For they reveal the universe to you."""

        if "story" in cleaned or "கதை" in cleaned:
            if lang == "tamil":
                return """### **கண் மற்றும் பார்வையின் கதை:**
முன்னொரு காலத்தில், 'ராட்' மற்றும் 'கோன்' என்ற இரண்டு விழித்திரை நண்பர்கள் ஒரு மனிதக் கண்ணில் வாழ்ந்து வந்தனர். ராட் இருட்டிலும் பார்ப்பதற்கு உதவினார், கோன் வண்ணங்களைப் பார்ப்பதற்கு உதவினார். ஒரு நாள், திடீரென அதிக பிரகாசமான ஒளி கண்ணில் விழுந்தது. இருவரும் இணைந்து வேலை செய்து, கண்ணைப் பாதுகாத்து, வண்ணமயமான உலகைக் காட்டினர். அவர்கள் ஒற்றுமையாக இருந்தால் மட்டுமே மனிதனால் தெளிவாகப் பார்க்க முடியும் என்பதை உணர்ந்தனர். அன்று முதல் அவர்கள் விழித்திரையின் காவலர்களாக வாழ்ந்து வருகின்றனர்."""
            elif lang == "tanglish":
                return """### **Story of Rod and Cone:**
Once upon a time, 'Rod' and 'Cone' nu rendu cells retina-la friends-ah irundhaanga. Rod-ku dark/night view romba pudikum, Cone-ku bright colours-naal dhaan energy. Oru naal high sugar levels-naala retina-la small vessels leak aagi swelling yerpattadhu. Rod and Cone sandhosham-ah vela panna mudiyama kashta pattaanga. Aparam treatment, anti-VEGF scans, and glucose control pathi check panni, health restoration aachu. Rod and Cone andha human view-ai clear panna thirumba activate aanaanga. Moral: Eye health database tracking and strict control is very important!"""
            else:
                return """### **The Guardians of Sight**
Once upon a time in a microscopic valley called the Macula, lived two specialized guardians named Rod and Cone. Rod was a quiet keeper who worked in the shadows of the night, allowing the host to navigate in dim light. Cone was a vibrant artist who loved the daylight, painting the world in shades of red, green, and blue. One day, a sudden wave of glucose flooded the valley, causing fluid leakage and swelling. Rod and Cone struggled to communicate with the optic nerve. Through early clinical detection and specialized treatment, the fluid was cleared. Rod and Cone returned to their duties, restoring the host's clear vision. Moral: Regular clinical checks protect your sight."""

        # 7. Coding Questions
        if any(w in cleaned for w in ["code", "python", "javascript", "html", "css", "c++", "java", "loop", "function", "class", "niral"]):
            if "loop" in cleaned:
                code_block = "```python\n# Python For Loop Example\nfor i in range(1, 6):\n    print(f'Iteration {i}')\n```"
                explanation = "A for loop is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string)."
                explanation_ta = "ஃபார் (for) லூப் என்பது ஒரு பட்டியலை மீண்டும் மீண்டும் இயக்கப் பயன்படுத்தப்படும் ஒரு கட்டுப்பாட்டு அமைப்பாகும்."
                explanation_tg = "For loop vandhu sequence or list arrays-ai loop/iterate panni condition check panna use aagum."
            elif "function" in cleaned or "def" in cleaned:
                code_block = "```python\n# Python Function Example\ndef greet_user(name):\n    return f'Hello, {name}! Welcome to VisionCare AI.'\n\n# Calling the function\nresult = greet_user('Doctor')\nprint(result)\n```"
                explanation = "A function is a block of code which only runs when it is called. You can pass data, known as parameters, into a function."
                explanation_ta = "சார்பு (function) என்பது ஒரு குறிப்பிட்ட செயலைச் செய்ய வடிவமைக்கப்பட்ட மற்றும் மீண்டும் பயன்படுத்தக்கூடிய குறியீட்டுத் தொகுதியாகும்."
                explanation_tg = "Function-nu சொல்றது reusable block of code. Indhula parameter send panni print trigger pannalam."
            else:
                code_block = "```python\n# General Python Class Example\nclass ClinicalAssistant:\n    def __init__(self, name):\n        self.name = name\n    \n    def get_status(self):\n        return f'Assistant {self.name} is running locally.'\n\nbot = ClinicalAssistant('Llama3-Sim')\nprint(bot.get_status())\n```"
                explanation = "This is a basic object-oriented program in Python showcasing class definitions and method instantiation."
                explanation_ta = "இது பைத்தானில் வகுப்பு (class) மற்றும் பொருள்களை (objects) உருவாக்கும் அடிப்படை நிரலாகும்."
                explanation_tg = "Python-la class matrum object-oriented programming define panna logical structure idho."

            if lang == "tamil":
                return f"### **நிரல் குறியீடு விளக்கம் (Code Block):**\n\n{code_block}\n\n**விளக்கம்:**\n{explanation_ta}"
            elif lang == "tanglish":
                return f"### **Technical Coding Code Example:**\n\n{code_block}\n\n**Explanation:**\n{explanation_tg}"
            else:
                return f"### **Technical Code Implementation:**\n\n{code_block}\n\n**Detailed Explanation:**\n{explanation}"

        # 8. Math Calculator Heuristics
        import re
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', cleaned)
        if math_match:
            num1 = int(math_match.group(1))
            op = math_match.group(2)
            num2 = int(math_match.group(3))
            result = 0
            if op == '+': result = num1 + num2
            elif op == '-': result = num1 - num2
            elif op == '*': result = num1 * num2
            elif op == '/': result = num1 / num2 if num2 != 0 else "Error (Division by Zero)"
            
            if lang == "tamil":
                return f"### **கணித கணக்கீடு (Math Calculation):**\n\n- **கேள்வி:** {num1} {op} {num2}\n- **விடை:** **{result}**"
            elif lang == "tanglish":
                return f"### **Math Calculator Result:**\n\n- **Operation:** {num1} {op} {num2}\n- **Result:** **{result}**"
            else:
                return f"### **Mathematical Calculation:**\n\n- **Expression:** {num1} {op} {num2}\n- **Result:** **{result}**"

        # 9. Recipes / Cooking
        if any(w in cleaned for w in ["cook", "recipe", "cake", "make food", "samayal", "how to make"]):
            if lang == "tamil":
                return """### **சமையல் குறிப்பு (Easy Cake Recipe):**

வீட்டிலேயே எளிதாக கேக் செய்ய தேவையான பொருட்கள் மற்றும் செய்முறை:

**தேவையான பொருட்கள்:**
- மைதா மாவு - 1 கப்
- சர்க்கரை - 1/2 கப்
- வெண்ணெய் - 1/4 கப்
- முட்டை - 1
- பேக்கிங் பவுடர் - 1 ஸ்பூன்

**செய்முறை:**
1. வெண்ணெய் மற்றும் சர்க்கரையை நன்றாகக் கலக்கவும்.
2. முட்டை மற்றும் மைதா மாவைச் சேர்த்து மெதுவாகக் கலக்கவும்.
3. பேக்கிங் பவுடர் சேர்த்து மாவைக் கேக் பாத்திரத்தில் ஊற்றவும்.
4. ஓவனில் 180 டிகிரியில் 30 நிமிடங்கள் பேக் செய்து பரிமாறவும்!"""
            elif lang == "tanglish":
                return """### **Baking / Cooking Guide (Tanglish Cake Recipe):**

Veetlaye easy-ah cake prepare panna instructions idho:

**Ingredients:**
- Flour (Maida) - 1 cup
- Sugar - 1/2 cup
- Butter - 1/4 cup
- Egg - 1 or Milk - 1/2 cup
- Baking powder - 1 tsp

**Steps:**
1. Butter matrum sugar-ai bowl-la nalla blend pannunga.
2. Mix standard flour, egg/milk dynamic-ah serthu mix panna koodathu (gentle mix).
3. Baking powder serthu cake pan-la pour pannuga.
4. Pre-heated oven-la 180 degrees-la 30 mins bake panni ready soft cake enjoy pannuga!"""
            else:
                return """### **Simple Baking Recipe: Classic Vanilla Cake**

Here is a straightforward recipe to bake a delicious sponge cake at home:

**Ingredients Needed:**
- All-Purpose Flour: 1 cup (120g)
- Granulated Sugar: 1/2 cup (100g)
- Unsalted Butter (Softened): 1/4 cup (60g)
- Large Egg: 1 (or 1/2 cup whole milk for eggless)
- Baking Powder: 1 teaspoon
- Vanilla Extract: 1 teaspoon

**Baking Instructions:**
1. **Creaming:** In a mixing bowl, cream together the softened butter and sugar until light and fluffy.
2. **Wet Ingredients:** Beat in the egg and vanilla extract until fully combined.
3. **Dry Ingredients:** Sift in the flour and baking powder. Mix gently until a smooth batter forms.
4. **Baking:** Pour the batter into a greased baking pan. Bake in a preheated oven at 350°F (180°C) for 25 to 30 minutes, or until a toothpick inserted in the center comes out clean. Let it cool before serving!"""

        # 10. General Science / Technology Dictionary
        science_dict = {
            "gravity": ("Gravity is a fundamental force of nature that attracts objects with mass towards each other. According to Einstein's General Relativity, gravity is the curvature of spacetime caused by mass and energy...",
                        "ஈர்ப்பு விசை என்பது பிரபஞ்சத்தில் நிறை உள்ள எந்தவொரு இரு பொருட்களையும் ஒன்றையொன்று ஈர்க்கும் ஒரு அடிப்படை விசை ஆகும். ஐன்ஸ்டீனின் பொது சார்பியல் கோட்பாட்டின்படி, ஈர்ப்பு என்பது விண்வெளியின் வளைவு ஆகும்...",
                        "Gravity-nu சொல்றது objects-ai center/mass-kitta attract panra global force. Einstein theory pathi solvadhanaal, mass spacetime-la curves create pannuradhaala dhaan gravity function aaguthu..."),
            "dna": ("Deoxyribonucleic acid (DNA) is a molecule that carries the genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms...",
                    "டிஎன்ஏ (DNA) என்பது உயிரினங்களின் வளர்ச்சி, செயல்பாடு மற்றும் இனப்பெருக்கம் ஆகியவற்றிற்கான மரபணு வழிமுறைகளைக் கொண்ட ஒரு மூலக்கூறு ஆகும்...",
                    "DNA-nu சொல்றது genetic codes store panni vechu cell growth and replication-ku instruct panra double-helix chemical molecule..."),
            "quantum": ("Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles...",
                        "குவாண்டம் மெக்கானிக்ஸ் என்பது அணுக்கள் மற்றும் துணை அணு துகள்களின் அளவில் இயற்கையின் இயற்பியல் பண்புகளை விவரிக்கும் இயற்பியலின் ஒரு அடிப்படை கோட்பாடு ஆகும்...",
                        "Quantum physics-na subatomic level atomic particles behaviors explain panra domain. Superposition matrum entanglement logic base panni work aagum..."),
            "blockchain": ("Blockchain is a distributed database or ledger that is shared among the nodes of a computer network, widely known for keeping secure and decentralized transaction records...",
                           "பிளாக்செயின் என்பது ஒரு பரவலாக்கப்பட்ட தரவுத்தளமாகும். இது கணினி நெட்வொர்க்கில் பரிவர்த்தனைகளைப் பாதுகாப்பாகவும் வெளிப்படையாகவும் பதிவு செய்யப் பயன்படுகிறது...",
                           "Blockchain-nu சொல்றது decentralized ledger technology. Details or data entries blocks-ah cryptographic hash connections valiyaga chain elements-ah link aagiyirukum..."),
            "speed of light": ("The speed of light in a vacuum is a universal physical constant exactly equal to 299,792,458 meters per second (approximately 300,000 km/s)...",
                               "வெற்றிடத்தில் ஒளியின் வேகம் என்பது விநாடிக்கு தோராயமாக 3,00,000 கிலோமீட்டர் ஆகும்...",
                               "Speed of light c value vacuum-la check panna standard-ah 300,000 km/s. Universe-la travel panna koodiya fastest speed constant idhu dhaan..."),
            "black hole": ("A black hole is a region of spacetime where gravity is so strong that nothing, including light, can escape from it...",
                           "கருந்துளை என்பது விண்வெளியில் ஈர்ப்பு விசை மிக அதிகமாக இருக்கும் ஒரு பகுதி ஆகும், இதிலிருந்து ஒளியும் கூட தப்பிக்க முடியாது...",
                           "Black hole-nu சொல்றது space-la gravity excessive-ah ulla dense point. Event horizon line cross panna details boundary level escape details empty aagum..."),
            "evolution": ("Evolution is the change in the heritable characteristics of biological populations over successive generations, driven by natural selection...",
                          "பரிணாம வளர்ச்சி என்பது அடுத்தடுத்த தலைமுறைகளில் உயிரினங்களின் மரபணு பண்புகளில் ஏற்படும் மாற்றமாகும்...",
                          "Evolution-na generations and years valiyaga living organism survival-kaga custom changes adaptations perform panra process. Natural selection based..."),
            "cell division": ("Mitosis is a part of the cell cycle in which replicated chromosomes are separated into two new nuclei, giving rise to genetically identical cells...",
                              "மைட்டோசிஸ் என்பது ஒரு தாய் செல் இரண்டு ஒத்த சேய் செல்களாகப் பிரியும் செல் பிரிதல் செயல்முறை ஆகும்...",
                              "Mitosis/Cell division-na healthy cells replace panna growth and recovery stage-la single cell divide aagi copy cells generate panra path...")
        }
        
        for key, value in science_dict.items():
            if key in cleaned:
                if lang == "tamil":
                    return f"### **விளக்கம்: {key.upper()}**\n\n{value[1]}"
                elif lang == "tanglish":
                    return f"### **Science Topic: {key.upper()}**\n\n{value[2]}"
                else:
                    return f"### **Scientific Overview: {key.upper()}**\n\n{value[0]}"

        # 11. Dynamic general fallback compiler
        subject = query.replace("what is", "").replace("tell me about", "").replace("explain", "").replace("who is", "").replace("?", "").strip()
        if not subject:
            subject = "your query"
            
        title_subject = subject.capitalize()
        if lang == "tamil":
            return f"""### **விளக்கம்: {title_subject}**

உங்களது பொதுவான கேள்விக்கு பதிலளிப்பதில் மகிழ்ச்சி:

1. **பொதுவான விளக்கம்:** '{subject}' என்பது ஒரு முக்கியமான அறிவியல், தொழில்நுட்ப அல்லது பொது அறிவு சார்ந்த தலைப்பாகும். இது பல்வேறு துறைகளில் முக்கிய பங்கு வகிக்கிறது.
2. **முக்கிய அம்சங்கள்:**
   - **பயன்பாடு:** மனித வாழ்க்கையையும் சிந்தனையையும் எளிதாக்க இது பெரிதும் உதவுகிறது.
   - **பரிணாமம்:** நவீன கண்டுபிடிப்புகள் மற்றும் ஆராய்ச்சிகள் மூலம் இத்துறை வேகமாக வளர்ந்து வருகிறது.
3. **முடிவுரை:** இத்தலைப்பு குறித்து மேலும் ஆழமான விவாதங்கள் மற்றும் பகுப்பாய்வுகளை மேற்கொள்ள நீங்கள் தாராளமாக மேலும் கேள்விகளைக் கேட்கலாம்.
"""
        elif lang == "tanglish":
            return f"""### **Details on: {title_subject}**

Ungaloda question pathi direct general explanation idho:

1. **Overview:** '{subject}'-nu solradhu oru key scientific, technical, or general knowledge topic. Edhu daily-life usages or research-la romba importad-ah iruku.
2. **Key Highlights:**
   - **Application:** Indha topic human work and learning process-ai simplify panna support pannum.
   - **Growth:** Modern technologies and advancements valiyaga indha domain/field continuous-ah upgrade aagitu varudhu.
3. **Summary:** Indha topic pathi specific terms or code structure edhavadhu sandhegam irundha, nalla explain panren, so kelunga!
"""
        else:
            return f"""### **Information about: {title_subject}**

Here is a structured, AI-generated overview based on your query:

1. **Introduction & Core Concept:**
   - '{subject}' represents a significant subject area in science, technology, or humanities.
   - Understanding its fundamental principles is key to analyzing its broader implications and applications.

2. **Key Dimensions:**
   - **Theoretical Foundation:** It relies on established principles, empirical research, and logical frameworks.
   - **Practical Utility:** It is widely utilized across industries and fields to solve problems, automate workflows, or expand human knowledge.

3. **Strategic Context:**
   - In modern systems, this topic interacts with other emerging concepts (like computational networks, automation, and analytics).
   - Continuous research in this area continues to drive innovation and discovery.

*As your AI Assistant, I can answer both general queries and detailed clinical questions about eye scans.*
"""

    def _parse_pasted_report(self, query: str) -> Optional[Dict]:
        """Parses a pasted report or classification string and extracts parameters."""
        normalized = query.lower()
        if not any(w in normalized for w in ["predicted", "disease", "confidence", "severity", "report", "stage", "affected", "fluid", "lesion"]):
            return None
            
        # Extract disease
        disease = "UNKNOWN"
        for d in ["dme", "cnv", "drusen", "normal", "amd", "glaucoma", "rvo", "retinopathy", "cataract"]:
            if d in normalized:
                disease = d.upper()
                break
                
        # Extract confidence
        confidence = 90.0  # default
        conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%', normalized)
        if conf_match:
            confidence = float(conf_match.group(1))
        else:
            conf_num_match = re.search(r'confidence\s*:?\s*(\d+(?:\.\d+)?)', normalized)
            if conf_num_match:
                confidence = float(conf_num_match.group(1))
                
        # Extract severity
        severity = "Unknown"
        for s in ["mild", "moderate", "severe", "critical", "normal"]:
            if s in normalized:
                severity = s.capitalize()
                break
                
        # Extract fluid and lesion percentages if any
        fluid_pct = 0.0
        lesion_pct = 0.0
        fluid_match = re.search(r'fluid\s*:?\s*(\d+(?:\.\d+)?)\s*%', normalized)
        if fluid_match:
            fluid_pct = float(fluid_match.group(1))
        lesion_match = re.search(r'lesion\s*:?\s*(\d+(?:\.\d+)?)\s*%', normalized)
        if lesion_match:
            lesion_pct = float(lesion_match.group(1))

        return {
            "label": disease,
            "confidence": confidence,
            "stage": severity,
            "patient_name": "Patient",
            "fluid_area_pct": fluid_pct,
            "lesion_area_pct": lesion_pct,
            "retina_affected_pct": 30.0 if disease != "NORMAL" else 0.0,
            "raw": query
        }

    def _interpret_report(self, report: Dict, lang: str, level: str) -> str:
        """Fulfills Feature 16: Advanced Report Interpretation Module."""
        d_code = report["label"]
        conf = report["confidence"]
        sev = report["stage"]
        pct = report["retina_affected_pct"]
        
        disease_names = {
            "DME": ("Diabetic Macular Edema (DME)", "நீரிழிவு விழித்திரை மேக்குலா நீர்வீக்கம் (DME)", "Diabetic Macular Edema (DME)"),
            "CNV": ("Choroidal Neovascularization (CNV)", "விழித்திரைக்கீழ் புதிய இரத்தக்குழாய் உருவாக்கம் (CNV)", "Choroidal Neovascularization (CNV)"),
            "DRUSEN": ("Drusen", "விழித்திரை கொழுப்பு படிவுகள் (Drusen)", "Drusen (fatty spots)"),
            "NORMAL": ("Normal Retina", "இயல்பான விழித்திரை", "Normal healthy retina"),
            "AMD": ("Age-Related Macular Degeneration (AMD)", "வயது முதிர்வு சார்ந்த மேக்குலா சிதைவு (AMD)", "Age-Related Macular Degeneration (AMD)"),
            "GLAUCOMA": ("Glaucoma", "கண் அழுத்த நோய் (Glaucoma)", "Glaucoma (high eye pressure)"),
            "RVO": ("Retinal Vein Occlusion (RVO)", "விழித்திரை நரம்பு அடைப்பு (RVO)", "Retinal Vein Occlusion (RVO)"),
            "RETINOPATHY": ("Diabetic Retinopathy", "நீரிழிவு விழித்திரை நோய்", "Diabetic Retinopathy"),
            "CATARACT": ("Cataract-Related Retinal Complications", "கண்புரை விழித்திரை சிக்கல்கள்", "Cataract-Related Retinal Complications")
        }
        
        d_names = disease_names.get(d_code, (d_code, d_code, d_code))
        d_name = d_names[0] if lang == "english" else (d_names[1] if lang == "tamil" else d_names[2])
        
        # Pathophysiology & terms translation
        patho = ""
        abnormal = ""
        risks = ""
        steps = ""
        
        # Find dictionary key for matching details
        dict_key = d_code
        if d_code in ["CNV", "DRUSEN"]:
            dict_key = "AMD"
        elif d_code in ["RETINOPATHY", "DME"]:
            dict_key = "DR"
            
        doc = RAG_KNOWLEDGE_BASE.get(dict_key, RAG_KNOWLEDGE_BASE["DR"])
        
        if lang == "english":
            abnormal = f"The report identifies **{d_name}** with a confidence score of **{conf:.1f}%** and a **{sev}** severity classification."
            risks = f"Vascular leakage, central macular swelling (edema), capillary blockages, and progressive loss of reading vision."
            steps = "Consult a retinal or glaucoma specialist immediately, undergo dilated fundus imaging, and maintain strict control of HbA1c and systemic blood pressure."
            
            if level == "expert":
                patho = "\n\n**Pathophysiological Assessment (Expert Level):** Chronic microvascular stress blocks capillary perfusion, causing local ischemia. This upregulates Vascular Endothelial Growth Factor (VEGF), leading to breakdown of the blood-retinal barrier, causing intraretinal cystic fluid accumulation."
            elif level == "intermediate":
                patho = f"\n\n**Clinical Breakdown:** Swelling in the macula leads to capillary leakage and structural thickening, compromising central vision. Symptoms correspond to: {doc['symptoms']['english']}"
            else:
                patho = "\n\n**Simple Translation (Beginner Level):** The blood vessels in the back of your eye are damaged and leaking fluid, causing swelling. It is like a wet sponge soaking up water, making the picture blurry."

            resp = f"""### **Advanced Report Analysis & Interpretation**

* **Condition Detected:** {d_name}
* **Confidence Level:** {conf:.1f}%
* **Severity Grade:** {sev}
* **Retinal Region Impacted:** Approximately {pct}% affected area

**Detailed Findings:**
{abnormal} {patho}

**Clinical Risk Profile:**
* {risks}

**Next Steps & Specialist Recommendation:**
* {steps}

*Disclaimer: This analysis is performed automatically by the RAG assistant. Please consult an ophthalmologist for a formal review.*"""

        elif lang == "tamil":
            abnormal = f"அறிக்கை **{d_name}** பாதிப்பைக் காட்டுகிறது. இதன் உறுதிப்பாடு சதவீதம் **{conf:.1f}%** மற்றும் தீவிரம் **{sev}** ஆகும்."
            risks = "விழித்திரை கசிவு, மேக்குலா நீர்வீக்கம், இரத்தக்குழாய் அடைப்பு மற்றும் மத்திய பார்வை இழப்பு ஆகியவை முக்கிய ஆபத்துகளாகும்."
            steps = "அடுத்த கட்டமாக உடனடியாக விழித்திரை சிறப்பு மருத்துவரை (Retinal Specialist) அல்லது கண் அழுத்த நோய் மருத்துவரை அணுகவும், சர்க்கரை மற்றும் இரத்த அழுத்தத்தை கட்டுப்பாட்டில் வைக்கவும்."
            
            if level == "expert":
                patho = "\n\n**pathophysiological மதிப்பீடு (நிபுணர் நிலை):** நாள்பட்ட மைக்ரோவாஸ்குலர் அடைப்பு காரணமாக இஸ்கிமியா (ischemia) ஏற்படுகிறது. இது VEGF-ஐ அதிகரித்து விழித்திரை தடையை உடைத்து திரவ கசிவை உண்டாக்குகிறது."
            elif level == "intermediate":
                patho = f"\n\n**மருத்துவ சுருக்கம்:** விழித்திரையின் மேக்குலா பகுதியில் திரவக் குமிழ்கள் கசிவதால் அடுக்கு தடிமன் அதிகரிக்கிறது. அறிகுறிகள்: {doc['symptoms']['tamil']}"
            else:
                patho = "\n\n**எளிய விளக்கம் (ஆரம்ப நிலை):** கண்ணில் உள்ள சிறிய இரத்த குழாய்களில் கசிவு ஏற்பட்டு, தண்ணீர் தேங்கி பார்வை மங்கலாகிறது. இது கேமராவின் பிலிமில் தண்ணீர் படுவது போலாகும்."

            resp = f"""### **விழித்திரை அறிக்கை பகுப்பாய்வு (Tamil)**

* **கண்டறியப்பட்ட பாதிப்பு:** {d_name}
* **AI உறுதிப்பாடு:** {conf:.1f}%
* **பாதிப்பின் தீவிரம்:** {sev}
* **பாதிப்பு பரப்பு:** தோராயமாக {pct}%

**விவரங்கள்:**
{abnormal} {patho}

**பாதிப்புகள் & அபாயங்கள்:**
* {risks}

**பரிந்துரைக்கப்படும் அடுத்த கட்டங்கள்:**
* {steps}

*எச்சரிக்கை: இந்த அறிக்கை தானியங்கி AI பகுப்பாய்வு மூலம் உருவாக்கப்பட்டது. முறையான சிகிச்சைக்காக கண் மருத்துவரை அணுகவும்.*"""

        else: # tanglish
            abnormal = f"Indha report-la **{d_name}** condition detect aagiyiruku, with a confidence of **{conf:.1f}%** and **{sev}** severity."
            risks = "Blood leak aaguradhu, macula-la fluid swelling (edema), vascular block, matrum severe center visual acuity loss-ku risk iruku."
            steps = "Immediate next step: Oru retina specialist or glaucoma specialist doctor-ai compulsory paarkanum, blood sugar and BP control pannanum."
            
            if level == "expert":
                patho = "\n\n**Pathophysiology (Expert Level):** Chronic hyperglycemia blocks capillaries, causing ischemia which increases VEGF expression, damaging the blood-retinal barrier."
            elif level == "intermediate":
                patho = f"\n\n**Clinical Summary:** Swelling in the macula causes cystic spaces, increasing foveal thickness. Symptoms: {doc['symptoms']['tanglish']}"
            else:
                patho = "\n\n**Simple Explanation (Beginner Level):** Kannula ulla chinna blood pipes damage aagi, neer leak aagi macula space-la swelling yerpaduthudhu. Sponge-la water urinjuna madhiri kannula neer veekkam yerpattu paarvai mangalaagudhu."

            resp = f"""### **Report Interpretation (Tanglish)**

* **Condition Detected:** {d_name}
* **Confidence Level:** {conf:.1f}%
* **Severity Grade:** {sev}
* **Retinal Region Affected:** Approx {pct}% area

**Analysis details:**
{abnormal} {patho}

**Risks involved:**
* {risks}

**Recommended Next Steps:**
* {steps}

*Note: Indha summary automated-ah verify panni generate pannapattadhu. Accurate diagnosis-ku doctor checkup must.*"""

        return resp

    def _explain_prediction(self, scan_data: Dict, lang: str, level: str) -> str:
        """Feature 2 & Feature 7: Explanation of predictions and retinal images."""
        d_code = scan_data.get("label", "DME")
        conf = scan_data.get("confidence", 90.0)
        if conf < 1.0:
            conf = conf * 100
        sev = scan_data.get("stage", scan_data.get("severity", "Moderate"))
        pct = scan_data.get("total_affected_pct", scan_data.get("retina_affected_pct", 30.0))
        fluid = scan_data.get("fluid_area_pct", 0.0)
        lesion = scan_data.get("lesion_area_pct", 0.0)
        edema = scan_data.get("edema_area_pct", 0.0)

        # Map details
        dict_key = d_code
        if d_code in ["CNV", "DRUSEN"]:
            dict_key = "AMD"
        elif d_code in ["RETINOPATHY", "DME"]:
            dict_key = "DR"

        doc = RAG_KNOWLEDGE_BASE.get(dict_key, RAG_KNOWLEDGE_BASE["DR"])
        d_name = doc["tamil_name"] if lang == "tamil" else doc["name"]
        
        # Build explanation of indicators found in the image
        indicators = []
        if fluid > 0:
            indicators.append(f"fluid accumulation ({fluid:.2f}%)" if lang == "english" else (f"திரவக் குவிப்பு ({fluid:.2f}%)" if lang == "tamil" else f"fluid pocket/leakage ({fluid:.2f}%)"))
        if lesion > 0:
            indicators.append(f"lesion deposition ({lesion:.2f}%)" if lang == "english" else (f"விழித்திரை சிதைவுப் பகுதி ({lesion:.2f}%)" if lang == "tamil" else f"lesion markers ({lesion:.2f}%)"))
        if edema > 0:
            indicators.append(f"macular thickening/edema ({edema:.2f}%)" if lang == "english" else (f"விழித்திரை தடிமன்/நீர்வீக்கம் ({edema:.2f}%)" if lang == "tamil" else f"macular edema ({edema:.2f}%)"))
            
        ind_str = ", ".join(indicators) if indicators else ("mild structural changes" if lang == "english" else ("லேசான விழித்திரை மாற்றங்கள்" if lang == "tamil" else "normal layers"))

        if lang == "english":
            patho = ""
            if level == "expert":
                patho = f"\n\n**Pathophysiological Assessment (Expert Level):** The model predicts **{d_name}** because of {ind_str} within the outer retinal layers. This represents a breakdown of the blood-retinal barrier and structural degradation."
            elif level == "intermediate":
                patho = f"\n\n**Clinical Interpretation:** The model indicates a **{sev}** severity stage based on the affected retinal region of **{pct:.2f}%**. The presence of {ind_str} correlates with standard clinical diagnostic criteria."
            else:
                patho = f"\n\n**Analogy (Beginner Level):** The scan shows {ind_str} covering **{pct:.2f}%** of the back of your eye. Think of this like water pooling in the ceiling of a room, which can cause the plaster to sag and blur the view below."

            return f"""### **Image Analysis & Prediction Explanation**

* **AI Diagnosis:** {d_name}
* **Confidence Level:** {conf:.1f}%
* **Severity Grade:** {sev}
* **Region Affected:** {pct:.2f}% of retinal layers

**Abnormalities & Indicators Detected:**
* The scan exhibits **{ind_str}**.

{patho}

**Clinical Management Advice:**
* Follow-up: Consult an ophthalmologist / retinal specialist within the recommended timeframe.
* Preventive Actions: {doc['prevention']['english']}"""

        elif lang == "tamil":
            patho = ""
            if level == "expert":
                patho = f"\n\n**Pathophysiology (நிபுணர் நிலை):** விழித்திரையின் அடியில் கண்டறியப்பட்ட {ind_str} காரணமாக AI மாதிரி **{d_name}** நோயறிதலைக் கணித்துள்ளது. இது இரத்த-விழித்திரை தடையின் முறிவைக் குறிக்கிறது."
            elif level == "intermediate":
                patho = f"\n\n**மருத்துவ சுருக்கம்:** விழித்திரையின் பாதிக்கப்பட்ட பகுதி **{pct:.2f}%** ஆக உள்ளதால், இது **{sev}** மட்டத்தில் வகைப்படுத்தப்பட்டுள்ளது. {ind_str} ஆகியவை முக்கிய நோயறிதல் குறிகாட்டிகளாகும்."
            else:
                patho = f"\n\n**எளிய விளக்கம் (ஆரம்ப நிலை):** உங்கள் கண்ணில் **{pct:.2f}%** அளவுக்கு {ind_str} உள்ளது. விழித்திரை என்பது கேமராவின் பிலிம் போன்றது. அதில் நீர் கசிவது படத்தைத் தெளிவாகக் காட்டாது."

            return f"""### **விழித்திரை படம் & நோயறிதல் விளக்கம்**

* **கண்டறியப்பட்ட பாதிப்பு:** {d_name}
* **AI உறுதிப்பாடு:** {conf:.1f}%
* **தீவிரம்:** {sev}
* **பாதிக்கப்பட்ட பகுதி:** {pct:.2f}%

**கண்டறியப்பட்ட அசாதாரண குறிகாட்டிகள்:**
* விழித்திரையில் **{ind_str}** கண்டறியப்பட்டுள்ளது.

{patho}

**பரிந்துரைகள்:**
* கண் மருத்துவரை அணுகவும்.
* தடுப்பு முறைகள்: {doc['prevention']['tamil']}"""

        else: # tanglish
            patho = ""
            if level == "expert":
                patho = f"\n\n**Pathophysiology (Expert):** Retinal layer check shows {ind_str}. It indicates mechanical leakage, increasing local foveal thickness."
            elif level == "intermediate":
                patho = f"\n\n**Clinical Summary:** Scan shows **{pct:.2f}%** affected retinal area. {ind_str} features point to **{sev}** severity stage."
            else:
                patho = f"\n\n**Simple Explanation (Beginner):** Kannin back camera film layer-la **{pct:.2f}%** அளவுக்கு {ind_str} இருக்கு. Idhu kannula tiny capillaries block-ஆகி, neer pool aaguradha solluvaanga."

            return f"""### **Scan Interpretation & Predictions (Tanglish)**

* **AI predicted:** {d_name}
* **Confidence score:** {conf:.1f}%
* **Severity level:** {sev}
* **Region affected:** {pct:.2f}% area

**Indicators found in scan:**
* Scan-la **{ind_str}** check aagiyiruku.

{patho}

**Doctor Recommendation & Management:**
* Quick doctor consultation recommended.
* Prevention advice: {doc['prevention']['tanglish']}"""

    def _check_symptoms(self, cleaned_query: str) -> Optional[str]:
        """Feature 4: Suggest possible conditions based on symptoms."""
        symptoms_map = {
            "DR": ["swelling", "sugar", "neer veekkam", "fluid leak", "blurry text", "wavy lines reading", "floaters", "spots moving", "fluctuating blur", "சர்க்கரை", "மிதக்கும் கரும்புள்ளிகள்", "நீர்வீக்கம்"],
            "GLAUCOMA": ["side vision", "tunnel vision", "halos", "eye pain", "pressure", "optic nerve", "அழுத்தம்", "கண் வலி", "பக்கவாட்டு பார்வை"],
            "AMD": ["wavy straight lines", "face recognizing", "central blur", "drusen", "gray spot center", "சிதைவு", "கோடுகள் வளைந்து", "மாகுலா சிதைவு"],
            "RVO": ["sudden painless loss", "one eye stroke", "vein block", "ஒற்றை கண்", "அடைப்பு", "திடீர் பார்வை மங்கல்"],
            "CATARACT_RETRAD": ["after cataract surgery", "cloudy lens retina", "irvine-gass", "அறுவை சிகிச்சைக்கு பின்", "கண்புரை"]
        }
        for code, syms in symptoms_map.items():
            if any(s in cleaned_query for s in syms):
                return code
        return None

    def _generate_symptom_response(self, disease_code: str, lang: str) -> str:
        """Returns symptom-based suggestion with disclaimer."""
        doc = RAG_KNOWLEDGE_BASE[disease_code]
        d_name = doc["tamil_name"] if lang == "tamil" else doc["name"]
        
        if lang == "english":
            return f"""### **Symptom-Based Informational Suggestion**

Based on the symptoms described, you might be experiencing indicators associated with **{d_name}**.

* **Symptoms explained:** {doc['symptoms']['english']}
* **Treatment & Guidance:** {doc['treatment']['english']}
* **Prevention & Lifestyle:** {doc['prevention']['english']}

*Disclaimer: This suggestion is for informational purposes only and does NOT constitute a final medical diagnosis. Please seek formal clinical examination.*"""

        elif lang == "tamil":
            return f"""### **அறிகுறி சார்ந்த வழிகாட்டுதல் (Tamil)**

நீங்கள் குறிப்பிட்ட அறிகுறிகள் **{d_name}** பாதிப்போடு ஒத்துப்போகின்றன.

* **அறிகுறிகளின் விவரம்:** {doc['symptoms']['tamil']}
* **சிகிச்சை வழிகாட்டுதல்:** {doc['treatment']['tamil']}
* **வாழ்க்கை முறை மாற்றம்:** {doc['prevention']['tamil']}

*எச்சரிக்கை: இது வழிகாட்டுதலுக்காக மட்டுமே. இது இறுதி நோயறிதல் அல்ல. கண் மருத்துவரை அணுகி முறையான பரிசோதனை செய்து கொள்ளவும்.*"""

        else: # tanglish
            return f"""### **Symptom Suggestion (Tanglish)**

Neenga sonna symptoms check panna pothu, ithu **{d_name}** symptoms-oda match aagalaam.

* **Symptoms detail:** {doc['symptoms']['tanglish']}
* **Treatment options:** {doc['treatment']['tanglish']}
* **Prevention:** {doc['prevention']['tanglish']}

*Note: Ithu intermediate informational guide mattume, final medical confirmation kidaiyathu. Doctor verification must.*"""

    def _generate_research_response(self, cleaned_query: str, lang: str) -> str:
        """Feature 14: Research insights."""
        # Find which disease they might be asking about
        disease_code = "DR"
        for code in RAG_KNOWLEDGE_BASE.keys():
            if code.lower() in cleaned_query:
                disease_code = code
                break
                
        doc = RAG_KNOWLEDGE_BASE[disease_code]
        research = doc["research_insights"][lang]
        
        if lang == "english":
            return f"### **Retinal Research Insight: {doc['name']}**\n\n{research}\n\n*Source: PubMed / Ophthalmology Journals (Simplified Summary)*"
        elif lang == "tamil":
            return f"### **மருத்துவ ஆராய்ச்சி விவரம்: {doc['tamil_name']}**\n\n{research}\n\n*மூலம்: சர்வதேச கண் மருத்துவ இதழ்கள் (எளிய வடிவம்)*"
        else: # tanglish
            return f"### **Research Summary: {doc['name']}**\n\n{research}\n\n*Source: Ophthalmology Research Papers (Simplified)*"

    def _get_faq_response(self, cleaned_query: str, lang: str) -> Optional[str]:
        """Feature 15: FAQ Module."""
        disease_code = None
        for code in RAG_KNOWLEDGE_BASE.keys():
            if code.lower() in cleaned_query:
                disease_code = code
                break
        if not disease_code:
            disease_code = "DR"
            
        doc = RAG_KNOWLEDGE_BASE[disease_code]
        faqs = doc["faqs"]
        
        resp = "### **Frequently Asked Questions (FAQ)**\n\n"
        for i, f in enumerate(faqs, 1):
            resp += f"**Q{i}: {f['q']}**\n"
            resp += f"A{i}: {f['a']}\n\n"
            
        return resp

    def _explain_disease(self, doc: Dict, d_key: str, lang: str, level: str) -> str:
        """Feature 1: Detailed information about retinal diseases."""
        d_name = doc["tamil_name"] if lang == "tamil" else doc["name"]
        
        if lang == "english":
            desc = doc["causes"]["english"]
            syms = doc["symptoms"]["english"]
            risks = doc["risk_factors"]["english"]
            treat = doc["treatment"]["english"]
            prev = doc["prevention"]["english"]
            
            patho = ""
            if level == "expert":
                patho = "\n\n**Pathophysiology (Expert Level):** High mechanical intraocular shear/pressure, or local ischemia, upregulates VEGF and triggers capillary leakage or mechanical structural damage."
            elif level == "intermediate":
                patho = "\n\n**Clinical Pathology:** Leads to fluid pooling or visual fields defects, structural deterioration of nerve layers."
                
            return f"""### **Disease Details: {d_name}**

* **Causes:** {desc} {patho}
* **Key Symptoms:** {syms}
* **Risk Factors:** {risks}
* **Diagnosis Methods:** {doc['diagnosis']['english']}
* **Prevention & Lifestyle:** {prev}
* **Treatments:** {treat}

*Disclaimer: This information is retrieved from trusted retinal guidelines for clinical guidance.*"""

        elif lang == "tamil":
            desc = doc["causes"]["tamil"]
            syms = doc["symptoms"]["tamil"]
            risks = doc["risk_factors"]["tamil"]
            treat = doc["treatment"]["tamil"]
            prev = doc["prevention"]["tamil"]
            
            patho = ""
            if level == "expert":
                patho = "\n\n**Pathophysiology (Expert):** விழித்திரை செல்களின் செயலிழப்பு மற்றும் திரவ கசிவு விழித்திரை நரம்புகளை சேதப்படுத்துகிறது."
            elif level == "intermediate":
                patho = "\n\n**மருத்துவ சுருக்கம்:** லேயர்களில் திரவக் குவிப்பு மற்றும் மேக்குலா தடிமன் அதிகரித்தல்."
                
            return f"""### **விழித்திரை நோய் விளக்கம்: {d_name}**

* **காரணங்கள்:** {desc} {patho}
* **முக்கிய அறிகுறிகள்:** {syms}
* **ஆபத்துக் காரணிகள்:** {risks}
* **கண்டறியும் முறைகள்:** {doc['diagnosis']['tamil']}
* **தடுப்பு & வாழ்க்கைமுறை:** {prev}
* **சிகிச்சைகள்:** {treat}

*எச்சரிக்கை: இது வழிகாட்டுதலுக்காக மட்டுமே. கண் மருத்துவரை அணுகி சிகிச்சை பெறவும்.*"""

        else: # tanglish
            desc = doc["causes"]["tanglish"]
            syms = doc["symptoms"]["tanglish"]
            risks = doc["risk_factors"]["tanglish"]
            treat = doc["treatment"]["tanglish"]
            prev = doc["prevention"]["tanglish"]
            
            patho = ""
            if level == "expert":
                patho = "\n\n**Pathophysiology (Expert):** Chronic leakage causes photoreceptors degeneration, high risk of permanent central vision damage."
            elif level == "intermediate":
                patho = "\n\n**Clinical summary:** Macular thickening and layer swelling."
                
            return f"""### **Disease Details: {d_name} (Tanglish)**

* **Causes:** {desc} {patho}
* **Symptoms:** {syms}
* **Risk Factors:** {risks}
* **Diagnosis:** {doc['diagnosis']['tanglish']}
* **Prevention:** {prev}
* **Treatment:** {treat}

*Note: Information retinal care center recommendations moolam verify panna pattadhu.*"""

    def _generate_chat_report(self, scan_data: Optional[Dict], lang: str) -> str:
        """Feature 11: Structured Report Generation."""
        if not scan_data:
            if lang == "english":
                return "No active patient scan data found in this session. Please upload a scan to generate a report."
            elif lang == "tamil":
                return "இந்த அமர்வில் நோயாளி ஸ்கேன் தரவு எதுவும் இல்லை. அறிக்கை உருவாக்க முதலில் ஸ்கேன் பதிவேற்றவும்."
            else:
                return "Session la current scan data illa. Report generate panna mudiyadhu. Mutha scan upload pannanum."

        d_code = scan_data.get("label", "DME")
        conf = scan_data.get("confidence", 90.0)
        if conf < 1.0:
            conf = conf * 100
        sev = scan_data.get("stage", scan_data.get("severity", "Moderate"))
        pname = scan_data.get("patient_name", "Patient")

        dict_key = d_code
        if d_code in ["CNV", "DRUSEN"]:
            dict_key = "AMD"
        elif d_code in ["RETINOPATHY", "DME"]:
            dict_key = "DR"

        doc = RAG_KNOWLEDGE_BASE.get(dict_key, RAG_KNOWLEDGE_BASE["DR"])
        d_name = doc["tamil_name"] if lang == "tamil" else doc["name"]
        findings = doc["symptoms"][lang]
        recs = doc["treatment"][lang]

        if lang == "english":
            return f"""### **Structured Medical Summary**

| Metric | Patient Value |
| :--- | :--- |
| **Patient Name** | {pname} |
| **Predicted Disease** | {d_name} |
| **Confidence Score** | {conf:.1f}% |
| **Severity Level** | {sev} |
| **Key Findings** | {findings} |
| **Recommendations** | {recs} |

*Follow-up actions: Schedule dilated exam with Retinal Specialist.*"""

        elif lang == "tamil":
            return f"""### **கட்டமைக்கப்பட்ட மருத்துவ சுருக்க அறிக்கை**

| விவரம் | மதிப்பு |
| :--- | :--- |
| **நோயாளி பெயர்** | {pname} |
| **நோயறிதல் கணிப்பு** | {d_name} |
| ** AI உறுதிப்பாடு** | {conf:.1f}% |
| **பாதிப்பின் தீவிரம்** | {sev} |
| **முக்கிய கண்டுபிடிப்புகள்** | {findings} |
| **பரிந்துரைகள்** | {recs} |

*அடுத்த கட்டம்: விழித்திரை சிறப்பு மருத்துவரை அணுகவும்.*"""

        else: # tanglish
            return f"""### **Structured Report Summary (Tanglish)**

| Metric | Details |
| :--- | :--- |
| **Patient Name** | {pname} |
| **AI Prediction** | {d_name} |
| **Confidence** | {conf:.1f}% |
| **Severity Level** | {sev} |
| **Findings** | {findings} |
| **Recommendations** | {recs} |

*Follow-up: Immediate eye checkup with retina specialist.*"""

    def _fallback_response(self, lang: str) -> str:
        """Specialty summary fallback."""
        if lang == "english":
            return "I am an AI Retinal Healthcare Assistant. I can help explain scan predictions, retinal images, abnormalities, and medical reports. I specialize in Diabetic Macular Edema (DME), Glaucoma, Age-Related Macular Degeneration (AMD), Diabetic Retinopathy, Retinal Vein Occlusion (RVO), and Cataract-Related Retinal Complications. What condition or report would you like to discuss?"
        elif lang == "tamil":
            return "நான் ஒரு விழித்திரை மருத்துவ AI உதவியாளர். கண் ஸ்கேன் முடிவுகள், படங்கள் மற்றும் மருத்துவ அறிக்கைகளை விளக்க நான் உதவ முடியும். நீரிழிவு விழித்திரை நோய் (DME), கண் அழுத்த நோய் (Glaucoma), வயது முதிர்வு விழித்திரை நோய் (AMD), விழித்திரை நரம்பு அடைப்பு (RVO) ஆகிய விழித்திரை சிக்கல்களில் நான் நிபுணத்துவம் பெற்றுள்ளேன். நீங்கள் எதைப் பற்றி அறிய விரும்புகிறீர்கள்?"
        else: # tanglish
            return "Naan oru Retinal Medical AI Assistant. Scan predictions, retinal images, abnormal features, matrum medical reports-ai explain panna help pannuven. DME, Glaucoma, AMD, DR, RVO, and Cataract complications-la specialize panniruken. Ungaluku enna condition pathi theriyanum?"

# Session memory dictionary for conversation context
SESSION_MEMORY: Dict[int, Dict[str, Any]] = {}

# Lazy-loaded real FAISS index singleton
faiss_index_instance: Optional[RetinalFAISSIndex] = None

def get_faiss_index() -> RetinalFAISSIndex:
    global faiss_index_instance
    if faiss_index_instance is None:
        faiss_index_instance = RetinalFAISSIndex()
    return faiss_index_instance

# Llama 3 model singleton
llama3_engine = Llama3InferenceEngine()

# decline message dictionary for non-medical topics
DECLINE_MESSAGES = {
    "english": "Sorry, I am a Medical AI Assistant and can only answer medical-related questions. Please ask a question related to retinal diseases, eye health, diagnosis, treatment, or other medical topics.",
    "tamil": "மன்னிக்கவும், நான் ஒரு மருத்துவ AI உதவியாளர் மற்றும் மருத்துவம் சார்ந்த கேள்விகளுக்கு மட்டுமே பதிலளிக்க முடியும். விழித்திரை நோய்கள், கண் ஆரோக்கியம், கண்டறிதல், சிகிச்சை அல்லது பிற மருத்துவ தலைப்புகள் தொடர்பான கேள்வியைக் கேட்கவும்.",
    "tanglish": "Sorry, naan oru Medical AI Assistant. Ennala medical-related questions-ku mattum dhaan reply panna mudiyum. Retinal diseases, eye health, diagnosis, treatment or other medical topics pathi kelvinga."
}

def is_medical_query(query: str) -> bool:
    """Classifies query as eye/medical related."""
    norm = query.lower()
    medical_keywords = [
        "eye", "retina", "macula", "fluid", "lesion", "edema", "cnv", "drusen",
        "glaucoma", "diabetic", "cataract", "vein", "occlusion", "blindness",
        "vision", "see", "optic", "nerve", "pressure", "iop", "floaters", "halos",
        "doctor", "ophthalmologist", "specialist", "treatment", "cure", "prevent",
        "scan", "prediction", "report", "pdf", "symptom", "stage", "severity",
        "dme", "cscr", "erm", "mh", "rd", "rp", "irvine-gass", "clinical", "med",
        "கண்", "பார்வை", "அழுத்தம்", "சர்க்கரை", "வலி", "அடைப்பு", "நீர்வீக்கம்",
        "சிதைவு", "விளக்கவும்", "ஆராய்ச்சி", "அறிக்கை", "சிகிச்சை", "மருந்து",
        "kannu", "paarvai", "rathakasiya", "veekkam", "marundhu", "sikichai",
        "valainju", "iruku", "epdi", "enna", "sugars", "clinical trial"
    ]
    return any(kw in norm for kw in medical_keywords)

def answer_from_attached_file(file_text: str, query: str, filename: str) -> str:
    cleaned_query = query.lower()
    
    # Heuristics for file Q&A
    if any(w in cleaned_query for w in ["summarize", "summary", "brief", "what is this", "explain this", "சுருக்கம்", "vilakku", "explain panna"]):
        sentences = [s.strip() for s in re.split(r'\.|\n', file_text) if len(s.strip()) > 10]
        return "\n- ".join(sentences[:6])
        
    # Word search matching
    sentences = [s.strip() for s in re.split(r'\.|\n', file_text) if len(s.strip()) > 5]
    query_words = [w for w in re.findall(r'\b\w+\b', cleaned_query) if len(w) > 2 and w not in ["what", "how", "why", "who", "when", "where", "the", "and", "for", "with", "this", "that", "from", "sollunga", "enna", "epdi"]]
    
    matched_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if any(qw in s_lower for qw in query_words):
            if s not in matched_sentences:
                matched_sentences.append(s)
                
    if not matched_sentences:
        matched_sentences = sentences[:5]
        
    return "\n- ".join(matched_sentences[:6])

def set_session_attachment(user_id: int, attachment_data: Dict[str, Any]):
    if user_id not in SESSION_MEMORY:
        SESSION_MEMORY[user_id] = {
            "language": "english",
            "last_scan": None,
            "memory": ConversationBufferMemory(memory_key="history")
        }
    SESSION_MEMORY[user_id]["attached_file"] = attachment_data

def clear_session_attachment(user_id: int):
    if user_id in SESSION_MEMORY:
        SESSION_MEMORY[user_id].pop("attached_file", None)

def process_chat_query_rag(message: str, user_id: int, scan_data: Optional[Dict] = None) -> Dict[str, str]:
    """Retrieves context using Sentence Transformers & FAISS, templates prompts using LangChain, and infers Llama 3 response."""
    cleaned = message.strip().lower()
    
    # 1. Initialize or Load Session Memory Context
    if user_id not in SESSION_MEMORY:
        SESSION_MEMORY[user_id] = {
            "language": "english",
            "last_scan": None,
            "memory": ConversationBufferMemory(memory_key="history")
        }
    session = SESSION_MEMORY[user_id]
    
    # Track the last scan data in active session
    if scan_data:
        session["last_scan"] = scan_data
    elif session["last_scan"]:
        scan_data = session["last_scan"]
        
    # 2. Detect requested/preferred language
    if any(w in cleaned for w in ["english", "in english", "explain in english"]):
        session["language"] = "english"
    elif any(w in cleaned for w in ["tamil", "தமிழில்", "விளக்கவும்", "தமிழ்"]):
        session["language"] = "tamil"
    elif any(w in cleaned for w in ["tanglish", "la explain", "pannunga", "tanglish la", "sollunga"]):
        session["language"] = "tanglish"
    else:
        # Auto-detect language
        if any('\u0b80' <= char <= '\u0bff' for char in message):
            session["language"] = "tamil"
        else:
            # check tanglish vocabulary
            tanglish_words = {"enna", "epdi", "epadi", "iruku", "irukku", "irukinga", "panna", "pannunga", "panunga", "tanglish", "ku", "kannula", "ratha", "sikichai", "marundhu", "sollunga", "sollu", "solunga", "pathi", "enaku", "la", "romba"}
            words = set(re.findall(r'\b[a-z]+\b', cleaned))
            if words.intersection(tanglish_words):
                session["language"] = "tanglish"
            else:
                session["language"] = "english"

    lang = session["language"]
    
    thought_steps = [
        f"Input query: '{message}'",
        f"Language set to: {lang.upper()}"
    ]

    # 3. Retrieve context and execute FAISS
    attached_file = session.get("attached_file")
    retrieved_docs = []
    context_list = []
    
    if is_medical_query(message):
        thought_steps.append("Medical query intent identified. Querying local FAISS vector database...")
        index = get_faiss_index()
        retrieved_docs = index.similarity_search(message, k=3)
        for doc in retrieved_docs:
            meta = doc["metadata"]
            context_list.append(f"[{doc['source']}]: {meta['text']}")
        if retrieved_docs:
            thought_steps.append(f"Retrieved {len(retrieved_docs)} matching documents from vector database.")
        else:
            thought_steps.append("No direct matches found in vector database.")
    else:
        thought_steps.append("Query is general/conversational. Bypassing vector database search.")
        
    if attached_file:
        filename = attached_file.get("filename", "document")
        thought_steps.append(f"Processing context from attached file: '{filename}'.")
        if not attached_file.get("is_medical_scan"):
            doc_text = attached_file.get("text", "")
            doc_context = answer_from_attached_file(doc_text, message, filename)
            context_list.append(f"[Attached File Context - {filename}]: {doc_context}")
        else:
            context_list.append(f"[Attached Scan Analysis - {filename}]: {attached_file.get('label_full')} detected with {attached_file.get('confidence')}% confidence, stage {attached_file.get('stage')}.")

    context_str = "\n\n".join(context_list) if context_list else "None"

    # 4. Load Memory History Variables
    memory_vars = session["memory"].load_memory_variables()
    history_str = memory_vars.get("history", "")

    # Format active scan information
    scan_info_str = "None"
    if scan_data:
        scan_info_str = f"Patient: {scan_data.get('patient_name', 'Patient')}, Diagnosis: {scan_data.get('label', 'DME')}, Confidence: {scan_data.get('confidence', 90.0):.1f}%, Severity: {scan_data.get('stage', 'Moderate')}, Retina affected: {scan_data.get('total_affected_pct', 30.0):.2f}%"
        thought_steps.append(f"Scan context found: Patient '{scan_data.get('patient_name')}' diagnosed with '{scan_data.get('label')}' (Confidence: {scan_data.get('confidence', 0.0):.1f}%).")
    else:
        thought_steps.append("No active scan context attached to session.")

    # Determine explanation level
    level = "beginner"
    if any(w in cleaned for w in ["expert", "professional", "technical", "scientific", "researcher", "clinician", "physician", "pathophysiology"]):
        level = "expert"
    elif any(w in cleaned for w in ["intermediate", "detailed", "moderate", "clinical summary", "doctor explanation"]):
        level = "intermediate"
    thought_steps.append(f"User explanation level detected: {level.upper()}.")

    # 5. Format Prompt Template using LangChain
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        history=history_str,
        query=message,
        language=lang,
        level=level,
        scan_info=scan_info_str
    )

    thought_steps.append("Assembling RAG prompts and invoking Llama 3 quantized clinical inference model...")
    thought_str = "\n".join([f"- {step}" for step in thought_steps])

    # 6. Generate Response from Llama 3 Engine
    response_text = llama3_engine.generate_response(prompt, message, retrieved_docs, scan_data, session)
    
    # 7. Save to Session Memory
    session["memory"].save_context({"user": message}, {"bot": response_text})
    
    return {
        "response": response_text,
        "thought": thought_str,
        "language": lang
    }
