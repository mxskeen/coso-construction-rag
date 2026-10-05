import json
import os
from typing import List, Dict, Any

CONSTRUCTION_BENCHMARKS = [
    # --- IS 456: ENGINEERING SPECIFICATIONS ---
    {
        "query": "What is the minimum nominal cover for concrete elements in severe environmental exposure?",
        "target_doc": "IS456-2000_concrete-code_EN.pdf",
        "target_clause": "Clause 26.4",
        "domain": "ENGINEERING_STANDARD",
        "positive": "Nominal cover to meet durability requirements. For severe exposure condition, minimum nominal cover required is 45 mm (Table 16).",
        "hard_negative": "Clause 26.5: Minimum Reinforcement requirements for beams and slabs specifying 0.12 percent of gross cross sectional area for mild steel.",
        "ground_truth_answer": "Under IS 456 Table 16 / Clause 26.4, the minimum nominal cover for severe exposure condition is 45 mm."
    },
    {
        "query": "What is the minimum period of curing required for concrete using ordinary Portland cement?",
        "target_doc": "IS456-2000_concrete-code_EN.pdf",
        "target_clause": "Clause 13.5",
        "domain": "ENGINEERING_STANDARD",
        "positive": "Clause 13.5 Curing: Moist curing of concrete using Ordinary Portland Cement shall be carried out for at least 7 days, and at least 10 days where mineral admixtures or blended cements are used.",
        "hard_negative": "Clause 14.1: Work in Extreme Weather Conditions requiring special precautions for cold weather and hot weather concreting.",
        "ground_truth_answer": "Under IS 456 Clause 13.5, concrete made with Ordinary Portland Cement must be moist cured for a minimum of 7 days."
    },
    {
        "query": "What is the characteristic compressive strength definition and minimum grade for reinforced concrete?",
        "target_doc": "IS456-2000_concrete-code_EN.pdf",
        "target_clause": "Clause 6.1",
        "domain": "ENGINEERING_STANDARD",
        "positive": "Clause 6.1 Characteristic Strength: The compressive strength of concrete is based on 150 mm cube strength at 28 days. The minimum grade of concrete for reinforced concrete work shall not be less than M20.",
        "hard_negative": "Clause 5.1: Cement conforming to IS 269 33-grade, IS 8112 43-grade, and IS 12269 53-grade ordinary Portland cement.",
        "ground_truth_answer": "According to IS 456 (Clause 6.1 & Table 2), characteristic strength is the 150 mm cube compressive strength at 28 days below which not more than 5% of test results fall. The minimum grade for RCC is M20."
    },
    {
        "query": "What is the maximum free water-cement ratio and minimum cement content for RCC under moderate exposure?",
        "target_doc": "IS456-2000_concrete-code_EN.pdf",
        "target_clause": "Table 5",
        "domain": "ENGINEERING_STANDARD",
        "positive": "Table 5 Minimum cementitious content, maximum free water-cement ratio and minimum grade of concrete for moderate exposure in RCC: Minimum cement 300 kg/m3, maximum w/c ratio 0.50, minimum grade M25.",
        "hard_negative": "Table 2: Grades of Concrete listing ordinary concrete M10 to M20, standard concrete M25 to M55, and high strength concrete M60 to M80.",
        "ground_truth_answer": "IS 456 Table 5 prescribes a minimum cement content of 300 kg/m3, a maximum free water-cement ratio of 0.50, and minimum grade M25 for reinforced concrete under moderate exposure."
    },
    {
        "query": "What is the lap length and splicing requirement for flexural tension reinforcement in beams?",
        "target_doc": "IS456-2000_concrete-code_EN.pdf",
        "target_clause": "Clause 26.2.5",
        "domain": "ENGINEERING_STANDARD",
        "positive": "Clause 26.2.5.1 Lap splices: Lap length in tension shall not be less than the development length or 30 times the bar diameter. Splicing shall not be made at sections where the bending moment is more than 50% of the design moment.",
        "hard_negative": "Clause 26.5.1.1: Minimum tension reinforcement in beams where As / (b * d) = 0.85 / fy.",
        "ground_truth_answer": "Under IS 456 Clause 26.2.5.1, the lap length in tension must equal the development length (Ld) but not less than 30 times the bar diameter, avoiding locations of maximum moment."
    },

    # --- CPWD GCC 2020: CONTRACTUAL & LEGAL CLAUSES ---
    {
        "query": "What is the penalty or compensation for delay if a contractor fails to complete work on time?",
        "target_doc": "CPWD-GCC-2020_construction-contract_EN.pdf",
        "target_clause": "Clause 2",
        "domain": "CONTRACT_LEGAL",
        "positive": "Clause 2 Compensation for Delay: If the contractor fails to maintain the required progress or to complete the work on or before the contract completion date, he shall pay compensation at 1.0% per month of delay calculated on daily basis, subject to a maximum of 10% of the tendered value.",
        "hard_negative": "Clause 2A: Incentive for early completion: If the contractor completes work before stipulated date, bonus at 1% per month is paid subject to max 5%.",
        "ground_truth_answer": "CPWD GCC Clause 2 states that delay compensation is levied at 1.0% per month of delay calculated on a per-day basis, capped at a maximum of 10% of the tendered contract value."
    },
    {
        "query": "What are the rules and timeline for submission of Performance Guarantee by the contractor?",
        "target_doc": "CPWD-GCC-2020_construction-contract_EN.pdf",
        "target_clause": "Clause 1",
        "domain": "CONTRACT_LEGAL",
        "positive": "Clause 1: The contractor shall submit an irrevocable Performance Guarantee of 5% (Five percent) of the tendered amount within the period specified in Schedule F from date of issue of letter of acceptance.",
        "hard_negative": "Clause 1A: Recovery of Security Deposit: Sum deducted at 2.5% from each running bill till the total security deposit is accumulated.",
        "ground_truth_answer": "Under CPWD GCC Clause 1, the contractor must submit an irrevocable Performance Guarantee of 5% of the tendered amount within the schedule F period (usually 7-14 days)."
    },
    {
        "query": "Under what conditions can the contract be determined or terminated by the Engineer-in-Charge?",
        "target_doc": "CPWD-GCC-2020_construction-contract_EN.pdf",
        "target_clause": "Clause 3",
        "domain": "CONTRACT_LEGAL",
        "positive": "Clause 3 When Contract can be Determined: The Engineer-in-Charge may determine the contract if contractor fails to proceed with due diligence, persistently disregards engineer's instructions, or assigns or sublets the contract without approval.",
        "hard_negative": "Clause 14: Carrying out part work at risk & cost of contractor without determining the entire contract.",
        "ground_truth_answer": "CPWD GCC Clause 3 empowers the Engineer-in-Charge to determine the contract if the contractor abandons the work, fails to rectify defects, persists in non-compliance, or assigns/sublets unauthorizedly."
    },
    {
        "query": "What is the contractor's liability for rectification of defects during the defect liability period?",
        "target_doc": "CPWD-GCC-2020_construction-contract_EN.pdf",
        "target_clause": "Clause 17",
        "domain": "CONTRACT_LEGAL",
        "positive": "Clause 17 Contractor liable for damages, defects during maintenance period: If contractor or his workmen cause damage, or if any defect or imperfection appears within 12 months (or as specified in Schedule F) after certificate of completion, contractor must amend and make good at his own expense.",
        "hard_negative": "Clause 18: Contractor to supply tools, plant, scaffolding, ladders and other appliances required for proper execution.",
        "ground_truth_answer": "Under CPWD GCC Clause 17, the contractor must rectify and make good all defects appearing within 12 months after the completion certificate at their own expense."
    },
    {
        "query": "What is the procedure for settlement of disputes and appointment of arbitrator under CPWD GCC?",
        "target_doc": "CPWD-GCC-2020_construction-contract_EN.pdf",
        "target_clause": "Clause 25",
        "domain": "CONTRACT_LEGAL",
        "positive": "Clause 25 Settlement of Disputes & Arbitration: Except where otherwise provided, all questions and disputes shall first be referred to Dispute Redressal Committee (DRC). If unresolved within 90 days, matter is referred to sole arbitration.",
        "hard_negative": "Clause 29: Withholding and lien in respect of sums claimed by Government against contractor.",
        "ground_truth_answer": "Under CPWD GCC Clause 25, disputes must first be submitted to the Dispute Redressal Committee (DRC); if not resolved, they proceed to arbitration by an appointed arbitrator."
    },

    # --- PMGSY-III: MULTILINGUAL / HINDI ROAD SCHEME ---
    {
        "query": "प्रधानमंत्री ग्राम सड़क योजना-III के तहत कुल कितनी लंबाई की सड़कें बनाने का लक्ष्य है और अनुमानित लागत क्या है?",
        "target_doc": "PMGSY-III-summary_HINDI_selectable-text.pdf",
        "target_clause": "प्रमुख बिंदु",
        "domain": "SCHEME_POLICY",
        "positive": "प्रमुख बिंदु: PMGSY-III के अंतर्गत देश भर में 1,25,000 किलोमीटर लंबी सड़कें बनाने की योजना है जिसकी अनुमानित लागत लगभग 80,250 करोड़ रुपए है।",
        "hard_negative": "पृष्ठभूमि: PMGSY दिसंबर 2000 में लॉन्च की गई थी जिसका उद्देश्य 500+ मैदानी और 250+ पहाड़ी क्षेत्रों को कनेक्टिविटी देना था।",
        "ground_truth_answer": "PMGSY-III के अंतर्गत कुल 1,25,000 किमी सड़कों के निर्माण का लक्ष्य है, जिसकी अनुमानित लागत लगभग 80,250 करोड़ रुपये है।"
    },
    {
        "query": "What is the fund sharing ratio between Central and State governments under PMGSY-III?",
        "target_doc": "PMGSY-III-summary_HINDI_selectable-text.pdf",
        "target_clause": "वित्तीय हिस्सेदारी",
        "domain": "SCHEME_POLICY",
        "positive": "वित्तीय हिस्सेदारी: केंद्र एवं राज्यों के बीच निधियों की हिस्सेदारी 60:40 के अनुपात में होगी, लेकिन 8 पूर्वोत्तर राज्यों तथा तीन हिमालयी राज्यों (जम्मू और कश्मीर, हिमाचल प्रदेश, उत्तराखंड) में यह 90:10 के अनुपात में होगी। कुल लागत 80,250 करोड़ में केंद्र का हिस्सा 53,800 करोड़ और राज्य का 26,450 करोड़ है।",
        "hard_negative": "योजना का क्रियान्वयन: ग्रामीण विकास मंत्रालय एवं राज्य सरकारों द्वारा क्रियान्वित अवधि 2019-20 से 2024-25 तक निर्धारित की गई है।",
        "ground_truth_answer": "Under PMGSY-III, standard fund sharing between Centre and States is 60:40, whereas for 8 North-Eastern states and 3 Himalayan states, it is 90:10."
    },
    {
        "query": "PMGSY-III में मैदानी और पहाड़ी क्षेत्रों में पुलों की अधिकतम लंबाई के क्या नए प्रावधान हैं?",
        "target_doc": "PMGSY-III-summary_HINDI_selectable-text.pdf",
        "target_clause": "योजना का क्रियान्वयन",
        "domain": "SCHEME_POLICY",
        "positive": "योजना का क्रियान्वयन: मैदानी क्षेत्रों में 150 मीटर तक लंबे पुलों का निर्माण और हिमालयी तथा पूर्वोत्तर राज्यों में 200 मीटर तक लंबे पुलों के निर्माण का प्रस्ताव है (वर्तमान प्रावधान क्रमशः 75 मीटर तथा 100 मीटर था)।",
        "hard_negative": "प्रमुख बिंदु: PMGSY के अंतर्गत बनी सड़कों का रखरखाव ग्रामीण विकास मंत्रालय एवं राज्य सरकारों द्वारा किया जाएगा।",
        "ground_truth_answer": "PMGSY-III के तहत पुलों की अनुमेय लंबाई मैदानी क्षेत्रों में 150 मीटर तक और हिमालयी/पूर्वोत्तर राज्यों में 200 मीटर तक बढ़ाई गई है।"
    },
    {
        "query": "What is the implementation timeline and duration for PMGSY-III?",
        "target_doc": "PMGSY-III-summary_HINDI_selectable-text.pdf",
        "target_clause": "योजना का क्रियान्वयन",
        "domain": "SCHEME_POLICY",
        "positive": "योजना का क्रियान्वयन: ग्रामीण विकास मंत्रालय एवं राज्य सरकारों द्वारा क्रियान्वित की जाने वाली प्रधानमंत्री ग्राम सड़क योजना-III की अवधि 2019-20 से 2024-25 तक निर्धारित की गई है।",
        "hard_negative": "पृष्ठभूमि: सरकार द्वारा वर्ष 2016 में चरमपंथ प्रभावित क्षेत्रों के लिए पृथक सड़क कनेक्टिविटी परियोजना लॉन्च की गई।",
        "ground_truth_answer": "The implementation period for PMGSY-III is from 2019-20 to 2024-25."
    },
    {
        "query": "What is the maintenance commitment required from States after road construction under PMGSY-III?",
        "target_doc": "PMGSY-III-summary_HINDI_selectable-text.pdf",
        "target_clause": "योजना का क्रियान्वयन",
        "domain": "SCHEME_POLICY",
        "positive": "राज्यों से PMGSY-III लॉन्च किये जाने से पहले समझौता ज्ञापन (MoU) करने को कहा जाएगा, ताकि PMGSY के अंतर्गत 5 वर्ष की निर्माण रखरखाव अवधि के बाद सड़कों के रखरखाव के लिये पर्याप्त धन उपलब्ध कराया जा सके।",
        "hard_negative": "चर्चा में क्यों?: हाल ही में मंत्रिमंडल की आर्थिक समिति ने पूरे देश में ग्रामीण सड़क कनेक्टिविटी को और मज़बूत बनाने के लिये मंजूरी दी।",
        "ground_truth_answer": "States must sign a Memorandum of Understanding (MoU) ensuring funds for maintenance following the mandatory initial 5-year post-construction maintenance period."
    }
]


def export_benchmark_dataset(output_path: str = "data/eval_benchmarks.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(CONSTRUCTION_BENCHMARKS, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(CONSTRUCTION_BENCHMARKS)} benchmark triplets to {output_path}")


if __name__ == "__main__":
    export_benchmark_dataset()
