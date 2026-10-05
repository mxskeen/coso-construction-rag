import os
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer import losses

TRAINING_CONSTRUCTION_PAIRS = [
    # IS 456 Concrete Code
    {
        "query": "What is the minimum grade of concrete for reinforced concrete?",
        "positive": "IS 456 Clause 6.1 & Table 2: The minimum grade of concrete for reinforced concrete work shall not be less than M 20.",
        "negative": "IS 456 Table 2: Grades of concrete designated as Ordinary Concrete M 10, M 15, M 20, and Standard Concrete M 25 to M 55."
    },
    {
        "query": "What is the required moist curing period for OPC concrete?",
        "positive": "IS 456 Clause 13.5 Curing: Exposed surfaces of concrete shall be kept continuously in a damp or wet condition by ponding or by other means for at least 7 days from the date of placing concrete made with Ordinary Portland Cement.",
        "negative": "IS 456 Clause 14.1 Work in Extreme Weather Conditions: Concreting in hot weather and cold weather requires special precautions."
    },
    {
        "query": "What is the minimum nominal cover for severe exposure condition?",
        "positive": "IS 456 Table 16 Nominal Cover: Minimum nominal cover to meet durability requirements for Severe environmental exposure condition is 45 mm.",
        "negative": "IS 456 Table 16: Minimum nominal cover for Moderate exposure condition is 30 mm and for Mild exposure condition is 20 mm."
    },
    {
        "query": "What is the maximum water cement ratio for RCC in moderate conditions?",
        "positive": "IS 456 Table 5: For reinforced concrete in moderate exposure, the maximum free water-cement ratio is 0.50 and minimum cement content is 300 kg/m3.",
        "negative": "IS 456 Table 5: For plain concrete in moderate exposure, maximum free water-cement ratio is 0.60 and minimum cement is 240 kg/m3."
    },
    {
        "query": "What are the rules for lap length in flexural tension reinforcement?",
        "positive": "IS 456 Clause 26.2.5.1 Lap Splices: Lap length in tension shall be equal to development length (Ld) but shall not be less than 30 times bar diameter. Splicing shall not be made at sections of maximum bending moment.",
        "negative": "IS 456 Clause 26.5.1.1 Minimum reinforcement in beams: As / (b * d) = 0.85 / fy."
    },
    {
        "query": "What is the permissible slump for standard pumping and reinforced foundation concrete?",
        "positive": "IS 456 Clause 7.1 Workability Table: Heavily reinforced sections in slabs, beams, walls, columns and pumped concrete require slump between 50 mm to 100 mm (Medium workability).",
        "negative": "IS 456 Clause 5.3 Aggregates: Nominal maximum size of coarse aggregate should be as large as possible."
    },
    {
        "query": "How is 28 day characteristic compressive strength defined in Indian standards?",
        "positive": "IS 456 Clause 6.1: Characteristic strength is defined as the compressive strength of 150 mm size cubes at 28 days below which not more than 5 percent of the test results are expected to fall.",
        "negative": "IS 456 Clause 15.1 Sampling and Acceptance Criteria: Samples from fresh concrete shall be taken as per IS 1199."
    },
    {
        "query": "What is the minimum longitudinal reinforcement required in columns?",
        "positive": "IS 456 Clause 26.5.3.1 Longitudinal reinforcement: The cross-sectional area of longitudinal reinforcement in columns shall be not less than 0.8 percent nor more than 6.0 percent of gross cross-sectional area.",
        "negative": "IS 456 Clause 26.5.3.2 Transverse reinforcement: Helical or lateral ties pitch and diameter specifications."
    },

    # CPWD GCC 2020 Construction Contracts
    {
        "query": "What is the compensation or penalty for delay in project completion?",
        "positive": "CPWD GCC Clause 2 Compensation for Delay: If the contractor fails to maintain required progress or complete the work on or before contract date, he shall pay compensation at 1.0% per month of delay calculated on daily basis, capped at 10% of tendered value.",
        "negative": "CPWD GCC Clause 2A Incentive for early completion: Bonus at 1% per month for completion before stipulated date, subject to maximum of 5%."
    },
    {
        "query": "What percentage is required for performance guarantee under CPWD?",
        "positive": "CPWD GCC Clause 1 Performance Guarantee: The contractor shall submit an irrevocable Performance Guarantee of 5% (Five percent) of the tendered amount within the period specified in Schedule F.",
        "negative": "CPWD GCC Clause 1A Recovery of Security Deposit: Sum deducted at 2.5% from running bills until full security deposit is recovered."
    },
    {
        "query": "When can the government terminate or determine the contract for contractor default?",
        "positive": "CPWD GCC Clause 3 When Contract can be Determined: Engineer-in-Charge may determine the contract if contractor fails to proceed with due diligence, persistently disregards engineer's instructions, or assigns or sublets the contract without approval.",
        "negative": "CPWD GCC Clause 14: Carrying out part work at risk & cost of contractor without determining the entire contract."
    },
    {
        "query": "What is the defect liability period and contractor duty to repair defects?",
        "positive": "CPWD GCC Clause 17 Contractor liable for damages and defects: If any defect or imperfection appears within 12 months after certificate of completion, contractor must amend and make good at his own expense.",
        "negative": "CPWD GCC Clause 18: Contractor to supply plant, ladder, scaffolding and tools for work."
    },
    {
        "query": "How are disputes resolved before going to arbitration in CPWD contracts?",
        "positive": "CPWD GCC Clause 25 Settlement of Disputes & Arbitration: All questions and disputes shall first be referred to Dispute Redressal Committee (DRC). If unresolved within 90 days, matter is referred to sole arbitration.",
        "negative": "CPWD GCC Clause 29: Withholding and lien in respect of sums claimed by Government against contractor."
    },
    {
        "query": "What are the rules for extension of time due to force majeure or engineer delay?",
        "positive": "CPWD GCC Clause 5 Extension of Time: If work is delayed by force majeure, abnormally bad weather, or serious loss by fire, contractor shall immediately give notice to Engineer-in-Charge who may grant reasonable extension of time.",
        "negative": "CPWD GCC Clause 6 / 6A: Measurement of Work Done and computerized measurement books."
    },
    {
        "query": "Can the contractor claim price escalation on materials and labour?",
        "positive": "CPWD GCC Clause 10CC: Payment due to increase/decrease in prices of materials and labour after receipt of tender, calculated using standard price adjustment formulas.",
        "negative": "CPWD GCC Clause 10B: Mobilization advance and plant & machinery advance terms."
    },

    # PMGSY-III Government Scheme (Hindi / Multilingual)
    {
        "query": "What is the total road target and budget outlay under PMGSY-III?",
        "positive": "PMGSY-III प्रमुख बिंदु: इसके अंतर्गत 1,25,000 किलोमीटर लंबी सड़कें बनाने की योजना है जिसकी अनुमानित लागत लगभग 80,250 करोड़ रुपए है।",
        "negative": "PMGSY-III पृष्ठभूमि: PMGSY दिसंबर 2000 में लॉन्च की गई थी जिसका उद्देश्य 500+ मैदानी और 250+ पहाड़ी क्षेत्रों को कनेक्टिविटी देना था।"
    },
    {
        "query": "What is the fund sharing ratio between Centre and States in PMGSY-III?",
        "positive": "PMGSY-III वित्तीय हिस्सेदारी: केंद्र एवं राज्यों के बीच निधियों की हिस्सेदारी 60:40 के अनुपात में होगी, लेकिन 8 पूर्वोत्तर राज्यों तथा तीन हिमालयी राज्यों में 90:10 के अनुपात में होगी।",
        "negative": "PMGSY-III योजना का क्रियान्वयन: ग्रामीण विकास मंत्रालय एवं राज्य सरकारों द्वारा क्रियान्वित अवधि 2019-20 से 2024-25 तक निर्धारित की गई है।"
    },
    {
        "query": "What are the permissible bridge lengths under PMGSY-III for plain and hilly regions?",
        "positive": "PMGSY-III योजना का क्रियान्वयन: मैदानी क्षेत्रों में 150 मीटर तक लंबे पुलों का निर्माण और हिमालयी तथा पूर्वोत्तर राज्यों में 200 मीटर तक लंबे पुलों के निर्माण का प्रस्ताव है।",
        "negative": "PMGSY-III प्रमुख बिंदु: PMGSY के अंतर्गत बनी सड़कों का रखरखाव ग्रामीण विकास मंत्रालय एवं राज्य सरकारों द्वारा किया जाएगा।"
    },
    {
        "query": "What is the duration and execution period of PMGSY Phase 3?",
        "positive": "PMGSY-III योजना का क्रियान्वयन: ग्रामीण विकास मंत्रालय एवं राज्य सरकारों द्वारा क्रियान्वित की जाने वाली प्रधानमंत्री ग्राम सड़क योजना-III की अवधि 2019-20 से 2024-25 तक निर्धारित है।",
        "negative": "PMGSY-III पृष्ठभूमि: सरकार द्वारा वर्ष 2016 में चरमपंथ प्रभावित क्षेत्रों के लिए पृथक सड़क कनेक्टिविटी परियोजना लॉन्च की गई।"
    },
    {
        "query": "What is the post-construction maintenance agreement requirement for States under PMGSY?",
        "positive": "PMGSY-III योजना का क्रियान्वयन: राज्यों से PMGSY-III लॉन्च किये जाने से पहले समझौता ज्ञापन (MoU) करने को कहा जाएगा ताकि 5 वर्ष की निर्माण रखरखाव अवधि के बाद सड़कों के रखरखाव के लिये पर्याप्त धन उपलब्ध कराया जा सके।",
        "negative": "PMGSY-III चर्चा में क्यों?: हाल ही में मंत्रिमंडल की आर्थिक समिति ने पूरे देश में ग्रामीण सड़क कनेक्टिविटी को और मज़बूत बनाने के लिये मंजूरी दी।"
    }
]


class ConstructionDomainAdapter:
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "models/construction-embedder-adapted"
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir

    def train(self, epochs: int = 4, batch_size: int = 8, lr: float = 2e-5) -> str:
        model = SentenceTransformer(self.base_model_name)
        train_examples = [
            InputExample(texts=[item["query"], item["positive"], item["negative"]])
            for item in TRAINING_CONSTRUCTION_PAIRS
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        os.makedirs(self.output_dir, exist_ok=True)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * epochs * 0.1),
            optimizer_params={"lr": lr},
            show_progress_bar=True
        )
        model.save(self.output_dir)
        return self.output_dir


def load_construction_embedder(use_finetuned: bool = True) -> SentenceTransformer:
    adapted_path = "models/construction-embedder-adapted"
    if use_finetuned:
        if not os.path.exists(adapted_path) or not os.listdir(adapted_path):
            ConstructionDomainAdapter().train(epochs=4)
        return SentenceTransformer(adapted_path)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    adapter = ConstructionDomainAdapter()
    adapter.train(epochs=4)
