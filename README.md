This is the repository associated with the paper "Etymology Can Inform Lexical Complexity Prediction" (Michael et al., 2026). In: Proceedings of the 15th Workshop on Natural Language Processing for Computer Assisted Language Learning (NLP4CALL 2026).

---

## Quick Overview

**Research Question**: Do word etymology features improve multilingual lexical complexity prediction?

**Dataset**: 2026 BEA shared-task dataset on vocabulary difficulty prediction for English learners (German, Spanish, Mandarin Chinese → English)  
**Model**: RoBERTa → LightGBM
**Validation**: 5-fold outer × 4-fold inner CV


To reproduce our results, please follow the steps below in the specified order.

Install:
pip install -r requirements.txt

Run:
python pooling.py
python preprocessig.py
python generate_similarity_features.py
python generate_established_features.py
python generate_relatedness_features.py
python fill_na.py
python generate_full_dataset.py
python generate_folds.py
python generate_cosine_sim_feature.py
