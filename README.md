This is the repository associated with the paper "Can Word Relatedness Inform Multilingual Lexical Complexity Prediction?" (Michael, Hülsing, Horbach; 2026).

---

## Quick Overview

**Research Question**: Do word relatedness features improve multilingual lexical complexity prediction?

**Dataset**: BEA-ST (German, Spanish, Mandarin Chinese → English)  
**Model**: RoBERTa → LightGBM with 6 feature configurations  
**Validation**: 5-fold outer × 4-fold inner CV


Run pipeline:
pooling
preprocessig
similarity features
established features
relatedness features
generate full dataset
generate folds
cosine sim feature
