This is the repository associated with the paper "Etymology Can Inform Lexical Complexity Prediction" (Michael, Hülsing, Horbach; 2026). In: Proceedings of the 15th Workshop on Natural Language Processing for Computer Assisted Language Learning (NLP4CALL 2026).

---

## Quick Overview

**Research Questions**:  
- What is the predictive potential of etymology features in LCP, both alone and combined with existing features?  
- To what extent do etymology features contribute information distinct from features previously employed in LCP?  
- Is there a specific NLD range where etymology features are most effective, and does it vary across L1s?
- To what extent do etymology features, alone or combined with existing features, further improve a transformer baseline in LCP?  

**Dataset**: 2026 BEA shared-task dataset on vocabulary difficulty prediction for English learners (German, Spanish, Mandarin → English)  
**Models**: (XLM-RoBERTa →) LightGBM  
**Validation**: 5-fold outer × 4-fold inner CV  
**Contribution**: 9 GPT-5.6-Luna-generated etymology features derived from Wiktionary etymologies, evaluated alongside surface-similarity and common lexical features in a stacked RoBERTa-LightGBM architecture.

## Findings

## Abstract

In Lexical Complexity Prediction (LCP) for second-language learners, cross-linguistic cognacy and borrowing relationships are commonly approximated using surface similarity. However, historical sound changes and orthographic differences can obscure etymological relationships that still facilitate vocabulary production. We investigate whether etymology-based features improve LCP for learners of English with German, Spanish, or Mandarin as source languages, using the 2026 BEA shared-task dataset. We expect etymological information to be most informative for German and Spanish, where historical relationships with English are frequent but often obscured by divergence. 
Mining Wiktionary etymologies, we derive nine etymology features for source-target word pairs, evaluated alongside surface-similarity and common lexical features in a stacked RoBERTa-LightGBM architecture. Across most configurations, adding etymology information significantly improves prediction performance for all three languages. For German and Spanish, the clearest contribution occurs for word pairs with normalized Levenshtein distances (NLDs) between $0.3$ and $0.7$, where surface similarity becomes less reliable, but common lexical features are not yet dominant. For Mandarin, etymology features remain useful, though less localized across NLD ranges. Our findings indicate that etymology effectively complements standard LCP approaches when surface similarity becomes less informative.

## Reproduction
To reproduce our results, please follow the steps below in the specified order.

Install:  
pip install -r requirements.txt

### Data Preparation and Feature Generation

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

### XLM-RoBERTa Training

### LightGBM Experiments


## Citation

BibTex:  
@inproceedings{michael2026etymology,  
  title     = {Etymology Can Inform Lexical Complexity Prediction},  
  author    = {Michael, Noah-Manuel and Hülsing, Anna and Horbach, Andrea},  
  booktitle = {Proceedings of the 15th Workshop on NLP for Computer-Assisted Language Learning (NLP4CALL)},  
  year      = {2026},  
  publisher = {LiU Electronic Press},  
  note      = {To appear},  
}
