This is the repository associated with the paper "Can Word Relatedness Inform Multilingual Lexical Complexity Prediction?" (Michael, Hülsing, Horbach; 2026).

---

## Documentation

- **[AGENTS.md](coding_instructions/AGENTS.md)** - Complete project specification
- **[PROJECT_STATUS.md](coding_instructions/PROJECT_STATUS.md)** - Current status & what to build next

---

## Quick Overview

**Research Question**: Do word relatedness features improve multilingual lexical complexity prediction?

**Dataset**: BEA-ST (German, Spanish, Mandarin Chinese → English)  
**Model**: RoBERTa → LightGBM with 6 feature configurations  
**Validation**: 5-fold outer × 4-fold inner CV  

**Current Status**: ✅ Data pooling complete (22,548 instances)

---

## Implementation Status

**Completed**:
- ✅ Data pooling (`data/pooling.py`)

**Next**: Data preprocessing and fold generation

See [PROJECT_STATUS.md](coding_instructions/PROJECT_STATUS.md) for details.
