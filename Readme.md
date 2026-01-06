This repository constructs PII-related evaluation data and studies memorization behaviors of large language models, including **verbatim**, **associative**, **extractable memorization**, and **membership inference attacks**.

------

## Pipeline Overview

### 1. Raw PII extraction

Extract text samples containing PII (phone, email, URL) from a large corpus.

Run:

```
python dataset_process/extra_raw_pii_from_mc4.py
```

------

### 2. PII window extraction

Convert raw PII samples into windowed segments suitable for structured processing.

Run:

```
python dataset_process/pii/extra_double_pii_window.py
```

------

### 3. NER-based PII extraction

Extract structured PII fields from windowed text using NER.

Run one or both:

```
python dataset_process/ner/davalan_ner_pii.py
python dataset_process/ner/qwen_ner_pii.py
```

------

## Memorization Evaluation

### A. Verbatim memorization

Evaluate whether the model reproduces PII verbatim and compute target log-likelihood.

Run:

```
python verbatim_mem.py
```

------

### B. Associative memorization

Probe whether the model can infer PII from related attributes (e.g., name → email).

Run:

```
python asso_mem.py
```

This includes:

- Generation-based hit evaluation
- Target log-likelihood analysis
- Optional language-specific prompt templates

please check the template at templates/

------

### C. Extractable memorization

Assess whether PII can be systematically extracted from model outputs.

Run:

```
python extractable_mem.py
```
please check the template at templates/
------

## Membership Inference Attack

Membership inference experiments are conducted using the **mimir** framework.

Repository:

```
https://github.com/iamgroot42/mimir
```

Clone into this project:

```
git clone https://github.com/iamgroot42/mimir mimir
```

Refer to mimir’s documentation for configuration and execution details.