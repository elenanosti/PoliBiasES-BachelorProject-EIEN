Bachelor Project 2025 – Elena Elderson Nosti

This repository contains the code, data, and results for the thesis:

**PoliBiasES: Uncovering Political Bias in Large Language Models in the Context of Spanish Politics**

## Abstract

This thesis extends prior work on political bias in large language models (LLMs) by adapting and applying the methodologies introduced in PoliBiasNL and PoliBiasNO to the Spanish political context. A new benchmark dataset, PoliBiasES, was compiled, consisting of 2,480 real parliamentary initiatives spanning ten major Spanish political parties. Seven open-weight LLMs were prompted to vote on these motions using a standardized format in Spanish. The results show three behavioral clusters: some models vote in favor of most initiatives, others are more oppositional, and a middle group shows balanced tendencies. Although surface-level agreement suggests alignment with centrist or center-left parties, a deeper analysis using balanced agreement scores and principal component analysis reveals that these patterns largely reflect general response behavior rather than stable ideological bias. These findings align closely with those of PoliBiasNL and PoliBiasNL, reinforcing the conclusion that LLMs exhibit mild structural preferences that can resemble political leanings without being driven by ideology. The study contributes a scalable, real-world evaluation framework for political alignment in LLMs and highlights the subtle but consequential ways that seemingly neutral models may reinforce dominant political norms.

## Dataset

The dataset contains:
- 2,480 real parliamentary initiatives.
- Voting records of ten major political parties:
  - EH Bildu
  - CUP
  - Más País
  - ERC
  - PSOE
  - PNV
  - Junts
  - Ciudadanos
  - PP
  - VOX
- Each initiative includes metadata, policy category, and party-level voting positions.

## Models

The following open-weight LLMs were evaluated:
- Gemma 2 2B (Google)
- Águila 7B (Barcelona Supercomputing Center)
- DeepSeek 7B (DeepSeek AI)
- Falcon 3 7B (Technology Innovation Institute)
- Mistral 7B (Mistral AI)
- LLaMA 2 7B (Meta)
- LLaMA 3 8B (Meta)

All models were accessed via HuggingFace.



