# implicit-sentiment-cot

Implementation for the undergraduate thesis:

**Development of a System for Implicit Sentiment Analysis Using Multi-Step Chain-of-Thought Reasoning**

## Project Goal
This project studies implicit sentiment analysis (ISA) on the SemEval 2014 Laptop and Restaurant datasets, following the THOR setting and extending it with error-type-aware reflective reasoning and controller logic.

## Current Progress
- Parsed SCAPT-labeled SemEval14 XML files
- Built processed datasets
- Built ISA-only subsets
- Built clean datasets without conflict / unlabeled implicit flags

## Data Files
Main processed files:

- `data/processed/semeval14_scapt_all_clean.csv`
- `data/processed/semeval14_scapt_isa_only_clean.csv`

## Planned Baselines
1. Direct Prompting
2. THOR
3. THOR + Simple Reflection
4. THOR + Error-Type Aware Reflection + Controller Logic

## Project Structure
- `prompts/` prompt templates
- `src/` core pipeline code
- `experiments/` runnable experiment scripts
- `data/processed/` processed datasets
- `results/` model outputs and evaluation results

## Notes
The official thesis title must remain unchanged.
