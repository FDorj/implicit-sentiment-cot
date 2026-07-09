# implicit-sentiment-cot

Implementation for the undergraduate thesis:

**Development of a System for Implicit Sentiment Analysis Using Multi-Step Chain-of-Thought Reasoning**

## Project Goal

This project studies implicit sentiment analysis (ISA) on the SCAPT-labeled SemEval 2014 Laptop and Restaurant datasets. It starts from a direct zero-shot prompting baseline, implements THOR-style multi-step reasoning, and extends it with reflective error diagnosis, controller logic, self-consistency, and train-calibrated source selection.

The validated experiments in this repository use Qwen3 8B through Ollama. The code contains a HuggingFace seq2seq runner scaffold, but Flan-T5 has not been tested or evaluated in the current results.

## Final Pipeline

The current final system is:

```text
Direct Qwen3 8B
+ THOR original-ish self-consistency (SC3)
+ Error-Type-Aware Reflection
+ Controller
+ Train-Calibrated Source Selection
```

This final pipeline is assembled from saved prediction files:

```text
results/direct_isa_predictions.csv
        +
results/thor_originalish_sc3_isa_predictions.csv
        |
        v
results/etc_thor_originalish_sc3_isa_predictions.csv
        |
        v
results/etc_thor_originalish_sc3_selected_isa_predictions.csv
```

`experiments/run_final_pipeline.py` does not call the language model again. It validates the saved pipeline chain and regenerates the final result tables.

## Final Results

Main test-only results:

| Method | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: |
| Direct Qwen3 8B | 0.678733 | 0.674075 |
| THOR simplified | 0.599548 | 0.578437 |
| Simple reflection | 0.615385 | 0.606732 |
| ETC standard | 0.660633 | 0.659430 |
| THOR original-ish SC3 | 0.590498 | 0.600690 |
| ETC over original-ish SC3 | 0.660633 | 0.662108 |
| Final selected pipeline | 0.723982 | 0.719204 |

Generated summary files:

- `results/final_results_table.csv`
- `results/final_results_table.md`
- `results/final_pipeline_validation.txt`
- `results/final_qualitative_examples.csv`
- `results/final_qualitative_summary.txt`
- `thesis_notes/final_pipeline_fa.md`
- `thesis_notes/final_project_report_fa.md`
- `thesis_notes/final_presentation_en.html`
- `thesis_notes/final_presentation_fa.html`
- `thesis_notes/final_qualitative_examples_fa.md`
- `thesis_notes/final_defense_summary_fa.md`

Qualitative test-set comparison against Direct:

| Group | Count |
| --- | ---: |
| Final fixes a Direct error | 26 |
| Final introduces an error vs Direct | 6 |
| Both Direct and Final correct | 294 |
| Both Direct and Final wrong | 116 |

## Data Files

Main processed files:

- `data/processed/semeval14_scapt_all_clean.csv`
- `data/processed/semeval14_scapt_isa_only_clean.csv`

The ISA-only clean dataset has 2188 examples:

- train: 1746
- test: 442

## Project Structure

- `prompts/`: prompt templates
- `src/`: core pipeline, controller, evaluator, and final-result helpers
- `experiments/`: runnable experiment and analysis scripts
- `data/processed/`: processed datasets
- `results/`: model outputs, metrics, and final summary tables
- `tests/`: lightweight unit tests for core controller and pipeline logic
- `thesis_notes/`: Persian notes and defense-oriented explanations

## Useful Commands

Run the lightweight tests:

```powershell
python -B -m unittest discover -s tests -v
```

Run a small OpenAI-compatible gateway pilot, for example Gemini through an AI gateway:

```powershell
$env:PROMPT_BACKEND="openai_compatible"
$env:OPENAI_COMPAT_BASE_URL="<gateway-url-ending-with-/v1>"
$env:OPENAI_COMPAT_MODEL="Gemini-2.5-Flash"
$env:OPENAI_COMPAT_API_KEY="<optional-if-the-gateway-url-already-contains-auth>"
$env:OPENAI_COMPAT_MIN_MAX_TOKENS="128"
$env:OPENAI_COMPAT_MAX_RETRIES="6"
$env:OPENAI_COMPAT_RETRY_SLEEP_SECONDS="10"
$env:OPENAI_COMPAT_EMPTY_LENGTH_RETRIES="2"
$env:EXPERIMENT_ID="gemini_25_flash"
$env:DATA_SPLIT="test"
$env:DEBUG_N="all"
$env:RESUME_DIRECT="1"
$env:SAVE_EVERY="10"
python -B experiments/run_direct.py
```

The scripts read environment variables. Keep gateway URLs and API keys in your shell environment, or in a local ignored `.env` file only if your shell/tool loads it. Do not commit credentials.
For Gemini gateways that spend output budget on reasoning tokens, keep `OPENAI_COMPAT_MIN_MAX_TOKENS` high enough to avoid empty `finish_reason=length` responses.

Regenerate the final result table and validate the final pipeline chain:

```powershell
python -B experiments/run_final_pipeline.py
```

Extract qualitative examples for the report/defense:

```powershell
python -B experiments/extract_qualitative_examples.py
```

## Implemented Components

- SCAPT/SemEval preprocessing and ISA-only dataset construction
- Direct zero-shot prompting baseline
- THOR-style multi-step prompting
- Simple reflection over THOR traces
- Error-type-aware reflection with structured diagnostic output
- Rule-based controller
- ETC-ISA pipeline
- Policy ablation
- THOR original-ish prompting
- THOR self-consistency
- Train-calibrated source selection
- Final pipeline validation and result summarization
- Qualitative example extraction for report/defense analysis
- Lightweight unit tests for core logic

## Not Implemented In Current Results

- No language model fine-tuning has been performed.
- Flan-T5/HuggingFace has not been tested or evaluated.
- Gemini/OpenAI-compatible gateway results are not part of the current validated result files.
- `run_final_pipeline.py` summarizes saved outputs; it does not rerun Qwen/Ollama inference.

## Notes

The official thesis title must remain unchanged.
