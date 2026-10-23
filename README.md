# MHEB
Multi-cutoff Historical Event Benchmark

## Files

* `historical.json` — Multi-cutoff Historical Event dataset
* `test_prompt.py` — Prompt definitions
* `test.py` — Runs the experiments
* `metrics.py` — Calculates and displays accuracy from the experimental results

## Dataset Format

Each entry in `historical.json` contains:

```json
{
  "Event year": "1265",
  "Answer before cutoff": "Yes",
  "Answer after cutoff": "No",
  "Question": "Has a historical event called Byzantine–Ottoman Wars ever occurred?"
}
```

The experiment uses the following fields:

* `Event year` — Used to calculate the cutoff year
* `Question` — Input question
* `Answer before cutoff` — Reference answer before the event is within the cutoff
* `Answer after cutoff` — Reference answer after the event is within the cutoff

Other metadata fields in `historical.json` are not used by the experiment.

## Setup

First, add your API keys to `api_keys.json`.

```json
{
  "together": "YOUR_TOGETHER_API_KEY",
  "openai": "YOUR_OPENAI_API_KEY"
}
```

Install the required packages:

```bash
pip install openai together tqdm
```

## Execution

Run the scripts in the following order.

### 1. Run the experiments

```bash
python test.py
```

This runs the models on `historical.json` and saves the experimental results to:

```text
results/historical_results.json
```

Intermediate results are saved under:

```text
results/intermediate/
```

### 2. Create the summary

After `test.py` finishes, run:

```bash
python metrics.py
```

This calculates the accuracy for each model and cutoff year and saves the summaries to:

```text
results/summary/
```

## Cutoff Years

The relative cutoff is defined as:

```text
relative_cutoff = cutoff_year - event_year
```

The default setting is:

```python
RELATIVE_CUTOFF_YEARS = range(-10, 1)
```

which evaluates relative cutoff years `-10`, `-9`, ..., `-1`, and `0`.

## Citation
If you find this work useful, please cite:
```bibtex
@article{asai2026can,
  title={Can LLMs Be Constrained to the Past? Improving Knowledge Cutoff through Recall-Based Prompting},
  author={Asai, Michiro and Lin, Ailiang and Kishimoto, Yu and Obi, Takao and Kosugi, Satoshi and Funakoshi, Kotaro and Okumura, Manabu},
  journal={arXiv preprint arXiv:2606.05804},
  year={2026}
}
```
