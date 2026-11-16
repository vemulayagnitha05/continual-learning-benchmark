# Continual Learning Benchmark

A reproducible class-incremental learning experiment using the official **MNIST** and **Fashion-MNIST** datasets.

The repository includes a Streamlit dashboard for inspecting saved benchmark results, training a selected configuration, and trying image predictions.

## Research snapshot

The committed CSV files contain **40 runs**: two datasets × four methods × five random seeds (0–4).

| Dataset | Method | Final accuracy (mean ± SD) | Forgetting (mean ± SD) |
|---|---|---:|---:|
| MNIST | Naive Fine-Tuning | 15.7% ± 5.4% | 94.9% ± 10.7% |
| MNIST | Replay-200 | 87.7% ± 1.5% | 14.8% ± 1.9% |
| MNIST | Replay-1,000 | 95.2% ± 0.6% | 5.1% ± 0.8% |
| MNIST | Joint Training (upper reference) | 98.3% ± 0.2% | 0.0% ± 0.0% |
| Fashion-MNIST | Naive Fine-Tuning | 19.9% ± 0.0% | 98.9% ± 0.1% |
| Fashion-MNIST | Replay-200 | 72.5% ± 2.4% | 31.6% ± 3.0% |
| Fashion-MNIST | Replay-1,000 | 81.7% ± 2.0% | 18.9% ± 3.0% |
| Fashion-MNIST | Joint Training (upper reference) | 87.6% ± 0.6% | 0.0% ± 0.0% |

Values are calculated from the per-seed CSV files. Accuracy and forgetting are shown as percentages; standard deviation is the sample standard deviation across five seeds. Small differences may occur due to rounding.

## Key observations

- Replay methods substantially outperform naive fine-tuning in these experiments.
- Replay-1,000 performs better than Replay-200 on both datasets, consistent with the larger memory buffer retaining more examples.
- Naive fine-tuning exhibits severe forgetting in this class-incremental setting.
- Joint training is included as an **upper-reference baseline**. It sees a shuffled stream containing all training examples and is not subject to the same sequential memory constraint as the continual-learning methods.

## Experimental methodology

- **Datasets:** Official MNIST and Fashion-MNIST downloaded through `torchvision`; no synthetic samples are generated.
- **Tasks:** Five sequential two-class tasks: (0,1), (2,3), (4,5), (6,7), and (8,9).
- **Model:** CNN with a shared 10-class output head.
- **Stream:** Each task is processed once, using batch size 32.
- **Methods:** Naive fine-tuning, reservoir replay with capacities 200 and 1,000, and joint training as an upper reference.
- **Seeds:** 0–4 for each method and dataset.
- **Metrics:** Average final accuracy across tasks and average forgetting.

Average final accuracy is the mean accuracy over all five tasks after the learning sequence. Average forgetting measures the drop from the best observed accuracy on each earlier task to its final accuracy.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads `results_mnist.csv` and `results_fashion.csv` from the repository directory. The **Train & predict** tab lets you train one selected configuration and try an uploaded image.

## Reproduce the benchmark

```bash
python continual.py --dataset mnist --seeds 5
python continual.py --dataset fashion --seeds 5
```

These commands download the official datasets if they are not already present and regenerate the corresponding CSV result file.

## Run with Docker

```bash
docker build -t continual-learning-benchmark .
docker run --rm -p 8501:8501 continual-learning-benchmark
```

Open `http://localhost:8501`.

The image needs outbound network access on first run to download MNIST/Fashion-MNIST. Ensure the deployment platform has sufficient memory and writable disk space for PyTorch and the datasets.

## Repository contents

- `continual.py` — benchmark implementation and experiment runner.
- `app.py` — Streamlit results dashboard, training demo, and image prediction.
- `results_mnist.csv` — per-seed MNIST results.
- `results_fashion.csv` — per-seed Fashion-MNIST results.
- `Dockerfile` — container configuration.

## Limitations

This is a compact educational/research benchmark, not a comprehensive evaluation of continual-learning methods. Results depend on the model, optimizer, task order, single-pass protocol, replay implementation, and random seeds. The joint baseline is not directly equivalent to a memory-constrained continual learner. Image upload prediction is a demonstration and works best with centered images resembling MNIST-style inputs.

## Data policy

Training data is obtained from the official MNIST and Fashion-MNIST benchmark datasets through `torchvision`. The project does not generate synthetic training samples.
