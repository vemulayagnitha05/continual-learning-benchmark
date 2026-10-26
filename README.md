# Continual Learning — Real MNIST / Fashion-MNIST

A deployable class-incremental learning demo using **real benchmark data only**.

## Data policy

This project does **not** generate synthetic training data. Training data comes from the real MNIST or Fashion-MNIST benchmark datasets downloaded by `torchvision`.

If the real dataset cannot be downloaded or loaded, the application fails rather than substituting synthetic data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL shown in the terminal.

## Run with Docker

```bash
docker build -t continual-learning-real .
docker run --rm -p 8501:8501 continual-learning-real
```

Then open `http://localhost:8501`.

## Benchmark

The original experiment remains available:

```bash
python continual.py --dataset mnist --seeds 5
python continual.py --dataset fashion --seeds 5
```

The benchmark uses one single-pass stream per task and compares naive fine-tuning, reservoir replay, and a joint-stream upper bound.

## Deployment

The included Docker image is suitable for container-based deployment platforms. The first run downloads the selected real dataset, so the deployment needs outbound network access and enough disk/RAM for the dataset and PyTorch.
