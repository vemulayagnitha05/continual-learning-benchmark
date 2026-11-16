import io
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from continual import TASKS, Net, load, evaluate, Reservoir

ROOT = Path(__file__).parent
METHOD_ORDER = ["naive", "replay-200", "replay-1000", "joint"]
METHOD_LABELS = {
    "naive": "Naive Fine-Tuning",
    "replay-200": "Experience Replay (200)",
    "replay-1000": "Experience Replay (1,000)",
    "joint": "Joint Training (upper reference)",
}

st.set_page_config(
    page_title="Continual Learning Benchmark",
    page_icon="🧠",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def load_results():
    frames = []
    for filename in ("results_mnist.csv", "results_fashion.csv"):
        path = ROOT / filename
        if path.exists():
            frame = pd.read_csv(path)
            required = {"dataset", "method", "seed", "final_acc", "forgetting"}
            if required.issubset(frame.columns):
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["dataset", "method", "seed", "final_acc", "forgetting"])
    data = pd.concat(frames, ignore_index=True)
    data["final_acc_pct"] = data["final_acc"] * 100
    data["forgetting_pct"] = data["forgetting"] * 100
    return data

@st.cache_data(show_spinner=False)
def get_dataset(dataset):
    return load(dataset)

@st.cache_resource(show_spinner=False)
def train_demo(dataset, method, buffer_size, seed):
    train, test = get_dataset(dataset)
    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = Net()
    X, Y = train
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    buf = Reservoir(buffer_size, seed) if method == "replay" else None

    def step(xb, yb):
        current_x, current_y = xb, yb
        if buf is not None and buf.x:
            xr, yr = buf.sample(32)
            xb, yb = torch.cat([current_x, xr]), torch.cat([current_y, yr])
        opt.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        opt.step()
        if buf is not None:
            buf.add(current_x, current_y)

    if method == "joint":
        permutation = torch.randperm(len(X))
        for start in range(0, len(X), 32):
            indices = permutation[start:start + 32]
            step(X[indices], Y[indices])
    else:
        for first, second in TASKS:
            indices = torch.nonzero((Y == first) | (Y == second)).squeeze(1)
            indices = indices[torch.randperm(len(indices))]
            for start in range(0, len(indices), 32):
                batch = indices[start:start + 32]
                step(X[batch], Y[batch])

    accuracies = evaluate(model, test)
    model.eval()
    return model, accuracies, time.time() - started

def show_results():
    results = load_results()
    st.subheader("Benchmark results")
    if results.empty:
        st.warning("No benchmark CSV files found. Run the benchmark scripts and place results_mnist.csv and results_fashion.csv beside app.py.")
        return

    dataset = st.selectbox("Benchmark dataset", ["mnist", "fashion"], format_func=lambda x: "MNIST" if x == "mnist" else "Fashion-MNIST", key="results_dataset")
    subset = results[results["dataset"] == dataset]
    if subset.empty:
        st.info("No saved results for this dataset yet.")
        return

    summary = (subset.groupby("method")
               .agg(accuracy_mean=("final_acc_pct", "mean"),
                    accuracy_std=("final_acc_pct", "std"),
                    forgetting_mean=("forgetting_pct", "mean"),
                    forgetting_std=("forgetting_pct", "std"),
                    seeds=("seed", "nunique"))
               .reindex(METHOD_ORDER))
    summary["accuracy_std"] = summary["accuracy_std"].fillna(0)
    summary["forgetting_std"] = summary["forgetting_std"].fillna(0)

    best = summary["accuracy_mean"].idxmax()
    m1, m2, m3 = st.columns(3)
    m1.metric("Best mean final accuracy", f"{summary.loc[best, 'accuracy_mean']:.1f}%", METHOD_LABELS[best])
    m2.metric("Replay-1,000 mean accuracy", f"{summary.loc['replay-1000', 'accuracy_mean']:.1f}%" if "replay-1000" in summary.index else "N/A")
    m3.metric("Completed seeds per method", f"{int(summary['seeds'].max())}")

    chart = summary[["accuracy_mean"]].rename(columns={"accuracy_mean": "Mean accuracy (%)"})
    st.markdown("#### Final accuracy")
    st.bar_chart(chart, y="Mean accuracy (%)", horizontal=True)
    forgetting_chart = summary[["forgetting_mean"]].rename(columns={"forgetting_mean": "Mean forgetting (%)"})
    st.markdown("#### Catastrophic forgetting")
    st.bar_chart(forgetting_chart, y="Mean forgetting (%)", horizontal=True)

    display = summary.reset_index().rename(columns={
        "method": "Method", "accuracy_mean": "Accuracy mean (%)",
        "accuracy_std": "Accuracy std (%)", "forgetting_mean": "Forgetting mean (%)",
        "forgetting_std": "Forgetting std (%)", "seeds": "Seeds"
    })
    display["Method"] = display["Method"].map(METHOD_LABELS)
    numeric = ["Accuracy mean (%)", "Accuracy std (%)", "Forgetting mean (%)", "Forgetting std (%)"]
    display[numeric] = display[numeric].round(2)
    st.markdown("#### Summary (mean ± standard deviation)")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Metrics are calculated from the saved per-seed CSV records. Standard deviation uses sample SD (pandas ddof=1).")

    st.markdown("#### Individual runs")
    raw = subset[["method", "seed", "final_acc_pct", "forgetting_pct"]].copy()
    raw["method"] = raw["method"].map(METHOD_LABELS)
    raw = raw.rename(columns={"method": "Method", "seed": "Seed", "final_acc_pct": "Final accuracy (%)", "forgetting_pct": "Forgetting (%)"})
    raw[["Final accuracy (%)", "Forgetting (%)"]] = raw[["Final accuracy (%)", "Forgetting (%)"]].round(2)
    st.dataframe(raw, use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed dataset results (CSV)",
        subset.drop(columns=["final_acc_pct", "forgetting_pct"]).to_csv(index=False).encode("utf-8"),
        file_name=f"continual_learning_{dataset}_results.csv",
        mime="text/csv",
    )

st.title("🧠 Continual Learning Benchmark")
st.caption("Class-incremental learning on the official MNIST and Fashion-MNIST datasets.")
st.info("Research benchmark: 2 datasets · 4 methods · 5 random seeds per dataset. Results below are loaded from the saved experiment CSV files.")

overview, train_tab, methodology = st.tabs(["📊 Results dashboard", "🧪 Train & predict", "📘 Methodology"])

with overview:
    show_results()

with train_tab:
    st.subheader("Train a model on real benchmark data")
    st.caption("This interactive demo trains one selected configuration. It is separate from the multi-seed benchmark results shown in the dashboard.")
    left, right = st.columns([1, 2])
    with left:
        dataset = st.selectbox("Dataset", ["mnist", "fashion"], format_func=lambda x: "MNIST" if x == "mnist" else "Fashion-MNIST", key="train_dataset")
        method_label = st.selectbox("Training method", ["naive", "replay-200", "replay-1000", "joint"], key="train_method")
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0, step=1, key="train_seed")
        train_button = st.button("Train model", type="primary", use_container_width=True)
    with right:
        st.markdown("**Data policy**")
        st.write("The application downloads the official MNIST or Fashion-MNIST dataset through torchvision. It does not generate synthetic training samples.")
        st.write("Training follows the project's single-pass stream setup, with five sequential tasks containing two classes each (except the joint upper-reference run).")

    method = "replay" if method_label.startswith("replay") else method_label
    buffer_size = int(method_label.split("-")[1]) if method_label.startswith("replay") else 0
    if train_button:
        with st.spinner(f"Loading real {dataset.upper()} data and training..."):
            model, accuracies, elapsed = train_demo(dataset, method, buffer_size, int(seed))
        st.session_state["model"] = model
        st.session_state["trained_dataset"] = dataset
        st.session_state["trained_method"] = method_label
        st.session_state["accuracies"] = accuracies
        st.session_state["elapsed"] = elapsed
        st.success("Training completed.")

    if "model" in st.session_state:
        c1, c2, c3 = st.columns(3)
        c1.metric("Average task accuracy", f"{np.mean(st.session_state['accuracies']):.1%}")
        c2.metric("Training time", f"{st.session_state['elapsed']:.1f}s")
        c3.metric("Trained configuration", METHOD_LABELS.get(st.session_state["trained_method"], st.session_state["trained_method"]))
        st.markdown("#### Upload an image for prediction")
        uploaded = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = Image.open(io.BytesIO(uploaded.read())).convert("L")
            image = ImageOps.invert(image)
            image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
            tensor = torch.tensor(np.asarray(image), dtype=torch.float32) / 255.0
            tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(st.session_state["model"](tensor), dim=1)[0]
                prediction = int(probabilities.argmax())
            st.image(image, caption=f"Predicted class: {prediction}", width=180)
            st.metric("Predicted class", str(prediction))
            probability_frame = pd.DataFrame({"Class": [str(i) for i in range(10)], "Probability": probabilities.numpy()})
            st.bar_chart(probability_frame.set_index("Class"))
            st.caption("The model was trained on the selected real benchmark dataset. For best results, upload a centered, grayscale-like image resembling the dataset format.")

with methodology:
    st.subheader("Experimental setup")
    st.markdown("""
- **Datasets:** Official MNIST and Fashion-MNIST via `torchvision`.
- **Task structure:** Five sequential tasks, each containing two classes: (0,1), (2,3), (4,5), (6,7), and (8,9).
- **Learning setting:** One shared 10-class output head; each task stream is processed once with batch size 32.
- **Naive fine-tuning:** Learns from the current stream without an episodic replay buffer.
- **Experience replay:** Uses reservoir sampling with a buffer capacity of 200 or 1,000 examples.
- **Joint training:** One-pass shuffled stream over all training examples; included as an upper-reference baseline, not as a memory-constrained continual-learning method.
- **Metrics:** Average final task accuracy and average forgetting.
- **Repetitions:** Five random seeds (0–4) for each method and dataset.
""")
    st.markdown("#### Metric definitions")
    st.latex(r"\\text{Final accuracy} = \\frac{1}{T}\\sum_{j=1}^{T} A_{T,j}")
    st.write("Average final accuracy is the mean accuracy over all five tasks after the learning sequence.")
    st.latex(r"\\text{Forgetting} = \\frac{1}{T-1}\\sum_{j=1}^{T-1}\\left(\\max_{t<T} A_{t,j} - A_{T,j}\\right)")
    st.write("Average forgetting measures the loss in performance on earlier tasks between their best observed accuracy and their final accuracy.")
    st.warning("The saved benchmark results describe this implementation and experimental setup. They should not be interpreted as a comparison against every continual-learning algorithm or as a universal performance guarantee.")
