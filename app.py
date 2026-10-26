import io
import time
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from continual import TASKS, Net, load, run, evaluate

st.set_page_config(page_title="Continual Learning — Real MNIST", page_icon="🧠", layout="wide")

st.title("🧠 Continual Learning on Real MNIST")
st.caption("Class-incremental learning demo using the official MNIST/Fashion-MNIST datasets. No synthetic data is generated or accepted.")

@st.cache_data(show_spinner=False)
def get_dataset(dataset):
    # torchvision downloads the real benchmark dataset from its configured source.
    return load(dataset)

@st.cache_resource(show_spinner=False)
def train_demo(dataset, method, buffer_size, seed):
    train, test = get_dataset(dataset)
    started = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net()
    # The run() function intentionally returns only metrics. For the demo we train a
    # prediction model using the same single-pass method implementation below.
    X, Y = train
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    import random
    random.seed(seed)
    from continual import Reservoir
    buf = Reservoir(buffer_size, seed) if method == "replay" else None

    def step(xb, yb):
        if buf is not None and buf.x:
            xr, yr = buf.sample(32)
            xb2, yb2 = torch.cat([xb, xr]), torch.cat([yb, yr])
        else:
            xb2, yb2 = xb, yb
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(xb2), yb2).backward()
        opt.step()
        if buf is not None:
            buf.add(xb, yb)

    if method == "joint":
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i+32]
            step(X[idx], Y[idx])
    else:
        for a, b in TASKS:
            idx = torch.nonzero((Y == a) | (Y == b)).squeeze(1)
            idx = idx[torch.randperm(len(idx))]
            for i in range(0, len(idx), 32):
                j = idx[i:i+32]
                step(X[j], Y[j])

    accs = evaluate(model, test)
    model.eval()
    return model, accs, time.time() - started

with st.sidebar:
    st.header("Real-data settings")
    dataset = st.selectbox("Dataset", ["mnist", "fashion"], index=0)
    method_label = st.selectbox("Training method", ["naive", "replay-200", "replay-1000", "joint"])
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)
    train_button = st.button("Train on real data", type="primary", use_container_width=True)

    st.divider()
    st.success("REAL DATA ONLY")
    st.write("The app downloads MNIST/Fashion-MNIST through torchvision. It does not create synthetic samples.")

method = "replay" if method_label.startswith("replay") else method_label
buffer_size = int(method_label.split("-")[1]) if method_label.startswith("replay") else 0

if train_button:
    with st.spinner(f"Downloading/loading real {dataset.upper()} data and training once through the stream…"):
        model, accs, elapsed = train_demo(dataset, method, buffer_size, int(seed))
    st.session_state["model"] = model
    st.session_state["dataset"] = dataset
    st.session_state["accs"] = accs
    st.session_state["elapsed"] = elapsed

if "model" not in st.session_state:
    st.info("Choose a dataset/method and click **Train on real data** to start.")
else:
    accs = st.session_state["accs"]
    c1, c2 = st.columns(2)
    c1.metric("Average final accuracy", f"{np.mean(accs):.1%}")
    c2.metric("Training time", f"{st.session_state['elapsed']:.0f}s")

    st.subheader("Upload an image for prediction")
    uploaded = st.file_uploader("PNG/JPG image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read())).convert("L")
        image = ImageOps.invert(image)
        image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
        x = torch.tensor(np.asarray(image), dtype=torch.float32) / 255.0
        x = ((x - 0.5) / 0.5).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = st.session_state["model"](x)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(probs.argmax())
        st.image(image, caption=f"Prediction: {pred}", width=180)
        st.metric("Predicted class", str(pred))
        st.write({str(i): round(float(p), 4) for i, p in enumerate(probs)})

    st.caption("Prediction model was trained only from the real benchmark dataset selected above.")
