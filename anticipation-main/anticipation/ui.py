import tempfile

import streamlit as st
import torch
from transformers import AutoModelForCausalLM

import pretty_midi
import matplotlib.pyplot as plt

from anticipation.sample import generate
from anticipation.convert import midi_to_events, events_to_midi


@st.cache_resource
def load_model(checkpoint: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model.to(device)


def plot_piano_roll(mid: str, fs: int = 50):
    """Return a matplotlib figure showing a piano roll for the given MIDI."""
    pm = pretty_midi.PrettyMIDI(mid)
    roll = pm.get_piano_roll(fs=fs)
    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [0, roll.shape[1] / fs, 0, 128]
    ax.imshow(roll, aspect="auto", origin="lower", cmap="gray_r", extent=extent)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch")
    return fig


st.title("Anticipatory Music Transformer")

model_ckpt = st.text_input(
    "Model checkpoint", value="stanford-crfm/music-medium-800k"
)

if st.button("Load model"):
    model = load_model(model_ckpt)
    st.session_state["model"] = model

if "model" in st.session_state:
    uploaded = st.file_uploader("Prompt MIDI", type=["mid", "midi"])
    inputs = []
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".mid") as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            inputs = midi_to_events(tmp.name)

    start = st.number_input("Start time (s)", value=0)
    end = st.number_input("End time (s)", value=10)
    top_p = st.slider("top_p", 0.0, 1.0, value=0.98)

    if st.button("Generate"):
        events = generate(st.session_state["model"], start, end, inputs=inputs, top_p=top_p)
        midi = events_to_midi(events)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as out:
            midi.save(out.name)
            out.flush()
            st.pyplot(plot_piano_roll(out.name))
            st.audio(out.name)
            st.download_button(
                label="Download MIDI",
                data=open(out.name, "rb").read(),
                file_name="generated.mid",
                mime="application/octet-stream",
            )

