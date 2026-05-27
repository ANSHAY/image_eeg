"""Visual Cortex Reconstructor — Streamlit demo app.

Layout (matches the spec ASCII):

  ┌─────────────────────────────┬─────────────────────────────┐
  │  LIVE EEG (128 ch stacked)  │  AI RECONSTRUCTION          │
  │  + frequency spectrum below │  + ground truth + metrics   │
  ├─────────────────────────────┴─────────────────────────────┤
  │  LATENT SPACE (UMAP scatter, fading EEG trail)            │
  └───────────────────────────────────────────────────────────┘

Run:

  # terminal 1 — the virtual EEG rig
  .venv/bin/python -m streaming.lsl_streamer --speed 1.0

  # terminal 2 — the demo app
  .venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch

from app.components.eeg_plot import build_eeg_plot
from app.components.fft_plot import build_fft_plot
from app.components.metrics_panel import MetricsPanel
from app.components.umap_plot import UMAPProjection
from app.theme import inject_theme_css
from generation.retrieval_generator import RetrievalGenerator
from models.eeg_encoder import EEGEncoder
from preprocessing.clip_embeddings import load_image_bank
from preprocessing.signal_processing import Pipeline
from utils.config import load_config
from utils.logging import setup_logging
from utils.seed import set_seed


# ------------------------------------------------------------------ globals

_TRIAL_TRAIL_LEN = 30
_LSL_QUEUE_MAX = 64
_LSL_POLL_TIMEOUT_S = 0.2


# ------------------------------------------------------------------ resource caches

@st.cache_resource(show_spinner="loading config + theme…")
def _load_cfg_cached():
    setup_logging()
    cfg = load_config()
    set_seed(cfg.project.seed)
    return cfg


@st.cache_resource(show_spinner="loading image bank…")
def _load_bank_cached() -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    try:
        return load_image_bank()
    except (FileNotFoundError, RuntimeError):
        return None


@st.cache_resource(show_spinner="building UMAP projection…")
def _build_umap_cached(_cfg):
    bank = _load_bank_cached()
    if bank is None:
        return None
    emb, labels, _ = bank
    return UMAPProjection(_cfg, emb, labels)


@st.cache_resource(show_spinner="initializing encoder…")
def _load_encoder_cached(_cfg):
    """For Phase 6, run an untrained encoder if no checkpoint is on disk.
    Real demo flow will look for the latest run's best.ckpt under cfg.paths.runs."""
    model = EEGEncoder(cfg=_cfg).eval()
    ckpt = _find_latest_checkpoint(_cfg)
    if ckpt is not None:
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("model", state))
            st.toast(f"loaded encoder from {ckpt.relative_to(Path(_cfg.paths.runs))}")
        except Exception as e:
            st.warning(f"checkpoint at {ckpt} failed to load: {e}")
    return model


def _find_latest_checkpoint(cfg) -> Optional[Path]:
    runs_dir = Path(cfg.paths.runs)
    if not runs_dir.is_dir():
        return None
    candidates = list(runs_dir.glob("*/fold_*/best.ckpt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


@st.cache_resource(show_spinner="loading retrieval index…")
def _load_retriever_cached(_cfg):
    try:
        return RetrievalGenerator(cfg=_cfg)
    except FileNotFoundError:
        return None


# ------------------------------------------------------------------ LSL inlet thread

def _start_lsl_thread(cfg, q: queue.Queue, stop_event: threading.Event) -> threading.Thread:
    """Background thread: pull trials + markers from LSL → queue. The main
    Streamlit fragment drains the queue per fragment tick."""

    def loop():
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError:
            return

        eeg_streams = resolve_byprop("name", cfg.streaming.outlet_name, timeout=cfg.streaming.resolve_timeout_s)
        mk_streams = resolve_byprop("name", cfg.streaming.marker_outlet_name, timeout=cfg.streaming.resolve_timeout_s)
        if not eeg_streams or not mk_streams:
            return

        eeg_inlet = StreamInlet(eeg_streams[0], recover=False)
        mk_inlet = StreamInlet(mk_streams[0], recover=False)

        while not stop_event.is_set():
            marker_sample, _ = mk_inlet.pull_sample(timeout=_LSL_POLL_TIMEOUT_S)
            if marker_sample is None:
                continue
            try:
                marker = json.loads(marker_sample[0])
            except (json.JSONDecodeError, IndexError):
                continue
            chunk, _ = eeg_inlet.pull_chunk(
                timeout=cfg.streaming.resolve_timeout_s,
                max_samples=cfg.eeg.trial_length_samples,
            )
            arr = np.asarray(chunk, dtype=np.float32)
            if arr.size == 0:
                continue
            # LSL chunk is (samples, channels) — we want (channels, samples).
            arr = arr.T
            if q.qsize() < _LSL_QUEUE_MAX:
                q.put((marker, arr), block=False)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ------------------------------------------------------------------ inference path

def _process_trial(
    eeg: np.ndarray,
    pipeline: Pipeline,
    encoder: EEGEncoder,
    retriever: Optional[RetrievalGenerator],
) -> tuple[np.ndarray, np.ndarray, Optional[dict], float]:
    """Preprocess → encode → retrieve.  Returns (cleaned_eeg, z, retrieval_result, latency_ms)."""
    t0 = time.perf_counter()
    cleaned = pipeline(eeg)
    with torch.no_grad():
        z = encoder(torch.from_numpy(cleaned).unsqueeze(0)).cpu().numpy()[0]
    ret = retriever.generate(z) if retriever is not None else None
    elapsed = (time.perf_counter() - t0) * 1e3
    return cleaned, z, ret, elapsed


# ------------------------------------------------------------------ app entrypoint

def _init_session_state(cfg) -> None:
    if "lsl_queue" not in st.session_state:
        st.session_state.lsl_queue = queue.Queue(maxsize=_LSL_QUEUE_MAX)
        st.session_state.lsl_stop = threading.Event()
        st.session_state.lsl_thread = _start_lsl_thread(
            cfg, st.session_state.lsl_queue, st.session_state.lsl_stop,
        )
        st.session_state.pipeline = Pipeline(cfg=cfg)
        st.session_state.metrics = MetricsPanel(cfg=cfg)
        st.session_state.history: deque = deque(maxlen=_TRIAL_TRAIL_LEN)
        st.session_state.last_marker = None
        st.session_state.last_retrieval = None
        st.session_state.last_eeg = None


def _drain_one(cfg) -> bool:
    """Pull one (marker, eeg) off the queue, process, update state.
    Returns True if a trial was processed."""
    try:
        marker, eeg = st.session_state.lsl_queue.get_nowait()
    except queue.Empty:
        return False

    if eeg.shape != (cfg.eeg.num_channels, cfg.eeg.trial_length_samples):
        return False

    encoder = _load_encoder_cached(cfg)
    retriever = _load_retriever_cached(cfg)
    cleaned, z, ret, latency_ms = _process_trial(
        eeg, st.session_state.pipeline, encoder, retriever,
    )
    st.session_state.last_eeg = cleaned
    st.session_state.last_marker = marker
    st.session_state.last_retrieval = ret

    true_label = int(marker.get("label", -1))
    pred_label = int(ret["label"]) if ret is not None else -1
    cosine = float(ret["score"]) if ret is not None else 0.0
    correct = (true_label == pred_label) and true_label >= 0
    st.session_state.metrics.record(correct=correct, cosine=cosine, latency_ms=latency_ms)
    st.session_state.history.append((z, true_label, pred_label))
    return True


def main() -> None:
    cfg = _load_cfg_cached()
    st.set_page_config(
        page_title=cfg.ui.strings.app_title,
        layout="wide",
        page_icon="🧠",
    )
    st.markdown(inject_theme_css(cfg), unsafe_allow_html=True)
    st.title(cfg.ui.strings.app_title)

    _init_session_state(cfg)

    # Drain one trial per refresh to avoid blocking the UI.
    _drain_one(cfg)

    bank = _load_bank_cached()
    if bank is None:
        st.info(
            "Image bank not found. Run `python -m preprocessing.run_pipeline` "
            "to build it from the real Spampinato dataset.",
        )

    # --- top row: EEG plot | reconstruction --------------------------------
    left, right = st.columns([1.3, 1.0])
    with left:
        st.markdown(f"### {cfg.ui.strings.panel_eeg}")
        if st.session_state.last_eeg is not None:
            st.plotly_chart(
                build_eeg_plot(st.session_state.last_eeg, cfg=cfg),
                use_container_width=True, key="eeg",
            )
            st.markdown(f"### {cfg.ui.strings.panel_fft}")
            st.plotly_chart(
                build_fft_plot(st.session_state.last_eeg, cfg=cfg),
                use_container_width=True, key="fft",
            )
        else:
            st.markdown(
                f'<div class="vcr-card">{cfg.ui.strings.no_stream}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(f"### {cfg.ui.strings.panel_recon}")
        if st.session_state.last_retrieval is not None:
            ret = st.session_state.last_retrieval
            st.image(ret["image"], use_container_width=True)
            mk = st.session_state.last_marker or {}
            cosine_label = cfg.ui.strings.metric_cosine
            cls_label = cfg.ui.strings.metric_predicted_class
            true_class = mk.get("class_name") or f"label={mk.get('label', '?')}"
            st.markdown(
                f'<div class="vcr-card">'
                f'<div class="vcr-metric-label">{cls_label}</div>'
                f'<div class="vcr-metric-value">{ret["label"]}</div>'
                f'<div class="vcr-metric-label">{cosine_label}</div>'
                f'<div class="vcr-metric-value">{ret["score"]:.3f}</div>'
                f'<div class="vcr-metric-label">{cfg.ui.strings.panel_truth}</div>'
                f'<div>{true_class}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="vcr-card">{cfg.ui.strings.no_stream}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(st.session_state.metrics.render_html(), unsafe_allow_html=True)

    # --- bottom strip: UMAP ------------------------------------------------
    st.markdown(f"### {cfg.ui.strings.panel_umap}")
    umap_proj = _build_umap_cached(cfg)
    if umap_proj is not None and st.session_state.history:
        st.plotly_chart(
            umap_proj.figure(list(st.session_state.history)),
            use_container_width=True, key="umap",
        )
    else:
        st.markdown(
            f'<div class="vcr-card">{cfg.ui.strings.no_stream}</div>',
            unsafe_allow_html=True,
        )

    # Self-refresh tick at the configured cadence.
    time.sleep(cfg.ui.refresh_interval_ms / 1000.0)
    st.rerun()


if __name__ == "__main__":
    main()
