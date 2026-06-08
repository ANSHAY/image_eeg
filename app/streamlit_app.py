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

# Ensure the project root is in sys.path so 'app' and 'generation' modules resolve
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import torch
from PIL import Image

from app.components.eeg_plot import build_eeg_plot
from app.components.fft_plot import build_fft_plot
from app.components.metrics_panel import MetricsPanel
from app.components.umap_plot import UMAPProjection
from app.display_packet import DisplayPacket
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
            print(f"loaded encoder from {ckpt.relative_to(Path(_cfg.paths.runs))}")
        except Exception as e:
            print(f"checkpoint at {ckpt} failed to load: {e}")
    return model


def _find_latest_checkpoint(cfg) -> Optional[Path]:
    import argparse
    import sys
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    args, _ = p.parse_known_args(sys.argv[1:])
    if args.ckpt:
        return Path(args.ckpt)

    runs_dir = Path(cfg.paths.runs)
    if not runs_dir.is_dir():
        return None

    final_candidates = list(runs_dir.glob("*/final/last.ckpt"))
    if final_candidates:
        return max(final_candidates, key=lambda p: p.stat().st_mtime)

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


MAGIC_SYNC_VAL = 999999.0

# ------------------------------------------------------------------ LSL inlet thread

def _start_lsl_thread(cfg, q: queue.Queue, stop_event: threading.Event) -> threading.Thread:
    """Background thread: reads EEG and markers and pairs them flawlessly
    using an explicit sequence ID embedded in the streams.

    Protocol: 
    - Streamer sends a sync sample: [MAGIC_SYNC_VAL, seq_id, 0, 0...]
    - Streamer then sends 500 EEG samples.
    - Streamer sends a JSON marker containing `"seq": seq_id`.
    This receiver pulls both, buffers them by `seq_id`, and only pushes
    a (marker, eeg) pair to the UI queue when both components arrive.
    """

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

        # Flush any garbage/old data in the buffers before we start
        time.sleep(0.5)
        eeg_inlet.pull_chunk(timeout=0.0, max_samples=81920)
        mk_inlet.pull_chunk(timeout=0.0)

        required = cfg.eeg.trial_length_samples
        pending_eeg = {}
        pending_markers = {}
        
        collecting_seq = None
        collected_samples = []

        while not stop_event.is_set():
            # 1. Pull EEG chunk
            chunk, _ = eeg_inlet.pull_chunk(timeout=0.0)
            if chunk:
                for sample in chunk:
                    if sample[0] == MAGIC_SYNC_VAL:
                        collecting_seq = int(sample[1])
                        collected_samples = []
                    elif collecting_seq is not None:
                        collected_samples.append(sample)
                        if len(collected_samples) == required:
                            pending_eeg[collecting_seq] = np.asarray(collected_samples, dtype=np.float32).T
                            
                            # Check if marker arrived earlier
                            if collecting_seq in pending_markers:
                                if q.qsize() < _LSL_QUEUE_MAX:
                                    q.put((pending_markers[collecting_seq], pending_eeg[collecting_seq]), block=False)
                                del pending_markers[collecting_seq]
                                del pending_eeg[collecting_seq]
                            
                            collecting_seq = None

            # 2. Pull marker
            marker_sample, _ = mk_inlet.pull_sample(timeout=0.0)
            if marker_sample:
                try:
                    marker = json.loads(marker_sample[0])
                    seq = marker.get("seq")
                    if seq is not None:
                        pending_markers[seq] = marker
                        
                        # Check if EEG arrived earlier
                        if seq in pending_eeg:
                            if q.qsize() < _LSL_QUEUE_MAX:
                                q.put((pending_markers[seq], pending_eeg[seq]), block=False)
                            del pending_markers[seq]
                            del pending_eeg[seq]
                            
                except (json.JSONDecodeError, IndexError):
                    pass

            # Cleanup memory if things get out of sync
            if len(pending_eeg) > 20:
                pending_eeg.clear()
            if len(pending_markers) > 20:
                pending_markers.clear()

            if not chunk and not marker_sample:
                time.sleep(0.005)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ------------------------------------------------------------------ processing pipeline

def _build_packet(
    marker: dict,
    eeg: np.ndarray,
    pipeline: Pipeline,
    encoder: EEGEncoder,
    retriever: Optional[RetrievalGenerator],
) -> Optional[DisplayPacket]:
    """Process raw EEG → build an atomic DisplayPacket.

    Everything the UI needs is bundled here in one shot:
    cleaned EEG, embedding, retrieval result, ground-truth path,
    labels, cosine score, and latency.
    """
    t0 = time.perf_counter()
    cleaned = pipeline(eeg)
    with torch.no_grad():
        z = encoder(torch.from_numpy(cleaned).unsqueeze(0)).cpu().numpy()[0]
    ret = retriever.generate(z) if retriever is not None else None
    latency_ms = (time.perf_counter() - t0) * 1e3

    true_label = int(marker.get("label", -1))
    pred_label = int(ret["label"]) if ret is not None else -1
    cosine = float(ret["score"]) if ret is not None else 0.0

    return DisplayPacket(
        cleaned_eeg=cleaned,
        z=z,
        retrieval=ret,
        marker=marker,
        gt_image_path=marker.get("image_path"),
        true_label=true_label,
        pred_label=pred_label,
        cosine=cosine,
        latency_ms=latency_ms,
        correct=(true_label == pred_label) and true_label >= 0,
    )


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
        st.session_state.current_packet: Optional[DisplayPacket] = None


def _drain_and_process(cfg) -> bool:
    """Drain ALL queued trials from the LSL queue, skip stale ones,
    process only the LATEST into a DisplayPacket.

    The streamer sends trials faster than the UI can display them
    (e.g. ~2 trials/s vs 1 display every 3s). Without draining,
    the queue fills up and the UI falls further and further behind.
    By always skipping to the newest trial, the UI stays in sync
    with the live stream.

    Returns True if a new packet was produced.
    """
    # Drain everything, keep only the latest (marker, eeg) pair.
    latest = None
    skipped = 0
    while True:
        try:
            item = st.session_state.lsl_queue.get_nowait()
        except queue.Empty:
            break
        latest = item
        skipped += 1

    if latest is None:
        return False

    if skipped > 1:
        print(f"[sync] skipped {skipped - 1} stale trials, showing latest")

    marker, eeg = latest

    if eeg.shape != (cfg.eeg.num_channels, cfg.eeg.trial_length_samples):
        return False

    encoder = _load_encoder_cached(cfg)
    retriever = _load_retriever_cached(cfg)
    packet = _build_packet(
        marker, eeg,
        st.session_state.pipeline, encoder, retriever,
    )
    if packet is None:
        return False

    st.session_state.current_packet = packet
    st.session_state.metrics.record(
        correct=packet.correct,
        cosine=packet.cosine,
        latency_ms=packet.latency_ms,
    )
    st.session_state.history.append((packet.z, packet.true_label, packet.pred_label))
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

    # Drain ALL stale trials from LSL, process only the latest.
    processed = _drain_and_process(cfg)

    # The single packet the UI renders — everything from one trial.
    pkt: Optional[DisplayPacket] = st.session_state.current_packet

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
        if pkt is not None:
            st.plotly_chart(
                build_eeg_plot(pkt.cleaned_eeg, cfg=cfg),
                use_container_width=True, key="eeg",
            )
            st.markdown(f"### {cfg.ui.strings.panel_fft}")
            st.plotly_chart(
                build_fft_plot(pkt.cleaned_eeg, cfg=cfg),
                use_container_width=True, key="fft",
            )
        else:
            st.markdown(
                f'<div class="vcr-card">{cfg.ui.strings.no_stream}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(f"### {cfg.ui.strings.panel_recon}")
        if pkt is not None and pkt.retrieval is not None:
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.markdown("**Ground Truth**")
                if pkt.gt_image_path and Path(pkt.gt_image_path).is_file():
                    gt_img = Image.open(pkt.gt_image_path).convert("RGB")
                    st.image(gt_img, use_container_width=True)
                else:
                    st.write(f"Image not found: {pkt.gt_image_path or '?'}")

            with img_col2:
                st.markdown("**Reconstruction**")
                st.image(pkt.retrieval["image"], use_container_width=True)

            true_class = pkt.marker.get("class_name") or f"label={pkt.true_label}"
            st.markdown(
                f'<div class="vcr-card">'
                f'<div class="vcr-metric-label">{cfg.ui.strings.metric_predicted_class}</div>'
                f'<div class="vcr-metric-value">{pkt.pred_label}</div>'
                f'<div class="vcr-metric-label">{cfg.ui.strings.metric_cosine}</div>'
                f'<div class="vcr-metric-value">{pkt.cosine:.3f}</div>'
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

    if processed:
        # Wait so the user can actually see the image.
        time.sleep(3.0)
    else:
        # Self-refresh tick at the configured cadence if no new trial.
        time.sleep(cfg.ui.refresh_interval_ms / 1000.0)

    st.rerun()


if __name__ == "__main__":
    main()
