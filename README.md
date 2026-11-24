# NDI Stream Dubber

**Real-Time Broadcast Simultaneous Audio Dubbing for Live NDI Streams**

---

## 🌟 Executive Summary & Overview

**NDI Stream Dubber** is a broadcast-grade, real-time simultaneous audio translation and dubbing pipeline engineered for live video productions, virtual conferences, webinars, and studio environments (vMix, OBS Studio, hardware NDI routers).

It ingests live audio (e.g., Russian speech) over NewTek NDI® or audio files, processes it through high-speed cloud AI translation pipelines (direct speech-to-speech with Google Gemini Live Translate or streaming STT + neural TTS via Soniox), and outputs a synchronized, broadcast-standard 48 kHz NDI stream (`Dub_EN`, `Dub_DE`, etc.) with sub-3-second end-to-end latency.

### Key Capabilities

- **Direct NDI Ingest & Egress**: Seamless network integration with video switchers (vMix, OBS, TriCaster).
- **Sub-3-Second Latency ($\le 3.0$ s SLA)**: Ultra-fast streaming audio pipeline engineered for live events.
- **Broadcast Safety Rules**:
  - **Zero Auto-Switching**: Cloud network hiccups or provider disconnects *never* spontaneously switch live broadcast feeds. Active routing changes strictly on verified operator command.
  - **Silence Heartbeat**: Keeps downstream NDI clock lock active with clean 48 kHz silence frames if upstream pauses.
  - **Freshness Gate ($\le 0.5$ s)**: Verifies that a standby branch is producing valid, timely audio before permitting a live switchover.
  - **15 ms Click-Free Crossfade**: Prevents pops and acoustic clicks during live route changes.
- **Adaptive Playout Buffer (60–320 ms)**: Dynamically absorbs cloud delivery jitter and burstiness, ensuring smooth, natural audio playout without underruns or buffer drift.
- **Dual Operator Control**:
  - **Web Operator Console** (`http://127.0.0.1:8000`): Easy-to-use web interface for live production operators (NDI discovery, language selection, live health metrics, and real-time transcription).
  - **Headless & Terminal CLI**: Rich TUI and FastAPI REST control endpoints for cloud VPS deployments.

---

## 🏗️ Architecture & Pipeline Flow

```
                      ┌────────────────────────┐
                      │    NDI Audio Source    │
                      │  (vMix / OBS / Studio) │
                      └───────────┬────────────┘
                                  │ 16 kHz Mono PCM (20 ms / 320 samples)
                                  ▼
              ┌─────────────────────────────────────────┐
              │         DubberEngine Dispatcher         │
              │  (Bounded Queues, Drop-Oldest Policy)   │
              └───────────┬─────────────────┬───────────┘
                          │                 │
              ┌───────────▼─────────┐   ┌───▼─────────────────────┐
              │  Branch 1: Gemini   │   │    Branch 2: Soniox     │
              │   Live Translate    │   │      STT v6 + TTS       │
              │  (WebSocket Audio)  │   │   (Streaming Pipeline)  │
              └───────────┬─────────┘   └───┬─────────────────────┘
                          │                 │
                          │ 48 kHz Resample │ 48 kHz Resample
                          │                 │
                          └────────┬────────┘
                                   ▼
                      ┌─────────────────────────┐
                      │       DubSelector       │
                      │  - Freshness Gate (0.5s)│
                      │  - Zero Auto-Switching  │
                      │  - 15 ms Crossfade      │
                      │  - Persistent Routing   │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ Adaptive Playout Buffer │
                      │  (60–320 ms Jitter Reg) │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │     NDIAudioSink        │
                      │  (48 kHz `Dub_EN` Out)  │
                      └─────────────────────────┘
```

### Architectural Principles

1. **Decoupled I/O & VPS Ready**: The core pipeline (`DubberEngine`, `DubSelector`, workers) is completely decoupled from network protocols through abstract base classes (`AudioSource`, `AudioSink`). Runs identically on local broadcast workstations or headless Linux cloud instances.
2. **Dual-Branch Translation Engine**:
   - **Branch 1: Gemini Live Translate (`gemini`)**: Direct speech-to-speech translation returning target audio (24 kHz) and intermediate transcripts via the `google-genai` async client, resampled to 48 kHz.
   - **Branch 2: Soniox STT v6 + Streaming TTS (`soniox`)**: Real-time STT streaming tokens to low-latency neural TTS.
3. **Drop-Oldest Queue Policy**: Ingest queues are bounded (`max_queue_size=100`, ~2 seconds). Stalled or slow workers never block ingest or sibling branches; the oldest frame is dropped with logged metrics to guarantee real-time latency.
4. **Independent Audit Logs**: Transcripts are recorded to `logs/transcript_gemini.jsonl` and `logs/transcript_soniox.jsonl` independently of audio output, ensuring complete auditability.

---

## 🚀 Quick Start: Web Operator Console (Recommended)

The Web Console provides an intuitive visual interface to configure, monitor, and operate live dubbing sessions without terminal interaction.

### 1. Launch the Web Console

In your terminal, run:

```bash
uv run dubber web
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

### 2. vMix Audio Routing & Workflow

Follow these steps to integrate with vMix:

1. **Configure vMix Audio Output**:
   - In vMix, open **Settings → Outputs / NDI**.
   - Under the outputs section, enable NDI for an available output (e.g., Output 2).
   - In the audio settings for that output, select an isolated audio bus containing the speaker's microphone (e.g., **Bus A**).
   - Route the speaker's microphone input to Bus A.
2. **Configure Translation in Web Panel**:
   - In the web panel, click **Refresh** next to the NDI Source dropdown.
   - Select the exact vMix NDI source name.
   - Choose the translation direction (e.g., **Russian → English**).
   - Enter your Gemini API key (or pre-configure it in `.env`).
   - Click **Start Translation**.
3. **Receive Dubbed Audio in vMix**:
   - In vMix, click **Add Input → NDI / Desktop Capture**.
   - Select the published NDI output: `<Your-PC-Name> (Dub_EN)`. This is a dedicated 48 kHz broadcast audio stream.
   - Route this input to your master stream, recording bus, or specialist/monitoring headphones.
   - ⚠️ **Important**: On this input, make sure to **disable sending audio to Bus A** (or whichever bus feeds the source audio) to prevent an acoustic feedback loop.
4. **Live Verification**:
   - Have the speaker deliver a few sentences, pause, and resume.
   - Confirm clear, continuous translated audio on both sides of the pause.
   - View real-time translated sentences rendered live in the Web Console transcription card.
5. **Stop Translation**:
   - Click **Stop Translation** to gracefully terminate the cloud session and close the NDI output.

### 3. Monitoring & Telemetry Indicators

- **Input Audio**: Indicates incoming NDI audio packets and signal activity (including detected pauses/silence).
- **Gemini Live**: Displays active WebSocket status, session initialization, and ready states.
- **Output Voicing**: Confirms active audio frame dispatch to the NDI output device.
- **Adaptive Playout Buffer**: Shows dynamic target buffer depth (60–320 ms) computed from cloud packet arrival cadence.
- **Diagnostics**:
  - `local_silence_ms`: Measures padding silence inserted by the playout engine to prevent underruns during network jitter.
  - `provider_zero_ms`: Measures silent/zero-amplitude PCM frames delivered directly from the upstream translation provider.

---

## 🛠️ CLI Operations & Headless Mode

For automated deployments, continuous integration, or headless VPS production, all features can be managed via the CLI.

### 1. Live Broadcast via CLI

Start live capture from NDI and publish to `Dub_EN`:

```bash
uv run dubber run --source ndi --sink ndi --mode live
```

### 2. Headless VPS Mode (No TUI)

For cloud instances without a display or interactive terminal:

```bash
uv run dubber run --source ndi --sink ndi --mode live --no-tui --host 0.0.0.0 --port 8000
```

This exposes the REST API on port 8000 for remote orchestration.

### 3. Offline Test Run & SLA Verification

Verify the full DSP pipeline, VAD, and latency measurement against the $\le 3.0$ s SLA without live hardware:

```bash
# Paced real-time test run (20 ms frame injection rate)
uv run dubber test-run --wav data/test_speech_ru.wav --out-wav output/test_broadcast_en.wav --real-time

# Accelerated unpaced test run (stress test)
uv run dubber test-run --wav data/test_speech_ru.wav --out-wav output/test_broadcast_en.wav
```

Example latency report output:

```
====================================================================
               PHARMALIVE DUBBER LATENCY REPORT
====================================================================
Execution Type:        File Run (WAV)
Provider Mode:         mock
Overall Status:        PASS
Total Evaluated:       85
Passed SLA:            85/85 (100.0%)
Latency SLA Target:    <= 3.00s
--------------------------------------------------------------------
Median latency (p50):  0.140s
95th percentile (p95): 0.142s
Min latency:           0.108s
Max latency:           0.143s
====================================================================
```

---

## 🎮 Operator Controls & REST API

### Interactive Terminal TUI Shortcuts

When running in interactive CLI mode (`dubber run` without `--no-tui`), an interactive dashboard displays real-time branch status:

| Key | Action |
| :---: | :--- |
| `1` / `g` | Switch broadcast route to **Gemini Live Translate** |
| `2` / `s` | Switch broadcast route to **Soniox STT v6 + TTS** |
| `f` | Simulate failure of current active branch (mock mode) |
| `r` | Simulate recovery of failed branch (mock mode) |
| `q` | Clean, graceful shutdown of engine, workers, and sinks |

### REST API Endpoints

The built-in FastAPI control server runs on `http://127.0.0.1:8000` (or configured `--host` / `--port`).

- **`GET /status`** (or `GET /health`):
  Retrieve comprehensive system health, active broadcast branch, freshness metrics, dropped block counters, and recent transcripts.
  ```bash
  curl http://127.0.0.1:8000/status
  ```

- **`POST /switch/{branch}`**:
  Request manual branch switch (`gemini` or `soniox`). Enforces the 0.5 s freshness gate.
  ```bash
  curl -X POST http://127.0.0.1:8000/switch/gemini
  curl -X POST http://127.0.0.1:8000/switch/soniox
  ```
  *Note:* To bypass the freshness gate in emergency situations, pass `?force=true`:
  ```bash
  curl -X POST "http://127.0.0.1:8000/switch/gemini?force=true"
  ```

- **`GET /metrics`**:
  Prometheus-compatible summary of total processed blocks, dropped blocks, and latency percentiles.

---

## 📦 Installation & Setup

### Prerequisites

- **Python**: 3.11 or higher
- **`uv`**: Fast Python package manager ([docs.astral.sh/uv](https://docs.astral.sh/uv/))
- **NDI Runtime 6**:
  - *Windows*: Install NDI Core Suite 6 ([NDI Tools](https://ndi.video/tools/ndi-core-suite/))
  - *Linux*: Install `libndi` / NDI Runtime 6 shared libraries

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/aqwarius2003/ndi-stream-dubber.git
cd ndi-stream-dubber

# Install dependencies using uv
uv sync --extra dev

# Configure environment variables
cp .env.example .env
```

### Environment Configuration (`.env`)

```ini
# Operating mode: mock (synthetic generator) or live (cloud APIs)
DUBBER_PROVIDER_MODE=live

# Audio I/O defaults
DUBBER_SOURCE_TYPE=ndi
DUBBER_NDI_SOURCE_NAME=
DUBBER_NDI_SINK_NAME=Dub_EN

# API Keys (required for live mode)
GEMINI_API_KEY=your_gemini_api_key_here
SONIOX_API_KEY=your_soniox_api_key_here

# Operator server & persistence
DUBBER_STATE_FILE=state.json
DUBBER_HTTP_HOST=127.0.0.1
DUBBER_HTTP_PORT=8000
```

---

## 🧪 Testing & Verification

The test suite validates audio block chunking, NDI wrappers, mock workers, freshness gating, crossfade DSP, state persistence, HTTP API endpoints, Rich TUI, adaptive playout buffer, and latency SLA reporting:

```bash
uv run pytest -v
```

All 141+ tests pass out of the box with zero external dependencies in mock mode.

---

## 📁 Repository Structure

```
├── dubber/
│   ├── cli.py               # CLI entry points ('web', 'run', 'test-run')
│   ├── config.py            # Pydantic v2 configuration (env + CLI options)
│   ├── engine.py            # Async engine coordinator & drop-oldest queues
│   ├── types.py             # AudioBlock, BroadcastAudioChunk, BranchState
│   ├── ingest/
│   │   ├── base.py          # AudioSource ABC
│   │   ├── ndi_source.py    # NDIAudioSource (NDI Runtime 6 capture)
│   │   └── wav_source.py    # WAVAudioSource (paced audio file ingest)
│   ├── workers/
│   │   ├── base.py          # BaseWorker ABC
│   │   ├── gemini_worker.py # Gemini Live Translate worker (live + mock)
│   │   ├── soniox_worker.py # Soniox STT v6 worker (live + mock)
│   │   └── soniox_tts.py    # Soniox streaming neural TTS client
│   ├── selector/
│   │   ├── state.py         # SelectorState persistence (state.json)
│   │   └── selector.py      # DubSelector (freshness check, 15 ms crossfade)
│   ├── sink/
│   │   ├── base.py          # AudioSink ABC
│   │   ├── ndi_sink.py      # NDIAudioSink (NDI Dub_EN publisher)
│   │   ├── playout.py       # Adaptive playout jitter buffer (60–320 ms)
│   │   └── wav_sink.py      # WAVAudioSink (file output recorder)
│   ├── operator/
│   │   ├── web.py           # Web Operator FastAPI backend
│   │   ├── console.html     # Web Operator UI single-page application
│   │   ├── cli.py           # Rich TUI operator console
│   │   └── server.py        # REST API control server
│   └── measure/
│       ├── latency.py       # LatencyAnalyzer and VAD onset detection
│       └── report.py        # SLA compliance report generator
├── data/
│   └── test_speech_ru.wav   # Synthetic Russian speech test audio (16 kHz mono)
├── scripts/
│   └── generate_test_audio.py # Audio asset generation script
├── tests/                   # Comprehensive pytest test suite (141+ tests)
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # Operator & client documentation
```

---

## 📜 License & Compliance

This software is provided for authorized live broadcast production and client evaluation. All rights reserved.
