# PharmaLive Dubber

**Real-Time Dual-Branch RU → EN Simultaneous Dubbing System (Task 1A Pilot)**

PharmaLive Dubber is a broadcast-grade simultaneous dubbing pipeline designed for live events, streams, and remote production. It captures Russian speech via NDI or file ingest, runs two independent cloud translation branches in parallel with millisecond-level telemetry, and routes the operator-selected branch to a single persistent NDI publisher (`Dub_EN`) under strict zero-auto-switching broadcast rules.

---

## Key Features & Architecture

```
                    ┌────────────────────────┐
                    │      AudioSource       │
                    │ (NDI / Real-Time WAV)  │
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
                    │  - State File (JSON)    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        AudioSink        │
                    │ (NDI Dub_EN / Out WAV)  │
                    └─────────────────────────┘
```

### 1. Architectural Isolation & VPS-Readiness
- **Decoupled I/O**: The translation core (`DubberEngine`, `DubSelector`, workers) is completely decoupled from audio transmission protocols through abstract base classes `AudioSource` and `AudioSink`.
- **VPS Deployment Ready**: The pipeline runs equally well in an on-premises studio with local NDI devices or on a headless Linux cloud VPS receiving and publishing audio streams (WebRTC, SRT, or NDI over bridge).

### 2. Dual-Branch Translation Core
- **Branch 1: Gemini Live Translate (`gemini`)**:
  - Model: `gemini-3.5-live-translate-preview` via `google-genai` async client.
  - Direct speech-to-speech translation returning English audio (24 kHz) and intermediate transcripts.
  - Resampled to broadcast standard 48 kHz PCM.
- **Branch 2: Soniox STT v6 + Streaming TTS (`soniox`)**:
  - Model: Soniox `stt-rt-v6` one-way real-time translation RU → EN.
  - Completed tokens are streamed immediately to Soniox TTS with pre-configured broadcast voice.
  - Synthesized audio is resampled to 48 kHz PCM.
- **Independent Transcript Logs**:
  - Transcripts are recorded to `logs/transcript_gemini.jsonl` and `logs/transcript_soniox.jsonl` independently of audio output, ensuring full auditability.
- **Drop-Oldest Queue Policy**:
  - Ingest queues are bounded (`max_queue_size=100`, ~2 seconds).
  - Stalled or slow workers never block ingest or the sibling branch; the oldest frame is dropped and logged with its `block_index`.

### 3. Broadcast Selector Rules (`DubSelector`)
- **Zero Auto-Switching**:
  - Failures, timeouts, network disconnects, or restarts **never** trigger automatic route switching. The active branch is changed **exclusively** by deliberate operator command.
  - If the active branch goes silent or fails, `Dub_EN` outputs clean silence frames to maintain downstream NDI clock lock while sounding an operator alert.
- **Freshness Gate ($\le 0.5$ s)**:
  - When the operator issues a switch command, the standby branch is checked for recent audio:
    $$\Delta t = t_{\text{current}} - t_{\text{last\_audio}} \le 0.5\text{ s}$$
  - If the standby branch is stale ($> 0.5$ s) or in an error state, the switch is rejected (`REJECTED: BACKUP NOT FRESH`). Late recovery requires an explicit new operator command.
- **15 ms Click-Free Crossfade**:
  - Seamless branch transitions using an equal-power/linear 15 ms fade between old and new audio streams.
- **Persistent Routing (`state.json`)**:
  - Active branch selection is atomically persisted and restored upon process restart.

---

## Requirements & Setup

### Prerequisites
- **Python**: 3.11 or higher
- **`uv`**: Fast Python package and environment manager ([docs.astral.sh/uv](https://docs.astral.sh/uv/))
- **NDI Runtime 6**: Required for live NDI streaming ([NDI SDK](https://ndi.video/tools/ndi-core-suite/))
  - *Windows*: Install NDI Core Suite / Runtime 6.
  - *Linux (VPS)*: Install `libndi` / NDI Runtime 6 shared libraries.

### Installation

Clone the repository and install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/pharmalive/audio_translator.git
cd audio_translator

# Synchronize dependencies with uv (creates virtual environment automatically)
uv sync
```

### Environment Configuration

Copy `.env.example` to `.env` and fill in API credentials if running in live mode:

```bash
cp .env.example .env
```

Key variables in `.env`:
```ini
# Operating mode: mock (synthetic generator) or live (cloud APIs)
DUBBER_PROVIDER_MODE=mock

# Audio I/O defaults
DUBBER_SOURCE_TYPE=wav
DUBBER_WAV_FILE=data/test_speech_ru.wav
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

## Quickstart & CLI Commands

All commands are executed using `uv run dubber <subcommand>`.

### 1. Generate Test Audio Asset
Generate a standardized 6.0-second synthetic Russian speech WAV file (16 kHz mono 16-bit PCM with speech harmonics, silence, and pause markers):

```bash
uv run python scripts/generate_test_audio.py
```

Output: `data/test_speech_ru.wav`.

### 2. Local Reproducible Test Run (SLA Verification)
Run an offline end-to-end verification through the full DSP pipeline, measuring audio latency against the $\le 3.0$ s SLA:

```bash
# Fast unpaced test run
uv run dubber test-run --wav data/test_speech_ru.wav --out-wav output/test_broadcast_en.wav

# Paced real-time test run
uv run dubber test-run --wav data/test_speech_ru.wav --out-wav output/test_broadcast_en.wav --real-time
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

### 3. Live Broadcast Run (NDI & Live Cloud APIs)
Start full live broadcasting capturing from NDI and publishing to `Dub_EN`:

```bash
uv run dubber run --source ndi --sink ndi --mode live
```

### 4. Headless VPS / Operator API Mode
Run in background/headless mode without interactive Rich TUI, exposing only the HTTP control API:

```bash
uv run dubber run --source ndi --sink ndi --mode live --no-tui --host 0.0.0.0 --port 8000
```

---

## Operator Controls

### Interactive Rich Console (TUI)
When running without `--no-tui`, an interactive dashboard displays real-time branch statuses, audio freshness, latencies, and latest transcripts.

| Key | Action |
|:---:|:---|
| `1` / `g` | Switch broadcast route to **Gemini Live Translate** |
| `2` / `s` | Switch broadcast route to **Soniox STT v6 + TTS** |
| `f` | Simulate failure of current active branch (mock mode) |
| `r` | Simulate recovery of failed branch (mock mode) |
| `q` | Clean, graceful shutdown of engine, workers, and sinks |

### HTTP REST API Endpoints
The built-in FastAPI control server runs on `http://127.0.0.1:8000` (or configured `--host` / `--port`).

- **`GET /status`** (or `GET /health`):
  Retrieve comprehensive system health, active broadcast branch, freshness metrics, dropped block counters, and recent transcripts.
  ```bash
  curl http://127.0.0.1:8000/status
  ```
- **`POST /switch/{branch}`**:
  Request manual branch switch (`gemini` or `soniox`). Validates against the 0.5 s freshness gate.
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

## Latency Measurement Methodology & Acceptance Criteria

PharmaLive Dubber implements a rigorous latency measurement pipeline:

1. **End-to-End Speech Latency ($\le 3.0$ s SLA)**:
   - Input speech onset ($T_{\text{in}}$) is detected using an energy-threshold voice activity detector (VAD).
   - Translated output onset ($T_{\text{out}}$) is measured at the output sink when the first audible translated word is emitted.
   - Total latency: $L = T_{\text{out}} - T_{\text{in}}$.
   - Acceptance requirement: $p50$, $p95$, and maximum latency must remain $\le 3.0$ seconds.
2. **Switching Response Latency ($\le 0.5$ s SLA)**:
   - Measured from operator command timestamp ($T_{\text{cmd}}$) to the first audible sample of the new branch ($T_{\text{switched}}$).
   - Freshness gate ensures standby branch was generating valid audio $\le 0.5$ s ago.
3. **Branch Failure Isolation**:
   - Disconnecting or stalling either branch must never interrupt the sibling branch.
   - Dropped blocks are tracked per-branch and logged.

---

## Running Unit Tests

The test suite validates all components including audio block chunking, NDI wrappers, mock workers, freshness gating, crossfade DSP, state persistence, HTTP API endpoints, Rich TUI, and latency reporting:

```bash
uv run pytest -v
```

All 117+ tests pass out of the box with zero external dependencies in mock mode.

---

## Project Structure

```
d:\audio_translator\
├── dubber/
│   ├── __init__.py
│   ├── cli.py               # CLI entry points ('run', 'test-run')
│   ├── config.py            # Pydantic v2 configuration (env + CLI options)
│   ├── engine.py            # Async engine coordinator & drop-oldest queues
│   ├── types.py             # AudioBlock, BroadcastAudioChunk, BranchState
│   ├── ingest/
│   │   ├── base.py          # AudioSource ABC
│   │   ├── ndi_source.py    # NDIAudioSource (NDI Runtime 6 capture)
│   │   └── wav_source.py    # WAVAudioSource (real-time paced audio file ingest)
│   ├── workers/
│   │   ├── base.py          # BaseWorker ABC
│   │   ├── gemini_worker.py # Gemini Live Translate worker (live + mock)
│   │   └── soniox_worker.py # Soniox STT v6 + streaming TTS worker (live + mock)
│   ├── selector/
│   │   ├── state.py         # SelectorState persistence (state.json)
│   │   └── selector.py      # DubSelector (freshness check, 15ms crossfade)
│   ├── sink/
│   │   ├── base.py          # AudioSink ABC
│   │   ├── ndi_sink.py      # NDIAudioSink (NDI Dub_EN publisher)
│   │   └── wav_sink.py      # WAVAudioSink (file output recorder)
│   ├── operator/
│   │   ├── cli.py           # Rich TUI operator console
│   │   └── server.py        # FastAPI control server & REST API
│   └── measure/
│       ├── latency.py       # LatencyAnalyzer and VAD onset detection
│       └── report.py        # Formatted SLA compliance report generator
├── data/
│   └── test_speech_ru.wav   # Synthetic Russian speech test audio (16 kHz mono)
├── scripts/
│   └── generate_test_audio.py # Audio asset generation script
├── tests/                   # Comprehensive pytest suite
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # Documentation and operator guide
```
