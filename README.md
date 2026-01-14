# Voice Assistant Robot - Phase 1

A modular voice assistant system with wake word detection, speech recognition, intent classification, and audio enhancement capabilities.

## Features

- **Wake Word Detection**: Efficient wake word detection using EfficientWord-Net
- **Audio Enhancement**: Real-time noise reduction using DeepFilterNet
- **Speech-to-Text**: High-quality transcription using OpenAI Whisper
- **Speaker Verification**: Speaker identification using SpeechBrain
- **Intent Classification**: Intent parsing using FunctionGemma LLM
- **Text-to-Speech**: Voice synthesis using Piper TTS

## Requirements

- Python 3.9
- NVIDIA GPU with CUDA support (recommended)
- Ubuntu/Debian Linux
- PortAudio development libraries

## Installation

### 1. Create Virtual Environment

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential portaudio19-dev python3-dev sox libsox-fmt-all
```

### 3. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```
```bash
pip install torch==2.0.1+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117
```

### 4. Install Wake Word Dependencies

```bash
pip install EfficientWord-Net
pip install tflite-runtime librosa
pip install pyaudio webrtcvad sounddevice scipy
```

### 5. Install Audio Enhancement

```bash
pip install deepfilternet
```

### 6. Install Speaker Verification

```bash
pip install speechbrain==1.0.3
pip install huggingface_hub==0.15.1
pip install "numpy==1.23.5" "requests==2.27.0"
```

### 7. Install LLM Dependencies

```bash
pip install accelerate
pip install optimum[onnxruntime-gpu]
pip install bitsandbytes>=0.41.0
pip install pydantic
pip install transformers
```

### 8. Install llama-cpp-python (with CUDA support)

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" \
FORCE_CMAKE=1 \
pip install llama-cpp-python
```
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=72" \
FORCE_CMAKE=1 \
pip install llama-cpp-python
```
### 9. Install Text-to-Speech

```bash
pip install piper-tts

# Download TTS voices
python -m piper.download_voices \
  --download-dir ./text-to-speech/model/us \
  en_US-lessac-high \
  en_US-lessac-low \
  en_US-lessac-medium
```

## Project Structure

```
phase-1/
├── main.py                          # Main entry point
├── config/
│   ├── logging_config.py            # Logging configuration
│   └── __init__.py
├── src/
│   ├── base/                        # Base classes
│   └── nodes/
│       ├── perception/              # Perception modules
│       │   ├── mic_driver/          # Microphone driver
│       │   │   ├── audio.py         # Audio utilities
│       │   │   ├── config.py        # Configuration
│       │   │   ├── enhance.py       # Audio enhancement
│       │   │   ├── model_loader.py  # Model loading
│       │   │   ├── recording.py     # Recording loop
│       │   │   └── wake_word.py     # Wake word detection
│       │   ├── mic_driver_node.py   # Mic driver node
│       │   └── speech_recognition/  # Speech recognition
│       │       ├── audio_utils.py
│       │       ├── config.py
│       │       ├── model_loader.py
│       │       ├── speaker_verification.py
│       │       ├── speech_to_text.py
│       │       └── transcription.py
│       ├── cognitive/               # Cognitive modules
│       │   ├── classify_intent/     # Intent classification
│       │   └── models/              # Model files
│       │       └── functiongemma-270m-it-Q4_K_M.gguf
│       └── actuator/                # Actuator modules
├── assets/
│   ├── audio/                       # Audio files
│   ├── ebedding-json/               # Wake word embeddings
│   ├── processed/                   # Processed audio
│   ├── raw/                         # Raw audio recordings
│   └── ref-voices/                  # Reference voices
└── venv/                            # Virtual environment
```

## Usage

### Run the Main Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run the mic driver node
python main.py
```

The application will:
1. Initialize all models (wake word detector, audio enhancer, STT, etc.)
2. Wait for the wake word "hello_robot"
3. When detected, start recording audio
4. Enhance the audio and transcribe to text
5. Classify intent and process the command

### Stop the Application

Press `Ctrl+C` to gracefully shutdown the application.

## Configuration

### Wake Word

- Default wake word: `hello_robot`
- Configuration file: `assets/ebedding-json/hello_robot_ref.json`
- Threshold: 0.68

### Audio Settings

- Sample rate: 48000 Hz
- Channels: 1 (mono)
- Format: 16-bit PCM
- Frame duration: 30ms

### Model Paths

- FunctionGemma model: `src/nodes/cognitive/models/functiongemma-270m-it-Q4_K_M.gguf`
- TTS models: `./text-to-speech/model/us/`

## Notes

- Ensure your microphone is properly configured and accessible
- GPU acceleration is recommended for better performance
- First run will download models from HuggingFace (may take time)
- Audio files are saved in `assets/raw/` and `assets/processed/` directories

## Troubleshooting

### Audio Device Issues

If you encounter audio device errors:
```bash
# Check available audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count())]"
```

### CUDA/GPU Issues

If GPU is not detected:
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Reinstall llama-cpp-python with CUDA support (see Installation step 8)

### PortAudio Issues

If PortAudio errors occur:
```bash
sudo apt install -y portaudio19-dev
pip install --force-reinstall pyaudio
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

