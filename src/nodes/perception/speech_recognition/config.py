# Speaker verification
ENROLL_FILE = "assets/ref-voices/my-ref-voice-en.wav"
SPEAKER_DEVICE = None
SPEAKER_THRESHOLD = 0.40  # Verification threshold (0.0-1.0)

# Speech-to-Text (Hugging Face Transformers - Whisper)
#
# Model options (faster to slower): "tiny" < "base" < "small" < "medium" < "large"
# For realtime with GPU: use "base" or "small"
STT_MODEL_ID = "base"  # Balanced: good speed + accuracy
STT_SAMPLE_RATE = 16000
STT_DEVICE = None  # None = auto-detect (cuda if available, else cpu)
STT_LANGUAGE = "en"  # or "auto"
STT_TASK = "transcribe"  # "transcribe" or "translate"
STT_NUM_BEAMS = 2  # Beam search size (1 = greedy/fastest, 5 = default/slower)
STT_MAX_NEW_TOKENS = 128  # Maximum tokens to generate
STT_USE_FP16 = True  # Use FP16 for GPU (faster, less memory)