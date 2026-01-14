import time
import logging
from dataclasses import dataclass

import numpy as np
import torch
import sounddevice as sd
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# =========================================================
# Logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("RealtimePhoWhisper5s")

# =========================================================
# Config
# =========================================================
@dataclass
class STTConfig:
    model_id: str = "openai/whisper-base"
    sample_rate: int = 16000
    record_seconds: int = 5

    # Model
    use_fp16: bool = True
    language: str = "en"
    task: str = "transcribe"

    # Generation
    max_new_tokens: int = 256
    num_beams: int = 1


# =========================================================
# PhoWhisper Engine
# =========================================================
class PhoWhisperEngine:
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU")
        logger.info(f"Device: {self.device}")

        logger.info(f"Loading model `{cfg.model_id}` ...")
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(cfg.model_id).to(self.device)

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=cfg.language, task=cfg.task
        )

        if self.device == "cuda" and cfg.use_fp16:
            self.model = self.model.half()
            logger.info("Model cast to FP16")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(
            audio,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding=True
        )

        input_features = inputs["input_features"].to(self.device)
        if self.device == "cuda" and self.cfg.use_fp16:
            input_features = input_features.half()

        start_time = time.perf_counter()
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(self.device == "cuda" and self.cfg.use_fp16)
        ):
            tokens = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams
            )
        inference_time = time.perf_counter() - start_time
        logger.info(f"Model inference took {inference_time:.2f}s")

        text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return text.strip()


# =========================================================
# Realtime Mic (Fixed 5 seconds)
# =========================================================
class RealtimeSTT:
    def __init__(self, cfg: STTConfig, engine: PhoWhisperEngine):
        self.cfg = cfg
        self.engine = engine

    def record(self) -> np.ndarray:
        print(f"🎤 Recording {self.cfg.record_seconds}s...")
        audio = sd.rec(
            int(self.cfg.record_seconds * self.cfg.sample_rate),
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        return audio[:, 0]

    def run(self):
        print("🎤 Fixed-window Speech-to-Text started (Ctrl+C to stop)\n")
        try:
            while True:
                audio = self.record()

                print("🧠 Transcribing...")
                start = time.perf_counter()
                text = self.engine.transcribe(audio)
                elapsed = time.perf_counter() - start

                if text:
                    print(f"🗣️  {text}")
                else:
                    print("🗣️  [No speech detected]")

                print(f"⏱️  {elapsed:.2f}s\n")

        except KeyboardInterrupt:
            print("\n🛑 Stopped")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    cfg = STTConfig(
        model_id="openai/whisper-base",
        record_seconds=10,
        use_fp16=True,
        language="en",
    )

    engine = PhoWhisperEngine(cfg)
    app = RealtimeSTT(cfg, engine)
    app.run()
