from __future__ import annotations

import logging
import tarfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("voxnote.diarization")

# Model từ GitHub releases chính thức của k2-fsa/sherpa-onnx — không cần tài khoản
# hay token (khác pyannote gốc bị khóa sau HF gate).
SEGMENTATION_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{EMBEDDING_FILE}"  # "recongition" là typo THẬT trong tag upstream
)

SPEAKER_COLORS = ("speaker-one", "speaker-two", "speaker-three")


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: int


class SpeakerDiarizer(Protocol):
    """Tách lượt nói theo người nói, chạy hoàn toàn trên máy."""

    name: str

    def diarize(self, audio: Any) -> list[SpeakerTurn]:
        """audio: mảng float32 mono 16kHz. Trả các lượt nói đã sắp theo thời gian."""


@dataclass(slots=True)
class NoopDiarizer:
    name: str = "none"

    def diarize(self, audio: Any) -> list[SpeakerTurn]:
        return []


@dataclass(slots=True)
class SherpaOnnxDiarizer:
    """pyannote segmentation-3.0 (ONNX) + embedding CAM++ 3D-Speaker + clustering.

    Chạy trên onnxruntime CPU — không cần PyTorch, không cần HF token.
    """

    model_dir: Path
    threshold: float = 0.5
    num_speakers: int = -1  # -1 = tự đoán số người nói
    auto_download: bool = True
    name: str = "sherpa-onnx"
    _diarizer: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def _segmentation_model(self) -> Path:
        return self.model_dir / SEGMENTATION_DIR / "model.onnx"

    @property
    def _embedding_model(self) -> Path:
        return self.model_dir / EMBEDDING_FILE

    def _download(self, url: str, target: Path) -> None:
        import httpx

        logger.info("Đang tải model diarization: %s", url)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as response:
            response.raise_for_status()
            with temp.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
        temp.replace(target)

    def _ensure_models(self) -> None:
        if self._segmentation_model.is_file() and self._embedding_model.is_file():
            return
        if not self.auto_download:
            raise FileNotFoundError(
                f"Thiếu model diarization trong {self.model_dir} và LOCAL_MODELS_ONLY đang bật."
            )
        if not self._segmentation_model.is_file():
            archive = self.model_dir / "segmentation.tar.bz2"
            self._download(SEGMENTATION_ARCHIVE_URL, archive)
            with tarfile.open(archive, "r:bz2") as tar:
                tar.extractall(self.model_dir, filter="data")
            archive.unlink(missing_ok=True)
        if not self._embedding_model.is_file():
            self._download(EMBEDDING_URL, self._embedding_model)

    def _load(self) -> Any:
        with self._lock:
            if self._diarizer is not None:
                return self._diarizer
            self._ensure_models()
            import sherpa_onnx

            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(self._segmentation_model)
                    ),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(self._embedding_model)
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=self.num_speakers, threshold=self.threshold
                ),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
            return self._diarizer

    def diarize(self, audio: Any) -> list[SpeakerTurn]:
        diarizer = self._load()
        result = diarizer.process(audio).sort_by_start_time()
        return [SpeakerTurn(float(s.start), float(s.end), int(s.speaker)) for s in result]


def single_speaker_layout(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hành vi khi không có diarization: mọi segment thuộc 'Người nói 1'."""
    for segment in segments:
        segment["speaker_id"] = "speaker-1"
    speakers = (
        [{"id": "speaker-1", "name": "Người nói 1", "initials": "N1", "color": SPEAKER_COLORS[0]}]
        if segments
        else []
    )
    return segments, speakers


def assign_speakers(segments: list[dict], turns: list[SpeakerTurn]) -> tuple[list[dict], list[dict]]:
    """Gán mỗi whisper-segment cho người nói có lượt nói giao thời gian nhiều nhất.

    Người nói được đánh số theo thứ tự xuất hiện (giọng đầu tiên = 'Người nói 1'),
    không theo nhãn cluster thô của model.
    """
    if not turns or not segments:
        return single_speaker_layout(segments)

    appearance_order: dict[int, int] = {}
    for segment in segments:
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
        center = (start + end) / 2
        best_turn = max(
            turns,
            key=lambda turn: (
                min(end, turn.end) - max(start, turn.start),  # overlap (âm nếu không giao)
                -abs(center - (turn.start + turn.end) / 2),  # tie-break: lượt gần nhất
            ),
        )
        raw = best_turn.speaker
        if raw not in appearance_order:
            appearance_order[raw] = len(appearance_order) + 1
        segment["speaker_id"] = f"speaker-{appearance_order[raw]}"

    speakers = [
        {
            "id": f"speaker-{index}",
            "name": f"Người nói {index}",
            "initials": f"N{index}",
            "color": SPEAKER_COLORS[(index - 1) % len(SPEAKER_COLORS)],
        }
        for index in sorted(appearance_order.values())
    ]
    return segments, speakers
