"""Feature extraction for CH-SVM and FFT12-LR."""
from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import welch

ROMANIAN_DIACRITICS = set("ăâîșțĂÂÎȘȚşţŞŢ")
MIN_CHUNK_SIZE = 256
MIN_CHAR_COUNT = 20

FFT12_NFFT = 2048
PERIOD_EDGES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
CHANNEL_DEFINITIONS: list[tuple[str, set[str]]] = [
    ("SPACE", {" "}),
    ("NEWLINE", {"\n"}),
    ("PERIOD", {"."}),
    ("COMMA", {","}),
    ("SEMICOLON", {";"}),
    ("COLON", {":"}),
    ("QUESTION", {"?"}),
    ("EXCLAMATION", {"!"}),
    ("DASH", {"-"}),
    ("QUOTE", {'"', "'"}),
    ("DIGIT", set("0123456789")),
    ("DIACRITIC", set("ăâîșțĂÂÎȘȚ")),
]


def is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


def chunk_text(text: str, chunk_size: int, overlap: int = 0, min_chunk_size: int = MIN_CHUNK_SIZE) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk) < min_chunk_size:
            break
        chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks or [text]


@dataclass
class CharHistogramVectorizer:
    """One-character marginal features plus scalar character statistics."""

    min_char_count: int = MIN_CHAR_COUNT
    vocabulary: list[str] | None = None

    def fit(self, texts: Sequence[str]) -> "CharHistogramVectorizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)
        self.vocabulary = sorted(ch for ch, count in counter.items() if count >= self.min_char_count)
        return self

    @property
    def feature_names(self) -> list[str]:
        if self.vocabulary is None:
            raise RuntimeError("Vectorizer is not fitted.")
        return [f"char_freq:{display_char(ch)}" for ch in self.vocabulary] + [
            "char_entropy",
            "space_ratio",
            "newline_ratio",
            "uppercase_ratio",
            "digit_ratio",
            "punctuation_ratio",
            "romanian_diacritic_ratio",
            "log1p_length",
        ]

    def transform_chunk(self, chunk: str) -> np.ndarray:
        if self.vocabulary is None:
            raise RuntimeError("Vectorizer is not fitted.")
        length = len(chunk)
        if length == 0:
            raise ValueError("Cannot transform an empty chunk.")
        vocab_index = {ch: i for i, ch in enumerate(self.vocabulary)}
        counts: Counter[str] = Counter(chunk)
        row = np.zeros(len(self.vocabulary) + 8, dtype=np.float64)
        for ch, count in counts.items():
            idx = vocab_index.get(ch)
            if idx is not None:
                row[idx] = count / length
        probs = np.asarray([count / length for count in counts.values()], dtype=np.float64)
        entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
        row[len(self.vocabulary) :] = np.asarray(
            [
                entropy,
                counts.get(" ", 0) / length,
                counts.get("\n", 0) / length,
                sum(1 for ch in chunk if ch.isalpha() and ch.isupper()) / length,
                sum(1 for ch in chunk if ch.isdigit()) / length,
                sum(1 for ch in chunk if is_punctuation(ch)) / length,
                sum(1 for ch in chunk if ch in ROMANIAN_DIACRITICS) / length,
                math.log1p(length),
            ],
            dtype=np.float64,
        )
        return row

    def transform_chunks(self, chunks: Sequence[str]) -> np.ndarray:
        rows = [self.transform_chunk(chunk) for chunk in chunks]
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack(rows)


def display_char(character: str) -> str:
    if character == " ":
        return "SPACE"
    if character == "\n":
        return "NEWLINE"
    if character == "\t":
        return "TAB"
    return character


@dataclass
class FFT12Vectorizer:
    """Twelve-channel positional signal features.

    Each channel is represented as an impulse train. Welch spectral density is
    summarized by ten period bands, spectral centroid, and spectral entropy.
    """

    nfft: int = FFT12_NFFT

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for channel_name, _ in CHANNEL_DEFINITIONS:
            for p0, p1 in zip(PERIOD_EDGES[:-1], PERIOD_EDGES[1:]):
                names.append(f"{channel_name}__period_{p0}_{p1}")
            names.append(f"{channel_name}__spectral_centroid")
            names.append(f"{channel_name}__spectral_entropy")
        return names

    def transform_text(self, text: str) -> np.ndarray:
        text = text.replace("\r\n", "\n")
        length = len(text)
        if length == 0:
            return np.zeros(len(self.feature_names), dtype=np.float64)
        nperseg = min(length, self.nfft)
        noverlap = nperseg // 2 if nperseg > 1 else 0
        values: list[float] = []
        for _, channel_chars in CHANNEL_DEFINITIONS:
            signal = np.fromiter((1.0 if ch in channel_chars else 0.0 for ch in text), dtype=np.float64, count=length)
            frequencies, spectrum = welch(
                signal,
                fs=1.0,
                window="hann",
                nfft=self.nfft,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False,
                scaling="density",
                return_onesided=True,
            )
            if spectrum.size:
                spectrum[0] = 0.0
            total = float(spectrum.sum())
            if total <= 0.0:
                values.extend([0.0] * 12)
                continue
            probs = spectrum / (total + 1e-12)
            for period_start, period_end in zip(PERIOD_EDGES[:-1], PERIOD_EDGES[1:]):
                lower_frequency = 1.0 / period_end
                upper_frequency = 1.0 / period_start
                mask = (frequencies >= lower_frequency) & (frequencies < upper_frequency)
                values.append(float(probs[mask].sum()))
            centroid = float((frequencies * probs).sum())
            positive = probs[probs > 0]
            entropy = 0.0 if positive.size == 0 else float(-(positive * np.log(positive)).sum() / np.log(len(probs)))
            values.extend([centroid, entropy])
        return np.asarray(values, dtype=np.float64)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self.transform_text(text) for text in texts]
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack(rows)
