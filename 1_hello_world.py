import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    os.makedirs("figures", exist_ok=True)

    # librosa ships with a few example audio files. "nutcracker" is
    # a short orchestral excerpt — swap for "brahms" (solo piano) or
    # any local WAV/MP3 later.
    sample_name = "nutcracker"
    filename = librosa.example(sample_name)

    # Load the first 20 seconds, preserving the original sample rate.
    y, sr = librosa.load(filename, sr=None, duration=20.0)

    print(f"Loaded example: {sample_name}")
    print(f"  sample rate: {sr} Hz")
    print(f"  duration:    {len(y) / sr:.2f} s")
    print(f"  samples:     {len(y):,}")

    # Mel-spectrogram: time-frequency representation scaled to human
    # pitch perception. The workhorse input to most audio ML models.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Chromagram: 12-dimensional representation of pitch class (C, C#, D, ...)
    # over time. The standard feature for key / chord analysis.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Tempo (BPM) and beat times — a one-line glimpse at rhythm analysis.
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # librosa may return tempo as a 0-d array; coerce to float for printing.
    tempo_val = float(np.atleast_1d(tempo)[0])
    print(f"  estimated tempo: {tempo_val:.1f} BPM")
    print(f"  beats detected:  {len(beats)}")

    # Plot: waveform, mel-spectrogram, chromagram — stacked, shared x-axis.
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform — {sample_name}")
    axes[0].set_ylabel("Amplitude")

    mel_img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], fmax=8000
    )
    axes[1].set_title("Mel spectrogram (dB)")
    fig.colorbar(mel_img, ax=axes[1], format="%+2.0f dB")

    chroma_img = librosa.display.specshow(
        chroma, sr=sr, x_axis="time", y_axis="chroma", ax=axes[2]
    )
    axes[2].set_title("Chromagram (pitch-class energy)")
    fig.colorbar(chroma_img, ax=axes[2])

    plt.tight_layout()
    out_path = os.path.join("figures", "first_look.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
