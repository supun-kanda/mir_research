import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


def mean_mel_db(sig, sr, n_mels=128, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(
        y=_to_mono(sig), sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=sr/2
    )
    return librosa.power_to_db(S.mean(axis=1), ref=np.max)


def _to_mono(sig):
    """Squeeze to 1-D float. Accepts (N,), (N, channels) as returned by
    sounddevice/soundfile/scipy, or (channels, N) as used by librosa."""
    sig = np.asarray(sig, dtype=np.float32)
    if sig.ndim == 2:
        ch_axis = 1 if sig.shape[0] > sig.shape[1] else 0
        sig = sig.mean(axis=ch_axis)
    return sig


def _hz_fmt(x, _):
    return f'{x/1000:g}k' if x >= 1000 else f'{x:g}'


def plot_signal(sig, sr, title=None, n_mels=128, n_fft=2048, hop_length=512):
    """Plot waveform (amplitude vs. time) and mean mel spectrum for a signal.

    `sig` can be a single array, or a dict {label: array} to overlay
    several signals (e.g. {'original': y, 'high-pass': y_hpf}).
    Returns (fig, (ax_wave, ax_mel)).
    """
    signals = sig if isinstance(sig, dict) else {None: sig}
    signals = {label: _to_mono(s) for label, s in signals.items()}

    fig, (ax_wave, ax_mel) = plt.subplots(2, 1, figsize=(10, 7))

    # waveform
    for label, s in signals.items():
        t = np.arange(len(s)) / sr
        ax_wave.plot(t, s, label=label, alpha=0.8, linewidth=0.6)
    ax_wave.set(xlabel='Time (s)', ylabel='Amplitude', title='Waveform')
    ax_wave.set_xlim(0, max(len(s) for s in signals.values()) / sr)
    ax_wave.grid(True, alpha=0.3)

    # mean mel spectrum
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr/2)
    for label, s in signals.items():
        ax_mel.plot(mel_freqs, mean_mel_db(s, sr, n_mels, n_fft, hop_length),
                    label=label)
    ax_mel.set(xlabel='Frequency (Hz)', ylabel='Power (dB)',
               title='Mean mel spectrum')
    ax_mel.set_xscale('log')

    # explicit ticks at musically meaningful frequencies
    ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax_mel.set_xticks([t for t in ticks if t <= sr/2])   # drop any above Nyquist
    ax_mel.xaxis.set_major_formatter(FuncFormatter(_hz_fmt))
    ax_mel.xaxis.set_minor_formatter(NullFormatter())    # suppress log minor-tick labels
    ax_mel.set_xlim(mel_freqs[mel_freqs > 0].min(), sr/2)
    ax_mel.grid(True, which='both', alpha=0.3)

    if None not in signals:
        ax_wave.legend()
        ax_mel.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_wave, ax_mel)
