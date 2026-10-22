"""Matplotlib visualizations of fitted vocal tracts and audio."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import librosa
from matplotlib import rc_context
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
import soundfile as sf


def save_tract_visualizations(
    areas,
    output: Path,
    *,
    audio_path: Path,
    target_audio,
    reconstructed_audio,
    length_cm: float,
    hop_seconds: float,
    window_seconds: float,
    fps: int = 25,
    max_frames: int = 300,
):
    """Save an area heatmap and an MP4 with tube, spectrograms, and fitted audio.

    Areas have shape (windows, sections), in cm². As in the reference
    imitation_autoencoder animation, tube radius is sqrt(area / pi).
    Window centers determine timestamps. Video duration follows audio_path,
    holding the endpoint geometry outside the window centers. Long animations
    are subsampled; all windows remain visible in the heatmap.
    Target and reconstructed audio are mono arrays at audio_path's sample rate
    and length. Their spectrograms share a dB reference and playback cursor.
    Pass the raw reconstruction to preserve amplitude differences in the plots,
    even when the soundtrack has been normalized for playback.
    """
    areas = np.asarray(areas)
    if areas.ndim != 2 or min(areas.shape) < 1:
        raise ValueError("Expected a nonempty (windows, sections) area array.")
    if not np.isfinite(areas).all() or np.any(areas < 0):
        raise ValueError("Tube areas must be finite and nonnegative.")
    if min(length_cm, hop_seconds, window_seconds, fps, max_frames) <= 0:
        raise ValueError("Geometry dimensions, timing, and frame limits must be positive.")
    audio_path = Path(audio_path).resolve()
    audio_info = sf.info(audio_path)
    if audio_info.frames == 0:
        raise ValueError("The animation soundtrack must contain audio samples.")
    signals = [np.asarray(target_audio), np.asarray(reconstructed_audio)]
    for signal in signals:
        if signal.shape != (audio_info.frames,) or not np.isfinite(signal).all():
            raise ValueError("Spectrogram audio must be finite, mono, and match the soundtrack length.")
    duration = audio_info.frames / audio_info.samplerate
    sr = audio_info.samplerate
    win_length = max(1, round(0.025 * sr))
    hop_length = max(1, round(0.00625 * sr))
    n_fft = 1 << (2 * win_length - 1).bit_length()
    # Explicit padding also supports recordings shorter than one FFT window.
    spectra = [
        np.abs(librosa.stft(
            np.pad(signal, (n_fft // 2, n_fft // 2)),
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False,
        ))
        for signal in signals
    ]
    reference = max(*(float(spectrum.max()) for spectrum in spectra), 1e-10)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(len(areas)) * hop_seconds + window_seconds / 2

    fig = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    heatmap = ax.imshow(
        areas.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(times[0] - hop_seconds / 2, times[-1] + hop_seconds / 2, 0, length_cm),
        vmin=0,
        vmax=max(float(areas.max()), 1e-6),
        cmap="viridis",
    )
    ax.set(xlabel="Time (s)", ylabel="Position: glottis → lips (cm)", title="Fitted tube area")
    fig.colorbar(heatmap, ax=ax, label="Cross-sectional area (cm²)")
    fig.savefig(output / "tube_geometry.png", dpi=150)
    fig.clear()

    radii = np.sqrt(areas / np.pi)
    # Connect the radii directly, as in the reference tube visualization.
    xs = np.linspace(0, length_cm, areas.shape[1])
    limit = max(float(radii.max()) * 1.15, 0.1)
    fig = Figure(figsize=(10, 9), dpi=80, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1.1, 1, 1]})
    ax = axes[0]
    ax.set(
        xlim=(0, length_cm), ylim=(-limit, limit),
        xlabel="Position: glottis → lips (cm)", ylabel="Radius (cm)",
    )
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    upper, = ax.plot(xs, radii[0], color="tab:blue", linewidth=1.5)
    lower, = ax.plot(xs, -radii[0], color="tab:blue", linewidth=1.5)
    tube = Polygon(np.zeros((2 * len(xs), 2)), color="tab:blue", alpha=0.2)
    ax.add_patch(tube)
    title = ax.set_title("")

    cursors = []
    for ax, spectrum, label in zip(
        axes[1:], spectra, ("Target spectrogram", "Reconstructed spectrogram")
    ):
        db = 20 * np.log10(np.maximum(spectrum / reference, 1e-4))
        half_bin_khz = sr / n_fft / 2000
        spectrogram = ax.imshow(
            db, origin="lower", aspect="auto", interpolation="nearest",
            extent=(
                -hop_length / (2 * sr), (spectrum.shape[1] - 0.5) * hop_length / sr,
                -half_bin_khz, sr / 2000 + half_bin_khz,
            ),
            cmap="magma", vmin=-80, vmax=0,
        )
        ax.set(
            title=label, xlabel="Time (s)", ylabel="Frequency (kHz)",
            xlim=(0, duration), ylim=(0, sr / 2000),
        )
        cursors.append(ax.axvline(0, color="white", linewidth=1.2))
    fig.colorbar(spectrogram, ax=axes[1:], label="Magnitude (dB)")

    frame_count = min(max_frames, max(1, int(np.ceil(duration * fps))))
    playback_fps = frame_count / duration
    playback_times = np.arange(frame_count) / playback_fps
    indices = np.clip(
        np.rint((playback_times - window_seconds / 2) / hop_seconds).astype(int),
        0, len(areas) - 1,
    )
    if frame_count > 1:
        indices[-1] = len(areas) - 1

    def update(frame):
        index = indices[frame]
        radius = radii[index]
        upper.set_ydata(radius)
        lower.set_ydata(-radius)
        tube.set_xy(np.column_stack([
            np.concatenate([xs, xs[::-1]]),
            np.concatenate([radius, -radius[::-1]]),
        ]))
        time = playback_times[frame]
        title.set_text(f"Fitted tube — t = {time:.3f} s")
        for cursor in cursors:
            cursor.set_xdata([time, time])
        return upper, lower, tube, title, *cursors

    animation = FuncAnimation(
        fig, update, frames=frame_count, interval=1000 / playback_fps,
        repeat=True, cache_frame_data=False,
    )
    # Only the tube and cursors move; solve the panel layout once at export DPI.
    update(0)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    try:
        with TemporaryDirectory(prefix="vocal-video-") as temporary:
            silent_video = Path(temporary) / "tube.mp4"
            with rc_context({"animation.ffmpeg_path": ffmpeg}):
                writer = FFMpegWriter(
                    fps=playback_fps, codec="h264",
                    extra_args=["-crf", "18", "-threads", "1"],
                )
                animation.save(silent_video, writer=writer, dpi=80)
            subprocess.run(
                [
                    ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                    "-i", str(silent_video), "-i", str(audio_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-t", str(duration), "-movflags", "+faststart",
                    str(output / "tube_geometry.mp4"),
                ],
                check=True, capture_output=True,
            )
    finally:
        fig.clear()
