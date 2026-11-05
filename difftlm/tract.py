import math
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import scipy.signal

from . import nasal
from .perlin import generate_fractal_noise


@dataclass
class SimulatorOptions:
    SR: float = 44100.0  # Sample rate (Hz)
    L: float = 17.0  # Vocal tract length (cm)
    f0: float = 120.0  # Fundamental frequency (Hz)
    TF: float = 1.0  # Simulation duration (s)
    N: int = 32  # Number of tube sections

    u_amp: float = 400.0  # Volume velocity amplitude (cm^3/s)

    # Physical parameters in CGS units
    rho: float = 1.14e-3  # Air density (g/cm^3)
    c: float = 3.43e4  # Speed of sound (cm/s)
    mu: float = 1.86e-4  # Dynamic viscosity of air (g/(cm*s))
    Mw: float = 2.1  # Wall mass per unit area (g/cm^2)
    Bw: float = 800.0  # Wall resistance per unit area (g/(cm^2*s))
    Kw: float = 84500.0  # Wall stiffness per unit area (g/(cm^2*s^2))
    Re_crit: float = 3500.0  # Critical Reynolds number for turbulence (dimensionless)

    # FDS Parameters
    win_size: float = 0.025  # Window length (s)
    fft_size: int = 512  # FFT size for synthesis (samples)
    hop_ratio: float = 4.0  # Hop size ratio
    a_win_alpha: float = 0.25  # Tukey window shape parameter

    # Nasal branch (rollout_nasal and step_frame_nasal only)
    port_fraction: float = 0.55  # Velopharyngeal port position, as a fraction of the oral sections from the glottis
    nasal_length: float = nasal.NASAL_LENGTH_CM  # Nasal cavity length (cm)
    paranasal_sinuses: bool = True  # Couple four Helmholtz resonators to the nasal cavity
    far_field_ref_area: float = 3.0  # Opening area (cm^2) that sets the far-field output gain


class TransferFunctions(NamedTuple):
    """Frequency responses of one frame on its rfft grid (last axis: frequency).

    Unbranched tube: ``oral`` and ``fric_oral`` give the pressure at the lip radiation
    impedance, Zr * U_lips (the original diffTLM output); ``nasal`` and ``fric_nasal``
    are None. Nasal branch: all outputs are far-field pressures, j*omega*G * U, with U
    the volume velocity at the lips or at the nostrils, so mouth and nose add correctly.
    """

    oral: jax.Array  # [F] glottal flow -> mouth output
    flow: jax.Array  # [n, F] glottal flow -> flow leaving oral section i
    fric_oral: jax.Array  # [n, F] noise flow at the junction after oral section i -> mouth output
    nasal: Optional[jax.Array] = None  # [F] glottal flow -> nostril output
    fric_nasal: Optional[jax.Array] = None  # [n, F] noise flow at the junction after oral section i -> nostril output


def _input_impedance(K, Z_load):
    """Impedance at the input of chain K ([p_out, u_out] = K @ [p_in, u_in]) with load Z_load."""
    return (Z_load * K[..., 1, 1] - K[..., 0, 1]) / (K[..., 0, 0] - Z_load * K[..., 1, 0])


def _flow_gain(K, Z_load):
    """Ratio u_out / u_in of chain K with load Z_load."""
    return 1.0 / (K[..., 0, 0] - Z_load * K[..., 1, 0])


def _shunt(Z):
    """Junction with a side branch of input impedance Z: p is continuous, u_out = u_in - p / Z."""
    one, zero = jnp.ones_like(Z), jnp.zeros_like(Z)
    return jnp.stack([one, zero, -1.0 / Z, one], axis=-1).reshape(Z.shape + (2, 2))


class TransmissionLineSimulator:
    """
    Frequency Domain Vocal Simulator.

    Models acoustic propagation and turbulent noise generation using the frequency domain transmission matrix representation.
    rollout() simulates the unbranched vocal tract. rollout_nasal() adds a nasal cavity, coupled through a
    velopharyngeal port of variable area.
    """

    def __init__(self, opt: SimulatorOptions = SimulatorOptions()):
        self.opt = opt
        self.N = opt.N
        self.SR = opt.SR
        self.TF = opt.TF

        self.NF = int(opt.SR * opt.TF)
        self.dt = 1.0 / opt.SR

        self.lowpass = jax.vmap(self.lowpass_filter, in_axes=(0, None, None))

        self.frame_length = int(opt.SR * opt.win_size)
        self.hop_length = self.frame_length // int(opt.hop_ratio)
        self.n_windows = (self.NF - self.frame_length) // self.hop_length + 1

        # The nasal branch joins the oral tube at the input of this oral section.
        self.port_index = min(max(int(round(opt.port_fraction * (self.N - 1))), 1), self.N - 2)
        # Far-field output gain: the low-frequency inertance of a piston of the reference area,
        # so that output levels stay close to those of the unbranched model.
        self.far_field_gain = (8.0 * opt.rho) / (
            3.0 * math.pi * math.sqrt(math.pi * opt.far_field_ref_area)
        )

        n_octaves = 5
        # Resolve every octave with at least four samples per lattice cell.
        res = self.NF // (4 * 2 ** (n_octaves - 1))
        beta = 2

        self.generate_noise = partial(
            generate_fractal_noise,
            shape=self.NF,
            res=res,
            octaves=n_octaves,
            persistence=2 ** (-0.5 * beta),
            lacunarity=2,
            tileable=False,
        )
        self.generate_noise = jax.vmap(self.generate_noise)

        self.analysis_window = jnp.asarray(
            scipy.signal.windows.tukey(self.frame_length, alpha=self.opt.a_win_alpha)
        )
        self.synthesis_window = jnp.hanning(self.frame_length)

    @staticmethod
    def lowpass_filter(x, f_c=2000.0, sr=16000.0):
        """Direct-form lowpass filter"""
        alpha = 2.0 * jnp.pi * f_c / sr

        def step(y_prev, x_n):
            y = y_prev + (x_n - y_prev) * alpha
            return y, y

        y0 = jnp.array(0.0, dtype=x.dtype)
        _, y = jax.lax.scan(step, y0, x)
        return y

    def consonant_gate(self, u_c, noise, A):
        """
        u_c: volume velocity at constriction
        noise: noise source
        A: area of constriction

        A differentiable gate for consonants
        """
        u_lp = self.lowpass(u_c, 2000.0, self.SR)
        Re2 = (
            (4.0 * self.opt.rho**2) / (jnp.pi * self.opt.mu**2) * u_lp**2 / A[..., None]
        )

        attn = jax.nn.softmax(-A, axis=0)[..., None]
        gate = jax.nn.softplus(Re2 - (self.opt.Re_crit / self.opt.u_amp) ** 2)
        u_n = 1e-4 * noise * gate * attn

        return u_n

    # ------------------------------------------------------------------ acoustic elements

    def section_impedances(self, A, li, omega):
        """Series (Za, half section) and shunt (Zb) impedances of tube sections.

        A: [n] areas (cm^2), li: section length (cm), omega: [F] angular frequencies.
        Returns two [F, n] arrays.
        """
        rho = self.opt.rho
        mu = self.opt.mu
        c = self.opt.c

        circumference = jnp.sqrt(A / jnp.pi) * 2.0 * jnp.pi
        Sa = circumference * li

        # Eq. (1) of Birkholz & Jackel (2004)
        # -------------------------------------
        Li = rho * li / (2.0 * A)
        Ri = (Sa / (2.0 * A**2)) * jnp.sqrt(rho * mu * 0.5)
        Ci = A * li / (rho * c**2)

        Lwi = self.opt.Mw / Sa
        Rwi = self.opt.Bw / Sa
        Cwi = Sa / self.opt.Kw
        # -------------------------------------

        omega = omega[..., None]
        jomega = 1j * omega

        Ri_omega = Ri * jnp.sqrt(omega)

        Za = Ri_omega + Li * jomega
        Zb1 = Rwi + Lwi * jomega + 1.0 / (jomega * Cwi)
        Zb2 = 1.0 / (jomega * Ci)
        Zb = Zb1 * Zb2 / (Zb1 + Zb2)
        return Za, Zb

    def section_matrices(self, A, li, omega):
        """Transfer matrices [F, n, 2, 2] of T-sections, with [p_out, u_out] = K @ [p_in, u_in]."""
        Za, Zb = self.section_impedances(A, li, omega)
        return jnp.stack(
            [1.0 + Za / Zb, -2.0 * Za - Za**2 / Zb, -1.0 / Zb, 1.0 + Za / Zb], axis=-1
        ).reshape(Za.shape + (2, 2))

    def radiation_impedance(self, A_open, omega):
        """Radiation impedance of an opening of area A_open (cm^2)."""
        rho = self.opt.rho
        c = self.opt.c

        # Eq. (2) of Birkholz & Jackel (2004)
        # -------------------------------------
        Lr = (8.0 * rho) / (3.0 * jnp.pi * jnp.sqrt(jnp.pi * A_open))
        Rr = (128.0 * rho * c) / (9.0 * jnp.pi**2 * A_open)
        # -------------------------------------

        jomega = 1j * omega
        return Rr * (jomega * Lr) / (Rr + jomega * Lr)

    def nasal_chain(self, A_vp, omega):
        """Transfer matrix [F, 2, 2] from the velopharyngeal port to the nostrils, and the nostril area."""
        areas = nasal.nasal_areas(A_vp)
        Kn = self.section_matrices(areas, self.opt.nasal_length / areas.shape[0], omega)
        sinuses = (
            nasal.sinus_impedances(omega, self.opt.rho, self.opt.c, self.opt.mu)
            if self.opt.paranasal_sinuses
            else {}
        )
        P = None
        for s in range(areas.shape[0]):
            P = Kn[:, s] if P is None else Kn[:, s] @ P
            if s in sinuses:
                P = _shunt(sinuses[s]) @ P
        return P, areas[-1]

    def transfer_functions(self, A, L, A_vp=None, n_fft=None):
        """Frequency responses of one frame, on the rfft grid of n_fft samples.

        A: [N] areas from the glottis to the lips; A[-1] is the lip opening.
        L: vocal tract length (cm). A_vp: velopharyngeal port area (cm^2), or None for
        the unbranched tube. Noise sources are flow sources at the junction after each
        oral section; for them, the glottis end is a pressure release (p = 0).
        """
        n_fft = self.opt.fft_size if n_fft is None else n_fft
        freqs = jnp.fft.rfftfreq(n_fft, 1.0 / self.SR).clip(min=1e-8)
        omega = 2.0 * jnp.pi * freqs

        Zr = self.radiation_impedance(A[-1], omega)
        Ks = self.section_matrices(A[:-1], L / self.N, omega)  # A[-1] is the boundary area
        n = Ks.shape[1]
        eye = jnp.broadcast_to(jnp.eye(2, dtype=Ks.dtype), Ks[:, 0].shape)

        if A_vp is not None:
            k = self.port_index
            Pn, A_nostril = self.nasal_chain(A_vp, omega)
            Zrn = self.radiation_impedance(A_nostril, omega)
            Zn = _input_impedance(Pn, Zrn)
            nose_gain = _flow_gain(Pn, Zrn)
            # The nasal branch is a side branch at the input of oral section k.
            Ks = Ks.at[:, k].set(Ks[:, k] @ _shunt(Zn))

        # fwd[:, i]  = K_i ... K_0:         glottis -> junction after section i.
        # down[:, i] = K_{n-1} ... K_{i+1}: junction after section i -> lips (identity for i = n-1).
        # Both index the same junction, so they can be combined.
        # This for loop is faster than jax.lax.associative_scan, likely because N is small.
        fwd = [Ks[:, 0]]
        for i in range(1, n):
            fwd.append(Ks[:, i] @ fwd[-1])
        down = [eye]
        for i in range(n - 1, 0, -1):
            down.append(down[-1] @ Ks[:, i])
        fwd = jnp.stack(fwd, axis=1)
        down = jnp.stack(down[::-1], axis=1)

        Zr_ = Zr[:, None]
        Z_up = -fwd[..., 0, 1] / fwd[..., 1, 1]  # Looking to the glottis (p = 0 there)
        Z_down = _input_impedance(down, Zr_)  # Looking to the lips
        flow = _flow_gain(fwd, Z_down)  # Glottal flow -> flow leaving section i
        p_src = Z_up * Z_down / (Z_up + Z_down)  # Junction pressure per unit noise flow
        U_lips = flow[:, -1]  # Z_down[:, -1] = Zr
        U_lips_src = p_src / Z_down * _flow_gain(down, Zr_)

        if A_vp is None:
            return TransferFunctions(
                oral=Zr * U_lips, flow=flow.T, fric_oral=(Zr_ * U_lips_src).T
            )

        # Pressure at the port junction, per unit glottal flow.
        Z_port = Z_down[:, k - 1]  # Nasal branch in parallel with the oral tract beyond it
        p_port = Z_port * flow[:, k - 1]

        # Port pressure per unit pressure at each noise source junction.
        ratio = []
        D = eye  # K_{k-1} ... K_{i+1}: source junction -> port junction
        for i in range(k - 1, -1, -1):
            ratio.append(Z_port / (Z_port * D[..., 1, 1] - D[..., 0, 1]))
            D = D @ Ks[:, i]
        ratio = ratio[::-1]
        for i in range(k, n):  # Sources beyond the port reach it through the upstream tract.
            ratio.append(fwd[:, k - 1, 0, 1] / fwd[:, i, 0, 1])
        ratio = jnp.stack(ratio, axis=1)

        U_nose = p_port / Zn * nose_gain
        U_nose_src = p_src * ratio / Zn[:, None] * nose_gain[:, None]

        G = 1j * omega * self.far_field_gain
        return TransferFunctions(
            oral=G * U_lips,
            flow=flow.T,
            fric_oral=(G[:, None] * U_lips_src).T,
            nasal=G * U_nose,
            fric_nasal=(G[:, None] * U_nose_src).T,
        )

    # ------------------------------------------------------------------ synthesis

    def _noise_spectrum(self, H_flow, uin_fft, noise, A):
        """Spectra of the turbulence noise flows, gated by the flow at each section."""
        u_c = jnp.fft.irfft(H_flow * uin_fft[None, ...])
        u_n = self.consonant_gate(u_c, noise, A)
        return jnp.fft.rfft(u_n)

    @partial(jax.jit, static_argnames=["self"])
    def step_frame(self, uin, A, noise, L):
        """Simulates acoustic pressure output for a single frame segment."""
        T = self.transfer_functions(A, L, n_fft=uin.shape[-1])
        uin_fft = jnp.fft.rfft(uin)
        u_n_fft = self._noise_spectrum(T.flow, uin_fft, noise, A[:-1])
        return jnp.fft.irfft(uin_fft * T.oral + (u_n_fft * T.fric_oral).sum(axis=0))

    @partial(jax.jit, static_argnames=["self"])
    def step_frame_nasal(self, uin, A, A_vp, noise, L):
        """Like step_frame, with a nasal branch. Returns the far-field (p_oral, p_nasal)."""
        T = self.transfer_functions(A, L, A_vp=A_vp, n_fft=uin.shape[-1])
        uin_fft = jnp.fft.rfft(uin)
        u_n_fft = self._noise_spectrum(T.flow, uin_fft, noise, A[:-1])
        p_oral = jnp.fft.irfft(uin_fft * T.oral + (u_n_fft * T.fric_oral).sum(axis=0))
        p_nasal = jnp.fft.irfft(uin_fft * T.nasal + (u_n_fft * T.fric_nasal).sum(axis=0))
        return p_oral, p_nasal

    def _frames(self, uin, key):
        """Windowed, zero-padded glottal flow and noise frames, and their sample indices."""
        noise = self.generate_noise(key=jax.random.split(key, self.N - 1))
        noise = noise.T

        starts = self.hop_length * jnp.arange(self.n_windows)
        frame_idx = jnp.arange(self.frame_length)
        indices = starts[:, None] + frame_idx[None, :]

        # Window input signals
        uins = uin[indices] * self.analysis_window[None, :]
        noises = noise[indices] * self.analysis_window[None, :, None]
        noises = noises.transpose(0, 2, 1)

        # Pad to FFT size
        padding = int(self.opt.fft_size - self.frame_length)
        uins = jnp.pad(uins, ((0, 0), (0, padding)))
        noises = jnp.pad(noises, ((0, 0), (0, 0), (0, padding)))
        return uins, noises, indices

    def _overlap_add(self, frames, indices):
        frames = frames[:, : self.frame_length] * self.synthesis_window
        # scatter-add
        return jnp.zeros(self.NF).at[indices].add(frames)

    @partial(jax.jit, static_argnames=["self"])
    def rollout(self, uin, As, L, key):
        """Performs overlap-add (OLA) synthesis across frames."""

        assert As.shape[0] == self.n_windows

        uins, noises, indices = self._frames(uin, key)
        frames = jax.vmap(lambda u, n, A: self.step_frame(u, A, n, L))(uins, noises, As)
        return self._overlap_add(frames, indices)

    @partial(jax.jit, static_argnames=["self"])
    def rollout_nasal(self, uin, As, A_vps, L, key):
        """Overlap-add synthesis with a nasal branch.

        A_vps: [n_windows] velopharyngeal port areas (cm^2). Returns the far-field
        (p_oral, p_nasal); the microphone signal is p_oral + p_nasal.
        """
        assert As.shape[0] == self.n_windows
        assert A_vps.shape == (self.n_windows,)

        uins, noises, indices = self._frames(uin, key)
        p_oral, p_nasal = jax.vmap(
            lambda u, n, A, a: self.step_frame_nasal(u, A, a, n, L)
        )(uins, noises, As, A_vps)
        return self._overlap_add(p_oral, indices), self._overlap_add(p_nasal, indices)
