"""Checks of the transmission-line algebra in tract.py.

The reference is a nodal solve of the same lumped circuit (T-sections, radiation loads,
nasal side branch, sinus resonators): it builds the admittance matrix and solves Y p = I.
It uses the simulator's element impedances but none of its chain-matrix algebra.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from difftlm import nasal  # noqa: E402
from difftlm.tract import SimulatorOptions, TransmissionLineSimulator  # noqa: E402


def make_sim():
    return TransmissionLineSimulator(SimulatorOptions(SR=32000, TF=0.25, L=17.5, fft_size=1024))


@pytest.fixture(scope="module")
def sim():
    return make_sim()


def area_function(N, seed, constriction=False):
    """Smooth random area function (cm^2), glottis to lips; the last value is the lip opening."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N)
    A = 2.0 + 1.5 * np.sin(2 * np.pi * (rng.uniform(1, 3) * x + rng.uniform())) + rng.uniform(0, 1, N)
    A = np.clip(A, 0.3, None)
    if constriction:
        A[N - 5 : N - 3] = 0.1
    return A


def omega_grid(sim):
    return 2 * np.pi * np.fft.rfftfreq(sim.opt.fft_size, 1.0 / sim.SR).clip(min=1e-8)


def assert_close(actual, expected, rtol=1e-8):
    """Max relative error over frequency, skipping DC (last axis)."""
    actual, expected = np.asarray(actual)[..., 1:], np.asarray(expected)[..., 1:]
    err = np.max(np.abs(actual - expected) / np.abs(expected))
    assert err < rtol, f"max relative error {err:.2e} >= {rtol:.0e}"


class Circuit:
    """Lumped circuit, solved by nodal analysis."""

    def __init__(self, n_freq):
        self.n_freq, self.n, self.elements = n_freq, 0, []

    def node(self):
        self.n += 1
        return self.n - 1

    def impedance(self, a, b, Z):
        """Impedance Z [F] between nodes a and b (b=None: ground)."""
        self.elements.append((a, b, np.asarray(Z)))

    def tube(self, start, Za, Zb):
        """T-sections Za - Zb - Za; returns the junction and middle nodes."""
        junctions, mids = [start], []
        for i in range(Za.shape[1]):
            m, j = self.node(), self.node()
            self.impedance(junctions[-1], m, Za[:, i])
            self.impedance(m, j, Za[:, i])
            self.impedance(m, None, Zb[:, i])
            junctions.append(j)
            mids.append(m)
        return junctions, mids

    def solve(self, sources, grounded=()):
        """Node pressures [F, nodes, sources] for a unit flow injected at each source node."""
        Y = np.zeros((self.n_freq, self.n, self.n), complex)
        for a, b, Z in self.elements:
            y = 1.0 / Z
            Y[:, a, a] += y
            if b is not None:
                Y[:, b, b] += y
                Y[:, a, b] -= y
                Y[:, b, a] -= y
        I = np.zeros((self.n, len(sources)))
        for col, node in enumerate(sources):
            I[node, col] = 1.0
        keep = [i for i in range(self.n) if i not in set(grounded)]
        rhs = np.broadcast_to(I[keep], (self.n_freq,) + I[keep].shape)
        p = np.zeros((self.n_freq, self.n, len(sources)), complex)
        p[:, keep] = np.linalg.solve(Y[:, keep][:, :, keep], rhs)
        return p


def oral_circuit(sim, A, omega):
    Za, Zb = sim.section_impedances(jnp.asarray(A[:-1]), sim.opt.L / sim.N, jnp.asarray(omega))
    Za, Zb = np.asarray(Za), np.asarray(Zb)
    Zr = np.asarray(sim.radiation_impedance(A[-1], jnp.asarray(omega)))
    circuit = Circuit(len(omega))
    glottis = circuit.node()
    J, M = circuit.tube(glottis, Za, Zb)
    circuit.impedance(J[-1], None, Zr)
    return circuit, glottis, J, M, Za, Zr


def add_nose(sim, circuit, port, A_vp, omega):
    areas = np.asarray(nasal.nasal_areas(A_vp))
    Za, Zb = sim.section_impedances(jnp.asarray(areas), sim.opt.nasal_length / len(areas), jnp.asarray(omega))
    Jn, _ = circuit.tube(port, np.asarray(Za), np.asarray(Zb))
    for s, Z in nasal.sinus_impedances(jnp.asarray(omega), sim.opt.rho, sim.opt.c, sim.opt.mu).items():
        circuit.impedance(Jn[s + 1], None, Z)  # Sinus at the output of nasal section s
    Zrn = np.asarray(sim.radiation_impedance(areas[-1], jnp.asarray(omega)))
    circuit.impedance(Jn[-1], None, Zrn)
    return Jn, Zrn


def section_flows(p, J, M, Za):
    """Flow leaving each section, from node pressures [F, nodes]."""
    return np.stack([(p[:, M[i]] - p[:, J[i + 1]]) / Za[:, i] for i in range(len(M))])


@pytest.mark.parametrize("seed", [0, 1])
def test_unbranched_tract_matches_nodal(sim, seed):
    A = area_function(sim.N, seed)
    omega = omega_grid(sim)
    T = sim.transfer_functions(jnp.asarray(A), sim.opt.L)
    circuit, glottis, J, M, Za, Zr = oral_circuit(sim, A, omega)
    p = circuit.solve([glottis])[..., 0]

    assert T.nasal is None and T.fric_nasal is None
    assert_close(T.oral, p[:, J[-1]])  # Pressure at the lip radiation impedance
    assert_close(T.flow, section_flows(p, J, M, Za))


def test_unbranched_noise_transfer_matches_nodal(sim):
    A = area_function(sim.N, 2, constriction=True)
    omega = omega_grid(sim)
    T = sim.transfer_functions(jnp.asarray(A), sim.opt.L)
    circuit, glottis, J, M, Za, Zr = oral_circuit(sim, A, omega)
    # Unit noise flow at the junction after each section; pressure release at the glottis.
    p = circuit.solve(J[1:], grounded=[glottis])
    assert_close(T.fric_oral, p[:, J[-1], :].T)


@pytest.mark.parametrize("A_vp", [0.05, 0.5, 2.0])
def test_nasal_branch_matches_nodal(sim, A_vp):
    A = area_function(sim.N, 0)
    omega = omega_grid(sim)
    T = sim.transfer_functions(jnp.asarray(A), sim.opt.L, A_vp=A_vp)
    circuit, glottis, J, M, Za, Zr = oral_circuit(sim, A, omega)
    Jn, Zrn = add_nose(sim, circuit, J[sim.port_index], A_vp, omega)
    p = circuit.solve([glottis])[..., 0]

    G = 1j * omega * sim.far_field_gain
    assert_close(T.oral, G * p[:, J[-1]] / Zr)  # Far field from the lip volume velocity
    assert_close(T.nasal, G * p[:, Jn[-1]] / Zrn)  # Far field from the nostril volume velocity
    assert_close(T.flow, section_flows(p, J, M, Za))


def test_nasal_noise_transfer_matches_nodal(sim):
    A = area_function(sim.N, 3, constriction=True)
    A_vp = 0.3
    omega = omega_grid(sim)
    T = sim.transfer_functions(jnp.asarray(A), sim.opt.L, A_vp=A_vp)
    circuit, glottis, J, M, Za, Zr = oral_circuit(sim, A, omega)
    Jn, Zrn = add_nose(sim, circuit, J[sim.port_index], A_vp, omega)
    p = circuit.solve(J[1:], grounded=[glottis])  # Sources before, at and after the port

    G = (1j * omega * sim.far_field_gain)[:, None]
    assert_close(T.fric_oral, (G * p[:, J[-1], :] / Zr[:, None]).T)
    assert_close(T.fric_nasal, (G * p[:, Jn[-1], :] / Zrn[:, None]).T)


def test_closed_port_approaches_unbranched_tract(sim):
    A = area_function(sim.N, 0)
    omega = omega_grid(sim)
    T0 = sim.transfer_functions(jnp.asarray(A), sim.opt.L)
    Zr = np.asarray(sim.radiation_impedance(A[-1], jnp.asarray(omega)))
    G = 1j * omega * sim.far_field_gain

    def flow_deviation(A_vp):
        """Largest flow change, relative to the peak flow of each section (antiresonances are near 0)."""
        T = sim.transfer_functions(jnp.asarray(A), sim.opt.L, A_vp=A_vp)
        d = np.abs(np.asarray(T.flow) - np.asarray(T0.flow))[:, 1:]
        return np.max(d.max(axis=1) / np.abs(np.asarray(T0.flow))[:, 1:].max(axis=1))

    T = sim.transfer_functions(jnp.asarray(A), sim.opt.L, A_vp=0.0)  # Clipped to nasal.MIN_AREA_CM2
    assert_close(T.oral, G * np.asarray(T0.oral) / Zr, rtol=1e-2)
    assert np.max(np.abs(T.nasal[1:]) / np.abs(T.oral[1:])) < 1e-2
    assert flow_deviation(0.0) < 1e-2
    assert flow_deviation(0.0) < flow_deviation(1e-3) < flow_deviation(1e-2)


def frame_inputs(sim, seed=0):
    rng = np.random.default_rng(seed)
    uin = jnp.asarray(rng.standard_normal(sim.NF))
    w = np.linspace(0.0, 1.0, sim.n_windows)[:, None]
    As = jnp.asarray((1 - w) * area_function(sim.N, 0) + w * area_function(sim.N, 1))
    A_vps = jnp.asarray(np.linspace(0.0, 1.0, sim.n_windows))
    return uin, As, A_vps


def expected_ola(sim, uin, transfer):
    """Overlap-add of irfft(rfft(frame) * H) for per-frame responses H [n_windows, F]."""
    out = np.zeros(sim.NF)
    pad = sim.opt.fft_size - sim.frame_length
    for f in range(sim.n_windows):
        start = f * sim.hop_length
        frame = np.asarray(uin[start : start + sim.frame_length]) * np.asarray(sim.analysis_window)
        y = np.fft.irfft(np.fft.rfft(np.pad(frame, (0, pad))) * transfer[f])
        out[start : start + sim.frame_length] += y[: sim.frame_length] * np.asarray(sim.synthesis_window)
    return out


def test_rollouts_match_frame_transfer_functions():
    """Without turbulence noise, both rollouts are an overlap-add of the frame responses."""
    sim = make_sim()  # Fresh instance: rollout is jitted with `self` static
    sim.generate_noise = lambda key: jnp.zeros((sim.N - 1, sim.NF))
    uin, As, A_vps = frame_inputs(sim)
    key = jax.random.key(0)
    T0 = [sim.transfer_functions(As[f], sim.opt.L) for f in range(sim.n_windows)]
    T = [sim.transfer_functions(As[f], sim.opt.L, A_vp=A_vps[f]) for f in range(sim.n_windows)]

    pout = sim.rollout(uin, As, sim.opt.L, key)
    p_oral, p_nasal = sim.rollout_nasal(uin, As, A_vps, sim.opt.L, key)

    np.testing.assert_allclose(pout, expected_ola(sim, uin, np.stack([t.oral for t in T0])), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(p_oral, expected_ola(sim, uin, np.stack([t.oral for t in T])), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(p_nasal, expected_ola(sim, uin, np.stack([t.nasal for t in T])), rtol=1e-9, atol=1e-12)


def test_rollout_nasal_gradient_in_port_area(sim):
    uin, As, _ = frame_inputs(sim)
    key = jax.random.key(0)

    def nasal_energy(a_vp):
        _, p_nasal = sim.rollout_nasal(uin, As, jnp.full(sim.n_windows, a_vp), sim.opt.L, key)
        return jnp.sum(p_nasal**2)

    a0, h = 0.3, 1e-5
    grad = jax.grad(nasal_energy)(a0)
    finite_difference = (nasal_energy(a0 + h) - nasal_energy(a0 - h)) / (2 * h)
    assert np.isfinite(grad) and grad != 0.0
    np.testing.assert_allclose(grad, finite_difference, rtol=1e-4)
