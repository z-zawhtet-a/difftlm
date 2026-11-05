"""Nasal cavity geometry for the velopharyngeal branch of the vocal tract model.

Area function: Dang, Honda & Suzuki (1994), subject 1, as tabulated in VocalTractLab
(P. Birkholz, Tube.cpp). Paranasal sinuses: four Helmholtz resonators (Dang & Honda,
1996) with the volumes, neck sizes and coupling positions used in VocalTractLab.

References:
    J. Dang, K. Honda, H. Suzuki. Morphological and acoustical analysis of the nasal
    and the paranasal cavities. JASA 96(4), 2088-2100 (1994).
    J. Dang, K. Honda. Acoustic characteristics of the human paranasal sinuses derived
    from transmission characteristic measurement and morphological observation.
    JASA 100(5), 3374-3383 (1996).
"""

import jax.numpy as jnp

NASAL_LENGTH_CM = 11.4  # Velopharyngeal port to nostrils

# Areas (cm^2) of 19 equal sections, from the velopharyngeal port to the nostrils.
NOSE_AREAS_CM2 = (
    1.63, 2.07, 2.72, 3.59, 4.24, 3.26, 3.04, 3.04, 2.72, 2.5,
    2.39, 2.39, 1.85, 0.76, 1.41, 1.74, 1.30, 1.74, 0.76,
)

# Paranasal sinuses, each coupled at the output of one nasal section.
SINUS_SECTIONS = (8, 9, 11, 12)
SINUS_VOLUMES_CM3 = (11.3, 6.8, 33.0, 6.2)
SINUS_NECK_LENGTHS_CM = (0.3, 0.3, 0.45, 1.0)
SINUS_NECK_AREAS_CM2 = (0.185, 0.185, 0.145, 0.11)
SINUS_LOSS = 1.0 / 65000.0  # Wall conductance per unit inner surface, as in VocalTractLab

PORT_SECTIONS = 4  # Number of nasal sections whose area follows the port opening
MIN_AREA_CM2 = 1e-4  # Closed port; also the floor of the oral area functions


def nasal_areas(A_vp):
    """Nasal area function (cm^2) for a velopharyngeal port area A_vp (cm^2).

    The first PORT_SECTIONS sections blend quadratically from A_vp to the fixed area of
    the next section (VocalTractLab, Tube::setVelumOpening). Areas below MIN_AREA_CM2 are
    clipped, so the gradient with respect to A_vp is zero for a closed port.
    """
    base = jnp.asarray(NOSE_AREAS_CM2)
    i = jnp.arange(PORT_SECTIONS)
    target = base[PORT_SECTIONS]
    port = A_vp + i**2 * (target - A_vp) / PORT_SECTIONS**2
    return jnp.concatenate([jnp.maximum(port, MIN_AREA_CM2), base[PORT_SECTIONS:]])


def sinus_impedances(omega, rho, c, mu):
    """Input impedance of each sinus resonator, keyed by the nasal section it couples to.

    Each sinus is a neck (inertance and boundary-layer resistance) in series with a cavity
    (compliance and a wall-loss conductance). omega: [F] angular frequencies.
    """
    out = {}
    for section, volume, length, area in zip(
        SINUS_SECTIONS, SINUS_VOLUMES_CM3, SINUS_NECK_LENGTHS_CM, SINUS_NECK_AREAS_CM2
    ):
        circumference = 2.0 * jnp.sqrt(jnp.pi * area)
        Z_neck = (
            1j * omega * rho * length / area
            + jnp.sqrt(rho * mu * 0.5) * length * circumference * jnp.sqrt(omega) / area**2
        )
        inner_surface = 4.0 * jnp.pi * (3.0 * volume / (4.0 * jnp.pi)) ** (2.0 / 3.0)
        Z_cavity = 1.0 / (inner_surface * SINUS_LOSS + 1j * omega * volume / (rho * c**2))
        out[section] = Z_neck + Z_cavity
    return out
