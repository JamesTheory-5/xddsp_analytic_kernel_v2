# xddsp_analytic_kernel_v2
MODULE NAME:
**xddsp_analytic_kernel_v2**

DESCRIPTION:
Band-limited analytic kernel oscillator built on a harmonic comb with configurable spectral envelopes (Dirichlet, DSF-exponential, Blackman, Gaussian, cosine band-pass). The kernel is synthesized directly in the time domain via a harmonic sum. This module is written in the XDDSP style: pure functional, NumPy + Numba, tuple-only state, and no side effects. It supports constant or time-varying frequency (per-sample) and returns a real-valued analytic kernel suitable for BLIT-style synthesis and further FDSP processing.

INPUTS:

* `x` : Dummy input array (shape `[n]`), currently unused. It exists solely to satisfy the generic XDDSP interface and to provide the time axis length for processing.
* `params` (for `tick`) : Scalar frequency in Hz and phase offset:

  * `params[0]` = `freq_hz` (float)
  * `params[1]` = `phase_shift` in radians (float)
* `params` (for `process`) : Either scalar or per-sample frequency plus phase offset:

  * If scalar freq: `params = (freq_hz_scalar, phase_shift)`
  * If per-sample freq: `params = (freq_hz_array, phase_shift)` with `freq_hz_array.shape[0] == x.shape[0]`

OUTPUTS:

* `y` : Real-valued analytic kernel samples (Dirichlet / windowed harmonic comb) of shape `[n]`
* `new_state` : Updated state tuple with advanced phase and possibly updated last-used frequency

STATE VARIABLES (tuple layout):
`state = (freq, sr, phase, N_pos, ks, Ak, envelope_type, dsf_a, gauss_sigma, bp_phi)`

Where:

* `freq` : Last-used base frequency in Hz (float64)
* `sr` : Sample rate in Hz (float64)
* `phase` : Current oscillator phase in cycles [0, 1) (float64)
* `N_pos` : Number of positive harmonics (int64); comb runs from `-N_pos` to `+N_pos`
* `ks` : Harmonic index array (float64, shape `[2*N_pos + 1]`), containing `k = -N_pos..N_pos`
* `Ak` : Spectral envelope magnitudes per harmonic index (float64, same shape as `ks`)
* `envelope_type` : Integer code selecting the envelope:

  * `0` = Dirichlet (flat comb)
  * `1` = DSF exponential decay
  * `2` = Blackman window over |k|/N
  * `3` = Gaussian over |k|/N
  * `4` = |cos(k·phi)| band-pass comb
* `dsf_a` : DSF envelope parameter `a` (0 < a < 1)
* `gauss_sigma` : Gaussian sigma controlling bandwidth (useful range ~0.2–0.5)
* `bp_phi` : Band-pass cosine spacing parameter in radians

EQUATIONS / MATH:

Time-domain analytic kernel (real part only):

* Phase in radians:

  * `θ[n] = 2π * phase[n] + phase_shift`

* Instantaneous kernel sample:

  * `y[n] = Σ_{m=0}^{M-1} A_k[m] * cos( ks[m] * θ[n] )`
  * where `ks[m] ∈ { -N_pos, ..., 0, ..., +N_pos }`
  * and `A_k[m]` is the envelope value for harmonic index `ks[m]`

* Phase update:

  * `phase[n+1] = ( phase[n] + freq[n] / sr ) mod 1`
  * implemented as `phase[n+1] = phase[n] + freq[n]/sr - floor(phase[n] + freq[n]/sr)`

* Envelope definitions (before normalization):

  1. Dirichlet:

     * `A_k = 1` for all k
  2. DSF exponential:

     * `A_k = dsf_a^{|k|}`, with `0 < dsf_a < 1`
  3. Blackman:

     * Let `x = |k| / N_pos`
       If `x <= 1`:

       * `A_k = 0.42 - 0.5*cos(2πx) + 0.08*cos(4πx)`
         Else:
       * `A_k = 0`
  4. Gaussian:

     * Let `d = |k| / (gauss_sigma * N_pos)`

       * `A_k = exp(-0.5 * d^2)`
  5. Band-pass cosine:

     * `A_k = |cos(k * bp_phi)|`

* Normalization rule (“unity at zero”):

  * After computing `A_k` for all k, compute:

    * `S = Σ_m A_k[m]`
  * If `S > 0`, then:

    * `A_k[m] ← A_k[m] / S` for all m
  * This ensures `h(0) ≈ 1` for symmetric envelopes, since:

    * `h(0) = Σ_k A_k ≈ 1`

through-zero rules:

* Frequency can be any real-valued scalar (including time-varying per sample).
* Negative frequency is allowed; it will simply reverse phase direction (still stable).
* Phase wraps with:

  * `phase' = phase + freq/sr`
  * `phase_wrapped = phase' - floor(phase')`
    giving a wrap in `[0, 1)` even for large frequency jumps.

phase wrapping rules:

* `phase` always kept in `[0, 1)` to avoid numerical growth.
* Wrap uses `floor`, not `mod` with negative semantics, to keep everything Numba-safe and unambiguous.

nonlinearities:

* Only trigonometric nonlinearity in `cos(k*θ)`.
* No additional nonlinear waveshaping is applied.

interpolation rules:

* Frequency can be:

  * Scalar constant for the entire block (wrapped into a full vector outside JIT), or
  * Per-sample vector (e.g., for vibrato/FM-style trajectories).
* No further interpolation is performed inside the jitted core; all parameter trajectories are precomputed outside.

time-varying coefficient rules:

* On a per-block basis, the spectral envelope is assumed **fixed** (Ak is not recomputed inside the jitted process loop).
* To change envelope parameters or harmonic count, call `analytic_kernel_update_state()` between processing blocks, which:

  * Recomputes `N_pos` according to Nyquist and/or requested harmonics.
  * Rebuilds `ks` and `Ak` arrays outside jitted code.
  * Refills `Ak` with a jitted envelope filler.

NOTES:

* `N_pos` must be ≥ 1 and is clamped accordingly.
* For stability and spectral correctness:

  * `dsf_a` is clamped to `(1e-9, 0.999999]`.
  * `gauss_sigma` is clamped to a small positive minimum (e.g. `1e-6`).
* Envelope and harmonic vectors are *fixed-size* during a processing call; no array allocations occur inside jitted code.
* The kernel is real and band-limited by harmonic truncation; it is suitable for BLIT-based waveform generation in subsequent modules.

---

## FULL PYTHON MODULE: `xddsp_analytic_kernel_v2.py`

```python
"""
xddsp_analytic_kernel_v2
------------------------

Analytic harmonic-kernel oscillator in XDDSP functional style.

This module synthesizes a band-limited analytic kernel:

    h(theta) = sum_{k=-N..N} A_k * exp(j*k*theta)

and returns the real part:

    y = Re{h(theta)} = sum_k A_k * cos(k*theta)

where theta = 2*pi*phase + phase_shift, phase in cycles [0,1).

Spectral envelope A_k is configurable:

    - Dirichlet     (flat comb over k)
    - DSF-exp       (A_k = a^{|k|})
    - Blackman      (window over |k|/N)
    - Gaussian      (exp(-0.5*(|k|/(sigma*N))^2))
    - Cos band-pass (A_k = |cos(k*phi)|)

Design constraints (XDDSP style):
- Pure functional NumPy + Numba.
- No classes, dataclasses, or dicts.
- State is a tuple of arrays/scalars.
- Tick and process are functional: they return (y, new_state) or (y_block, new_state).
- All Numba functions are @njit(cache=True, fastmath=True).
- No dynamic array allocations inside jitted code (arrays are allocated outside).
- No Python objects inside jitted code.
- No Python branching depending on array values inside jitted code.
"""

import math
from typing import Tuple, Union

import numpy as np
from numba import njit

# -------------------------------------------------------------------------
# Envelope type codes
# -------------------------------------------------------------------------

ENV_DIRICHLET = 0
ENV_DSF_EXP = 1
ENV_BLACKMAN = 2
ENV_GAUSSIAN = 3
ENV_BANDPASS = 4


# -------------------------------------------------------------------------
# Internal helpers (Numba-jitted, no dynamic allocation)
# -------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _ak_fill_envelope_jit(
    ks: np.ndarray,
    Ak: np.ndarray,
    N_pos: int,
    envelope_type: int,
    dsf_a: float,
    gauss_sigma: float,
    bp_phi: float,
) -> None:
    """
    Fill Ak in-place based on ks and envelope parameters.
    Also performs unity-at-zero style normalization (sum(Ak) = 1).
    """
    size = ks.shape[0]
    a_clamped = dsf_a
    if a_clamped < 1e-9:
        a_clamped = 1e-9
    if a_clamped > 0.999999:
        a_clamped = 0.999999

    sigma = gauss_sigma
    if sigma < 1e-9:
        sigma = 1e-9

    s = 0.0
    inv_N = 1.0 / float(N_pos)

    for i in range(size):
        k = ks[i]
        abs_k = abs(int(k))  # k is stored as float, but it is integer-valued
        val = 0.0

        if envelope_type == ENV_DIRICHLET:
            # Flat comb
            val = 1.0

        elif envelope_type == ENV_DSF_EXP:
            # DSF-like exponential decay
            # A_k = a^{|k|}
            val = math.pow(a_clamped, abs_k)

        elif envelope_type == ENV_BLACKMAN:
            # Blackman window over |k|/N_pos
            x = float(abs_k) * inv_N
            if x <= 1.0:
                # Standard Blackman window
                val = (
                    0.42
                    - 0.5 * math.cos(2.0 * math.pi * x)
                    + 0.08 * math.cos(4.0 * math.pi * x)
                )
            else:
                val = 0.0

        elif envelope_type == ENV_GAUSSIAN:
            # Gaussian over |k|/N_pos
            # d = |k| / (sigma * N_pos)
            d = float(abs_k) / (sigma * float(N_pos))
            val = math.exp(-0.5 * d * d)

        elif envelope_type == ENV_BANDPASS:
            # Simple cosine band-pass comb: |cos(k * phi)|
            angle = k * bp_phi
            val = abs(math.cos(angle))

        else:
            # Fallback: flat
            val = 1.0

        Ak[i] = val
        s += val

    # Normalize so that sum(Ak) = 1 (unity-at-zero style)
    if s > 0.0:
        inv_s = 1.0 / s
        for i in range(size):
            Ak[i] *= inv_s
    else:
        # Degenerate: all zero, set DC to 1 for safety
        mid = size // 2
        for i in range(size):
            Ak[i] = 0.0
        if mid >= 0 and mid < size:
            Ak[mid] = 1.0


@njit(cache=True, fastmath=True)
def _ak_tick_core_jit(
    freq: float,
    sr: float,
    phase: float,
    N_pos: int,
    ks: np.ndarray,
    Ak: np.ndarray,
    envelope_type: int,
    dsf_a: float,
    gauss_sigma: float,
    bp_phi: float,
    freq_param: float,
    phase_shift: float,
) -> Tuple[float, float, float]:
    """
    Core tick: compute one kernel sample and updated phase.
    Returns (y, new_phase, new_freq).
    """
    # Use the per-sample freq_param as the effective frequency
    f = freq_param
    # If you want to clamp frequency, do it here if desired:
    # if f < 1e-9:
    #     f = 1e-9

    # Incremental phase (cycles/sample)
    inc = f / sr

    # Phase in radians
    theta = 2.0 * math.pi * phase + phase_shift

    # Harmonic sum: y = sum_k Ak * cos(k*theta)
    acc = 0.0
    size = ks.shape[0]
    for i in range(size):
        k = ks[i]
        amp = Ak[i]
        angle = k * theta
        acc += amp * math.cos(angle)

    y = acc

    # Phase update with wrap to [0, 1)
    new_phase = phase + inc
    # Wrap safely even for large increments
    new_phase = new_phase - math.floor(new_phase)

    # Return y, updated phase, and the last-used frequency (for state)
    return y, new_phase, f


@njit(cache=True, fastmath=True)
def _ak_process_core_jit(
    x: np.ndarray,
    freq_vec: np.ndarray,
    sr: float,
    phase: float,
    N_pos: int,
    ks: np.ndarray,
    Ak: np.ndarray,
    envelope_type: int,
    dsf_a: float,
    gauss_sigma: float,
    bp_phi: float,
    phase_shift: float,
    y_out: np.ndarray,
) -> Tuple[float, float]:
    """
    Core process: loop over samples and fill y_out.
    Returns (final_phase, final_freq).
    """
    n = x.shape[0]
    # We track freq as "last used" for state
    last_freq = 0.0
    ph = phase

    for i in range(n):
        f_i = freq_vec[i]
        y_i, ph, last_freq = _ak_tick_core_jit(
            last_freq,  # not used inside, but kept for uniform call
            sr,
            ph,
            N_pos,
            ks,
            Ak,
            envelope_type,
            dsf_a,
            gauss_sigma,
            bp_phi,
            f_i,
            phase_shift,
        )
        y_out[i] = y_i

    return ph, last_freq


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def analytic_kernel_init(
    freq_hz: float = 110.0,
    sr: float = 48000.0,
    harmonics: Union[int, None] = None,
    envelope_type: int = ENV_DIRICHLET,
    dsf_a: float = 0.98,
    gauss_sigma: float = 0.35,
    bp_phi: float = math.pi / 4.0,
    phase: float = 0.0,
):
    """
    Initialize analytic-kernel state.

    Parameters
    ----------
    freq_hz : float
        Base frequency in Hz.
    sr : float
        Sample rate in Hz.
    harmonics : int or None
        Number of positive harmonics (N_pos). If None, it is chosen
        automatically so that N_pos * freq <= sr/2 (Nyquist-safe).
    envelope_type : int
        Envelope selector (ENV_DIRICHLET, ENV_DSF_EXP, ENV_BLACKMAN,
        ENV_GAUSSIAN, ENV_BANDPASS).
    dsf_a : float
        DSF exponential decay parameter (0 < a < 1).
    gauss_sigma : float
        Gaussian sigma parameter (useful range ~0.2..0.5).
    bp_phi : float
        Band-pass cosine spacing parameter in radians.
    phase : float
        Initial phase in cycles [0, 1).

    Returns
    -------
    state : tuple
        Analytic kernel state tuple:
        (freq, sr, phase, N_pos, ks, Ak, envelope_type, dsf_a, gauss_sigma, bp_phi)
    """
    sr_f = float(sr)
    f0 = float(freq_hz)
    if f0 < 1e-9:
        f0 = 1e-9

    if harmonics is None:
        # Nyquist-safe N_pos such that N_pos * f <= sr/2
        max_h = int((sr_f * 0.5) // f0)
        if max_h < 1:
            max_h = 1
        N_pos = max_h
    else:
        N_pos = int(harmonics)
        if N_pos < 1:
            N_pos = 1
        # Also respect Nyquist safety
        max_h = int((sr_f * 0.5) // f0)
        if max_h < 1:
            max_h = 1
        if N_pos > max_h:
            N_pos = max_h

    # Construct ks and Ak arrays
    size = 2 * N_pos + 1
    ks = np.empty(size, dtype=np.float64)
    Ak = np.empty(size, dtype=np.float64)

    # Fill ks: -N_pos .. +N_pos
    for i in range(size):
        ks[i] = float(i - N_pos)

    # Fill envelope in-place (jitted)
    _ak_fill_envelope_jit(
        ks,
        Ak,
        N_pos,
        int(envelope_type),
        float(dsf_a),
        float(gauss_sigma),
        float(bp_phi),
    )

    # Wrap phase into [0, 1)
    ph = float(phase) % 1.0

    state = (
        f0,
        sr_f,
        ph,
        N_pos,
        ks,
        Ak,
        int(envelope_type),
        float(dsf_a),
        float(gauss_sigma),
        float(bp_phi),
    )
    return state


def analytic_kernel_update_state(
    state,
    freq_hz: Union[None, float] = None,
    harmonics: Union[None, int] = None,
    envelope_type: Union[None, int] = None,
    dsf_a: Union[None, float] = None,
    gauss_sigma: Union[None, float] = None,
    bp_phi: Union[None, float] = None,
):
    """
    Update analytic-kernel state (out of band, non-jitted).

    Parameters
    ----------
    state : tuple
        Existing state tuple.
    freq_hz : float or None
        New base frequency. If None, existing freq is kept.
    harmonics : int or None
        New N_pos (positive harmonics). If None, recomputed from Nyquist/freq.
    envelope_type : int or None
        New envelope selector. If None, existing is kept.
    dsf_a : float or None
        New DSF parameter. If None, existing is kept.
    gauss_sigma : float or None
        New Gaussian sigma. If None, existing is kept.
    bp_phi : float or None
        New band-pass phi. If None, existing is kept.

    Returns
    -------
    new_state : tuple
        Updated state with possibly new harmonic and envelope arrays.
    """
    (
        freq,
        sr,
        phase,
        N_pos,
        ks,
        Ak,
        env_type_old,
        dsf_a_old,
        gauss_sigma_old,
        bp_phi_old,
    ) = state

    new_freq = freq if freq_hz is None else float(freq_hz)
    if new_freq < 1e-9:
        new_freq = 1e-9

    new_env_type = env_type_old if envelope_type is None else int(envelope_type)
    new_dsf_a = dsf_a_old if dsf_a is None else float(dsf_a)
    new_gauss_sigma = gauss_sigma_old if gauss_sigma is None else float(gauss_sigma)
    new_bp_phi = bp_phi_old if bp_phi is None else float(bp_phi)

    # Decide new N_pos
    if harmonics is None:
        max_h = int((sr * 0.5) // new_freq)
        if max_h < 1:
            max_h = 1
        new_N_pos = max_h
    else:
        h_req = int(harmonics)
        if h_req < 1:
            h_req = 1
        max_h = int((sr * 0.5) // new_freq)
        if max_h < 1:
            max_h = 1
        if h_req > max_h:
            h_req = max_h
        new_N_pos = h_req

    # Rebuild ks and Ak if N_pos changes; otherwise, reuse arrays
    if new_N_pos != N_pos:
        size = 2 * new_N_pos + 1
        ks_new = np.empty(size, dtype=np.float64)
        Ak_new = np.empty(size, dtype=np.float64)
        for i in range(size):
            ks_new[i] = float(i - new_N_pos)
        # Fill new envelope via jitted helper
        _ak_fill_envelope_jit(
            ks_new,
            Ak_new,
            new_N_pos,
            new_env_type,
            new_dsf_a,
            new_gauss_sigma,
            new_bp_phi,
        )
        ks_out = ks_new
        Ak_out = Ak_new
        N_out = new_N_pos
    else:
        # Reuse existing arrays, but refill envelope if parameters changed
        ks_out = ks
        Ak_out = Ak
        N_out = N_pos
        _ak_fill_envelope_jit(
            ks_out,
            Ak_out,
            N_out,
            new_env_type,
            new_dsf_a,
            new_gauss_sigma,
            new_bp_phi,
        )

    new_state = (
        new_freq,
        sr,
        phase,
        N_out,
        ks_out,
        Ak_out,
        new_env_type,
        new_dsf_a,
        new_gauss_sigma,
        new_bp_phi,
    )
    return new_state


def analytic_kernel_tick(
    x: float,
    state,
    params: Tuple[float, float],
):
    """
    Single-sample tick for analytic kernel.

    Parameters
    ----------
    x : float
        Dummy input (ignored). Kept for XDDSP interface compatibility.
    state : tuple
        Current analytic-kernel state.
    params : (freq_hz, phase_shift)
        freq_hz : float
            Instantaneous frequency in Hz for this sample.
        phase_shift : float
            Phase shift in radians.

    Returns
    -------
    y : float
        Kernel sample at current phase.
    new_state : tuple
        Updated state with advanced phase and last-used frequency.
    """
    (
        freq,
        sr,
        phase,
        N_pos,
        ks,
        Ak,
        env_type,
        dsf_a,
        gauss_sigma,
        bp_phi,
    ) = state

    freq_param = float(params[0])
    phase_shift = float(params[1])

    y, new_phase, new_freq = _ak_tick_core_jit(
        freq,
        sr,
        phase,
        N_pos,
        ks,
        Ak,
        env_type,
        dsf_a,
        gauss_sigma,
        bp_phi,
        freq_param,
        phase_shift,
    )

    new_state = (
        new_freq,
        sr,
        new_phase,
        N_pos,
        ks,
        Ak,
        env_type,
        dsf_a,
        gauss_sigma,
        bp_phi,
    )
    return y, new_state


def analytic_kernel_process(
    x: np.ndarray,
    state,
    params: Tuple[Union[float, np.ndarray], float],
):
    """
    Block processing for analytic kernel.

    Parameters
    ----------
    x : np.ndarray
        Dummy input array, shape (n,). Its length defines the output length.
    state : tuple
        Current analytic-kernel state.
    params : (freq_param, phase_shift)
        freq_param : float or np.ndarray
            - If float: constant frequency in Hz for the whole block.
            - If 1D array: per-sample frequency trajectory in Hz, shape (n,).
        phase_shift : float
            Constant phase shift in radians.

    Returns
    -------
    y : np.ndarray
        Output kernel samples, shape (n,).
    new_state : tuple
        Updated state with advanced phase and last-used frequency.
    """
    (
        freq,
        sr,
        phase,
        N_pos,
        ks,
        Ak,
        env_type,
        dsf_a,
        gauss_sigma,
        bp_phi,
    ) = state

    freq_param = params[0]
    phase_shift = float(params[1])

    n = x.shape[0]

    # Build frequency vector outside JIT (no dynamic allocation in jitted core)
    if np.isscalar(freq_param):
        freq_vec = np.full(n, float(freq_param), dtype=np.float64)
    else:
        freq_arr = np.asarray(freq_param, dtype=np.float64)
        if freq_arr.shape[0] != n:
            raise ValueError("freq_param array length must match x length")
        freq_vec = freq_arr

    # Preallocate output
    y = np.empty(n, dtype=np.float64)

    # Call jitted core
    new_phase, new_freq = _ak_process_core_jit(
        x,
        freq_vec,
        sr,
        phase,
        N_pos,
        ks,
        Ak,
        env_type,
        dsf_a,
        gauss_sigma,
        bp_phi,
        phase_shift,
        y,
    )

    new_state = (
        new_freq,
        sr,
        new_phase,
        N_pos,
        ks,
        Ak,
        env_type,
        dsf_a,
        gauss_sigma,
        bp_phi,
    )
    return y, new_state


# -------------------------------------------------------------------------
# Smoke test / demo
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sounddevice as sd

    sr = 48000.0
    dur = 1.0
    n = int(sr * dur)

    # Dummy input (ignored by the kernel, only used for length)
    x = np.zeros(n, dtype=np.float64)

    # Initialize state with Dirichlet comb at 110 Hz
    state = analytic_kernel_init(
        freq_hz=110.0,
        sr=sr,
        harmonics=None,  # Nyquist-safe
        envelope_type=ENV_DIRICHLET,
        dsf_a=0.98,
        gauss_sigma=0.35,
        bp_phi=math.pi / 4.0,
        phase=0.0,
    )

    # Constant frequency parameter
    freq_param = 110.0
    phase_shift = 0.0

    # Process block
    y, state_out = analytic_kernel_process(
        x,
        state,
        (freq_param, phase_shift),
    )

    # Normalize for listening
    max_abs = np.max(np.abs(y))
    if max_abs > 0.0:
        y_norm = y / max_abs
    else:
        y_norm = y

    print("Playing analytic kernel (Dirichlet comb) as audio...")
    sd.play(y_norm, int(sr))
    sd.wait()

    # Plot a short window of the kernel
    W = 600
    t = np.arange(W) / sr

    plt.figure(figsize=(10, 5))
    plt.plot(t, y[:W])
    plt.title("Analytic Kernel (Dirichlet comb) - Time Domain (First {} samples)".format(W))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot magnitude spectrum (for curiosity)
    Y = np.fft.rfft(y * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, np.abs(Y) + 1e-12)
    plt.title("Analytic Kernel Magnitude Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|Y(f)|")
    plt.xlim(0, 4000)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
```

---

### Next modules / things to do

Some natural follow-ups in this XDDSP ecosystem:

1. **BP-BLIT module** (`xddsp_bp_blit_v2`):

   * Uses `xddsp_analytic_kernel_v2` as the spectral kernel.
   * Implements:

     * `bp_blit[n] = h(theta[n]) - h(theta[n] + 2π*duty)`
   * Then, separate modules for **square** and **triangle** via leaky integrators.

2. **Window-designer module** (`xddsp_harmonic_envelopes_v2`):

   * Standalone envelope/Aₖ generator for custom FDSP kernels.

3. **Parameter-smoothing module** (`xddsp_param_smooth_v2`):

   * Numba-jitted 1-pole smoothing of frequency and other parameters, vectorized over blocks.

If you want, I can generate the BP-BLIT + square/triangle module next in the exact same XDDSP style.
