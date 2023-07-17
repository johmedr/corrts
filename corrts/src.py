import numpy as np
import scipy.signal
import scipy.stats

from numpy.lib.stride_tricks import sliding_window_view
from numpy.core.multiarray import normalize_axis_index


def corrts(x, y, axis=-1, coef="pearson", p_value=True, correction="bh"): 
    """
    """
    shape = np.broadcast_shapes(x.shape, y.shape)
    
    x = np.broadcast_to(x, shape)
    y = np.broadcast_to(y, shape)

# Utils
# -----
def standardize(x, axis=-1, ddof=1): 
    return (x - np.mean(x, axis=axis, keepdims=True)) / np.std(x, axis=axis, keepdims=True, ddof=ddof)

def fisher_transform(r): 
    return np.arctanh(r)

def inv_fisher_transform(z): 
    return np.tanh(z)

def rank(x, axis=-1): 
    return np.argsort(x, axis=axis).argsort(axis=axis)

def rank_normalization(x, axis=-1): 
    u = rank(x, axis=axis) / (x.shape[axis] - 1)

    return scipy.stats.norm.ppf(u)

def array_slice(a, axis, start, end, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

def toeplitz(r, axis=-1, dtype=None, hermitian=True, subok=False, writeable=False): 
    r = np.asarray(r, dtype)
    axis = normalize_axis_index(axis, r.ndim)
    
    if hermitian: 
        r = np.concatenate([array_slice(r, axis, r.shape[axis],  0, -1), r], axis)
    
    T = sliding_window_view(r, (r.shape[axis] // 2 + 1,), axis=axis, subok=subok, writeable=writeable)
    T = np.flip(T, axis + 1)
    
    return T

# Correlations
# ------------
def pearson_r(x, y, axis=-1): 
    u = standardize(x, axis)
    v = standardize(y, axis)

    return (u * v).mean(axis=axis)

def spearman_r(x, y, axis=-1): 
    u = rank(x, axis=axis)
    v = rank(y, axis=axis)

    return pearson_r(u, v, axis=axis)

def stats_summary(metrics, x, y, axis=-1, n_surrogates=5000): 
    if metrics == 'pearson': 
        metrics = pearson_r
    elif metrics == 'spearman':
        metrics = spearman_r

    # Basics
    r = metrics(x, y, axis)
    z = fisher_transform(r)
    n = x.shape[axis]
    p = dict() 

    rg = estimate_roughness(x, y, axis=axis)

    # Uncorr
    p['uncorrected'] = z_pvalue(z, n)


    # EDF correction
    acf_x    = acf(x, axis=axis)
    acf_y    = acf(y, axis=axis)

    a = dict() 
    a['rft']        = edfs_factor_rft(x, y, axis)
    a['bh']         = edfs_factor_bh(acf_x, acf_y, axis)
    a['quenouille'] = edfs_factor_quenouille(acf_x, acf_y, axis)
    a['bartlett']   = edfs_factor_bartlett(acf_x, acf_y, axis)

    edfs  = dict()
    kappa = dict()
    for k, ak in a.items(): 
        edfs[k]  = ak * n
        p[k]     = z_pvalue(z, edfs[k])
        kappa[k] = fisher_scaling(ak, r) 


    # Sampling
    if n_surrogates > 0:
        p['surrogates'] = bootstrap_p(metrics, x, y, n_surrogates, axis=axis)


    summary = {
        'r': r, 'z': z, 'n': n, 'p':p, 'aedfs': a, 'nedfs': edfs, 'roughness': rg, 'kappa': kappa
    } 

    return summary

def z_pvalue(z, n): 
    return 2 * scipy.stats.norm.cdf(-np.abs(z), 0, 1./np.sqrt(n-3))

def fisher_scaling(a, r): 
    z = fisher_transform(r)
    w = inv_fisher_transform(a * z)
    
    return w

# Autocorrelation estimators
# --------------------------
def acf_welch(x, axis=-1):
    x   = standardize(x, axis)

    Pxx = 0.5 * scipy.signal.welch(x, axis=axis)[1] # 0.5 here compensate for left + right part of spectra
    Pxx = np.concatenate([Pxx, np.flip(np.delete(Pxx, 0, axis=axis), axis=axis)], axis=axis) 
    Rxx = np.real(np.fft.ifft(Pxx, axis=axis)) 
    Rxx /= Rxx[0]

    return Rxx 

def acf_fft(x, axis=-1): 
    x   = standardize(x, axis)

    Rxx = np.real(np.fft.ifft(np.abs(np.fft.fft(x, axis=axis))**2, axis=axis))
    Rxx /= Rxx[0]
    
    return Rxx

def acf_sample(x, axis=-1, mode='unbiased'):
    assert(mode in ('classic', 'unbiased'))

    x = np.moveaxis(x, axis, -1)
    x = standardize(x, -1)

    if mode == 'unbiased': 
        u = np.ones(x.shape[axis])
        n = np.correlate(u, u, mode='same')
        corr = lambda y: np.correlate(y,y, mode='same') / n
    else: 
        corr = lambda y: np.correlate(y,y, mode='same') / x.shape[-1]

    Rxx  = np.apply_along_axis(corr, -1, x)
    Rxx  = np.concatenate([Rxx[..., Rxx.shape[-1]//2:], Rxx[..., :Rxx.shape[-1]//2]], axis=-1)
    Rxx /= Rxx[0]

    Rxx  = np.moveaxis(Rxx, -1, axis)

    return Rxx

_ACF_METHOD_MAP = {
    "fft": acf_fft, 
    "welch": acf_welch, 
    "sample": acf_sample
}

def acf(x, axis=-1, method="fft", **kwargs): 
    return _ACF_METHOD_MAP[method](x=x, axis=-1, **kwargs)

# Random field theory stuff
# -------------------------
def estimate_roughness_rft(x, y, axis=-1, dt=1): 
    n = x.shape[axis] 

    x = standardize(x, axis=axis)
    y = standardize(y, axis=axis)

    u = np.diff(x, axis=axis) 
    v = np.diff(y, axis=axis)

    return np.sum(u**2 + v**2, axis=axis) / (n * dt**2)

def estimate_roughness_fft(x, y, axis=-1, dt=1): 
    x = standardize(x, axis=axis)
    y = standardize(y, axis=axis)

    Sx = np.fft.fft(x, axis=axis)
    Sy = np.fft.fft(y, axis=axis)

    f  = np.fft.fftfreq(x.shape[axis]) / dt

    return np.sqrt(np.mean(f**2 *(Sx + Sy)))

_RG_METHOD_MAP = {
    "rft": estimate_roughness_rft, 
    "fft": estimate_roughness_fft, 
}

def estimate_roughness(x, y, method='rft', axis=-1, dt=1): 
    return _RG_METHOD_MAP[method](x, y, axis=axis, dt=dt)


# Effective degrees of freedoms estimators
# ----------------------------------------
def edfs_factor_bartlett(acf_x, acf_y, axis=-1): 
    """ 
    Bartlet: AR(1), edfs: n * (1 - r**2) / (1 + r**2) 
    """
    rx = np.take(acf_x, 1, axis)
    ry = np.take(acf_y, 1, axis)

    return  (1 - rx * ry) / (1 + rx * ry)

def edfs_factor_quenouille(acf_x, acf_y, axis=-1): 
    return 1 / np.sum(acf_x * acf_y, axis=axis)

def edfs_factor_bh(acf_x, acf_y, axis=-1):
    n  = acf_x.shape[axis] // 2 + 1

    rx = np.take(acf_x, range(1, n), axis)
    ry = np.take(acf_y, range(1, n), axis)

    return 1 / (1 + (2./n) * np.sum( (n - np.arange(1,n)) * rx * ry , axis))

def edfs_factor_ws(acf_x, acf_y, axis=-1): 
    n  = acf_x.shape[axis]//2
    rx = np.take(acf_x, range(n+1), axis)
    ry = np.take(acf_y, range(n+1), axis)

    Kx = toeplitz(rx)
    Ky = toeplitz(ry)

    return n / np.trace(Kx @ Ky)

def edfs_factor_rft(x, y, axis=-1, kernel='gaussian',  method='rft'): 
    w = estimate_roughness(x, y, axis=axis, method=method)

    if kernel == 'gaussian':
        a = np.sqrt(w / (2 * np.pi))
    elif kernel == 'sinc': 
        a = np.sqrt(3 * w) / (2*np.pi)
    elif kernel == 'ar(1)': 
        a = np.sqrt(w) / 2

    return a

# Resampling
# ----------
def phrndsurr(x, n, axis=-1): 
    x = np.moveaxis(x, axis, -1)

    Pxx = np.abs(np.fft.fft(x, axis=-1))
    Pxx = Pxx.reshape([1, *Pxx.shape])

    phi = np.random.uniform(0, 2*np.pi, (n, *x.shape[:-1], x.shape[-1]//2))
    phi = np.concatenate([phi, phi[..., ::-1]], axis=-1)
    phi = np.exp(1j * phi)

    z = np.real(np.fft.ifft(phi * Pxx))
    z = np.moveaxis(z, -1, axis)

    return z

def bootstrap_p(metrics, x, y, n, axis=-1, return_samples=False):
    Xs = phrndsurr(x, n, axis=axis)
    Ys = phrndsurr(y, n, axis=axis)

    Zs = metrics(Xs, Ys, axis=axis)
    Zr = metrics(x, y, axis=axis)
    Zr = Zr.reshape([1, *Zr.shape])

    p = 2*min(np.count_nonzero(Zs < Zr, axis=0), np.count_nonzero(Zs > Zr, axis=0)) / n

    if return_samples:
        return p, Zs, Zr
    else:
        return p


# Simulating random processes
# ---------------------------
def correlated_noise(n: int, m=1, r=1, dt=1, kernel='gaussian'):
    t = np.linspace(-n//2, n//2, n+1, endpoint=True) * dt 

    if kernel == 'gaussian': 
        k = np.exp(-t**2 / (2 * r**2))
    elif kernel == 'absolute': 
        k = np.exp(-np.abs(t) / r)
    
    w  = np.random.normal(0, 1, (m, n)) 

    fn = lambda x: scipy.signal.convolve(x, k, mode='same')
    x  = np.apply_along_axis(fn, axis=-1, arr=w) 

    x  = standardize(x, axis=-1)
    x  = np.squeeze(x)

    return x




# def correlated_noise(N: int, r, S=1, dt = 1, M = 1, repeat=-1, kernel='gaussian'): 
    
#     if kernel == 'gaussian': 
#         k = lambda u: np.exp(-u**2 / (2 * r**2))
#     elif kernel == 'absolute': 
#         k = lambda u: np.exp(-np.abs(u) / r)
#     elif kernel == 'sinc': 
#         k = lambda u: np.sinc(r * u)
#     elif kernel == 'sinc2': 
#         k = lambda u: np.sinc(r * u)**2

#     K  = toeplitz(k(t))
#     K  = K @ np.diag(1. / np.sqrt( np.diag(K @ K)))

#     if S == 1:
#         if repeat > 0: 
#             for i in range(repeat):
#                 yield (np.random.randn(M, N) @ K).squeeze() 
#         else:
#             while True:
#                 yield (np.random.randn(M, N) @ K).squeeze()    
#     else: 
#         P  = np.eye(M) * 1./np.sqrt(S)
    
#         if repeat > 0: 
#             for i in range(repeat):
#                 yield (P @ np.random.randn(M, N) @ K).squeeze() 
#         else:
#             while True:
#                 yield (P @ np.random.randn(M, N) @ K).squeeze()

