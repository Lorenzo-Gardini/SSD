from numpy.fft import fft, fftfreq
from numpy import argsort, arange, absolute, angle, zeros, cos, pi, array, sin


class FourierRegressor:
    def __init__(self, h_harm=10):
        self._signal = array([])
        self._n_harm = h_harm

    def fit(self, signal):
        self._signal = signal
        return self

    def forecast(self, steps):
        n = self._signal.size
        freq_dom = fft(self._signal)
        f = fftfreq(n)  # frequencies
        # sort indexes by frequency, lower -> higher
        indexes = argsort(absolute(f))

        t = arange(0, n + steps)
        amplitudes = absolute(freq_dom) / n  # amplitudes
        phases = angle(freq_dom)  # phases
        restored_sig = zeros(t.size)

        for i in indexes[:self._n_harm]:
            restored_sig += amplitudes[i] * cos(2 * pi * f[i] * t + phases[i])
        return restored_sig[-steps:]
