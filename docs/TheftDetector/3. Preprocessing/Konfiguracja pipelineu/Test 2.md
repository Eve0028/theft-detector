1. **Aggressive filter (data rescue):**
	a) **Noth filter** - **50 Hz**
	b) IIR Butterworth **0.5–30 Hz** (order 4) with steep rolloff. Designed to remove large slow-wave drift while preserving the P300 band.
2. **Epoching:** 
   -200 ms - 1000 ms
3. **Baseline correction:**
   Okres od -200 ms do 0 ms

4. **Artifact Rejection:**
   b) W MNE-Python możemy użyć algorytmu `autoreject` (dla mniej niż 4 kanałów, czyli w naszym przypadku: `get_rejection_threshold` computes an optimal global µV threshold via Bayesian optimization, applies it, and shows the computed threshold value)

5. **Wybór ostatecznego kanału do analizy:**
   Głównym kanałem do analizy statystycznej powinien być **Pz**

6. **Wybieramy okno czasowe P300** - dopasowanie do uczestnika na bazie zadania S2 - targetu:
	a) Takie same parametry filtrów, preprocessingu i epochingu robimy na S2; 
	a) Epoch smoothing (peak-based methods): 
		- Smoothing applied to individual epochs before peak-based amplitude extraction (zero-phase Butterworth low-pass, with smoothing_lowpass_hz = **12 Hz**) 
	b) Finds the positive peak on a user-selected channel (default Pz) within a configurable search window
	c) Returns `peak ± margin` (0.15 s) as the individualized time window

7. **Metoda BAD (Bootstrapped Amplitude Difference)**
	a) Epoch smoothing (peak-based methods): 
		- Smoothing applied to individual epochs before peak-based amplitude extraction (zero-phase Butterworth low-pass, with smoothing_lowpass_hz = **12 Hz**)
	b) Obliczamy średnią amplitudę w tym oknie P300 dla każdego pojedynczego triala S1.
	c) Używamy iteracyjnego losowania ze zwracaniem (np. 10 000 iteracji): losujemy próbki z puli Probe i puli Irrelevant, obliczając peak-to-peak.
	c) Budujemy rozkład różnic i sprawdzamy, czy w co najmniej 80% przypadków różnica dla Probe jest istotnie statystycznie większa niż dla Irrelevant.
