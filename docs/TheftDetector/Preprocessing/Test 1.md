1. **Band-pass filter:** 
   0.1 - 30 Hz
2. **Epoching:** 
   -200 ms - 800 ms (lub 1000 ms)
3. **Baseline correction:**
   Okres od -200 ms do 0 ms
4. **Artifact Rejection:**
   a) Peak-to-Peak Amplitude Rejection: 75 µV do 100 µV
5. **Wybór ostatecznego kanału do analizy:**
   Głównym kanałem do analizy statystycznej powinien być **Pz**, ewentualnie uśredniony sygnał z Pz i Cz. Kanał Fz w tym konkretnym montażu będzie zdominowany przez bliskość referencji (Fp1) oraz potencjalne resztkowe artefakty oczne, dlatego posłuży nam bardziej jako kanał kontrolny do identyfikacji mrugnięć niż do ostatecznej analizy.
6. **Metoda BAD (Bootstrapped Amplitude Difference)**
   W testach CIT rzadko polegamy na klasycznej analizie wariancji (ANOVA), ponieważ zależy nam na diagnozie dla _pojedynczej osoby_ (intra-individual analysis), a rozkłady często nie są normalne. Zastosujemy metodę Bootstrappingu:
	- Wybieramy okno czasowe dla P300, zazwyczaj **300 ms – 600 ms** (lub dopasowane do uczestnika).
	- Obliczamy średnią amplitudę w tym oknie dla każdego pojedynczego triala S1.
	- Używamy iteracyjnego losowania ze zwracaniem (np. 10 000 iteracji): losujemy próbki z puli Probe i puli Irrelevant, obliczając różnicę ich średnich.
	- Budujemy rozkład różnic i sprawdzamy, czy w co najmniej 90% przypadków amplituda dla Probe jest istotnie statystycznie większa niż dla Irrelevant.
