## Opis
Metoda BAD (Bootstrap Amplitude Difference) to **nieparametryczny test statystyczny** stosowany w analizie P300 w eksperymentach typu Concealed Information Test (CIT).  
Jej celem jest sprawdzenie, czy odpowiedzi na bodźce **probe** (znane sprawcy) różnią się od odpowiedzi na bodźce **irrelevant** (neutralne, nieznane sprawcy).  

---
## Pipeline

1. **Preprocessing sygnału EEG**  
   - [[Preprocessing]] lub [[Preprocessing - Muse]]

2. **Wybór okna analizy**  
   - Okno czasowe: 300–600 ms po bodźcu.  
   - Dla każdej epoki obliczamy **średnią amplitudę** z całego okna.  
   - Otrzymujemy zestaw wartości:  
     - Probe amplitudes = [p1, p2, ..., pn]  
     - Irrelevant amplitudes = [i1, i2, ..., im]  

3. **Bootstrap**  
   - Losowo próbkuj z powtórzeniem z listy probe i z listy irrelevant.  
   - Oblicz średnią amplitudę dla obu grup.  
   - Zanotuj, czy `mean(probe) > mean(irrelevant)`.  
   - Powtórz procedurę np. 1000 razy.  

4. **Statystyka końcowa**  
   - Oblicz proporcję iteracji, w których `probe > irrelevant`.  
   - Jeśli proporcja ≥ 90% → uczestnik uznany za „guilty”.  
   - Jeśli proporcja ≈ 50% → brak dowodu na rozpoznanie (uczestnik „innocent”).  


**Podsumowując:**
Obliczamy średnie lub peak-to-peak amplitudy z sygnałów które zmierzyliśmy 'na każdym pokazanym zdjęciu' (zazwyczaj bierzemy 300-600 ms od chwili pojawienia się zdjęcia).
Potem bootsrtapujemy (np. 1000 razy):
- Losowo wybieramy średnie z epok 'probe' (czyli tych na których było zdjęcie kradzionego przedmiotu) i 'irrelevant' (zdjęć 4-5 pozostałych przedmiotów)
- Obliczamy różnice tych średnich: `diff_k = mean_probe - mean_irrelevant`
- Sprawdzamy czy `diff_k > 0` (czy probe > irrelevant)
- Obliczamy odsetek tych iteracji (`p_bootstrap`) w których `diff_k > 0`
- Jeśli `p_bootstrap >= 0.90` -> klasyfikacja = guilty (czyli uczestnik wykazał znaczącą wartość P300 do probe)
- Jeśli `p_bootstrap < 0.90` -> klasyfikacja = innocent / niepewna

---
## ## Dlaczego bierzemy średnią z całego okna 300–600 ms?

1. **Stabilność sygnału** – P300 nie zawsze osiąga maksimum w tym samym momencie u wszystkich uczestników. Średnia z całego okna „łapie” efekt mimo indywidualnych różnic w latencji, co zmniejsza ryzyko, że wybór pojedynczego piku będzie zależny od szumu lub przypadkowego przesunięcia czasowego

2. W literaturze istnieją różne podejścia:  
   - część badań stosuje **peak-to-peak** lub **baseline-to-peak**
   - inne podkreślają zalety średniej amplitudy w określonym oknie czasowym, zwłaszcza gdy fala P300 ma zróżnicowany kształt lub wiele szczytów
   W badaniach nad protokołem CIT zarówno **mean amplitude**, jak i **peak-to-peak** były stosowane i porównywane. Wyniki sugerują, że mean amplitude bywa stabilniejszym wskaźnikiem, gdy dane są zaszumione.

**Podsumowanie:**  
Średnia z całego okna 300–600 ms jest częściej uznawana za stabilniejszy i bardziej praktyczny wskaźnik w warunkach o niższym stosunku sygnału do szumu (low-density EEG). Jednak metoda peak-to-peak również bywa stosowana i w pewnych warunkach (np. wyraźne P300, dobre SNR) może dawać silniejsze efekty. Dlatego wybór metody należy uzasadniać charakterystyką sprzętu i celu badania

