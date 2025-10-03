Z Paper'u: Item Roles Explored in a Modified P300-Based CTP Concealed Information Test
https://pmc.ncbi.nlm.nih.gov/articles/PMC6685925/

## 1. Wstępne przygotowanie sygnału
- **Zmiana częstotliwości próbkowania**: dane zostały przeskalowane do **250 Hz** – obniżenie sampling rate redukuje ilość danych i szum, przy zachowaniu wystarczającej rozdzielczości czasowej do analizy ERP.
- **Filtracja pasmowoprzepustowa (bandpass filter)**: filtr o pasmie **0.1–12 Hz** (lub 0.3-30 Hz; lub 0.1-30 Hz; lub 0.3-12 Hz) (tłumienie −6 dB) - P300 ma energię głównie w zakresie **1–8 Hz**.
    - Dolne odcięcie (0.1 Hz) usuwa dryf niskoczęstotliwościowy (artefakty np. z ruchu czy potu).
    - Górne odcięcie (12 Hz) eliminuje wysokoczęstotliwościowy szum mięśniowy i zakłócenia.

---
## 2. Segmentacja (epoching)
- **Wyciąć epoki czasowe −200 do 800 ms względem bodźca**.
- **Korekcja bazowa**: odjąć średnią aktywność z okresu **−200 do 0 ms** (baseline), aby wyeliminować różnice w poziomie napięcia przed bodźcem.
Po co korekcja?
- EEG ma naturalne wahania napięcia (drift) niezwiązane z bodźcem.
- Odjęcie średniej wartości przedbodźcowej wyrównuje poziom napięcia w każdej epoce, co umożliwia porównanie odpowiedzi ERP między różnymi epokami i uczestnikami.

---
## 3. Odrzucanie epok z dużymi amplitudami
- Epoki o amplitudzie przekraczającej **±100 μV** na elektrodach
- Po odrzuceniu średnia liczba epok na kategorię (probe/irrelevant) wynosiła ok. **27,55 ± 2,89** na uczestnika - **sprawdź**

---
## 4. Ręczne sprawdzenie pozostałych epok
- Jeśli duże artefakty pozostały

---
## 5. Brak sygnału / przerwanie pomiaru
- Jeśli w trakcie triala sygnał spada do zera lub jest wyraźnie zakłócony (artefakty hardware’owe, odłączenie opaski):
    - **Usunąć całe epoki**, w których nie ma sygnału.
    - Nie próbujemy interpolować brakującego sygnału – utrata jednej epoki jest mniej szkodliwa niż wprowadzenie fałszywego P300.

---
## 6. Średnia P300 dla probe vs irrelevants
- Średnie ERP dla probe i dla każdego irrelevant (na kanałach TP9/TP10, ew. AF7/AF8 jako uzupełnienie).
- P300 mierzyć w oknie **300–600 ms** (peak-to-peak lub średnia amplituda).