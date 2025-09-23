Z Paper'u: Item Roles Explored in a Modified P300-Based CTP Concealed Information Test
https://pmc.ncbi.nlm.nih.gov/articles/PMC6685925/

## 1. Wstępne przygotowanie sygnału

- **Zmiana częstotliwości próbkowania**: dane zostały przeskalowane do **250 Hz** – obniżenie sampling rate redukuje ilość danych i szum, przy zachowaniu wystarczającej rozdzielczości czasowej do analizy ERP.
    
- **Filtracja pasmowoprzepustowa (bandpass filter)**: zastosowano filtr FIR z oknem Hamminga (Hamming-windowed sinc FIR) o pasmie **0.3–30 Hz** (tłumienie −6 dB).
    - Dolne odcięcie (0.3 Hz) usuwa dryf niskoczęstotliwościowy (artefakty np. z ruchu czy potu).
    - Górne odcięcie (30 Hz) eliminuje wysokoczęstotliwościowy szum mięśniowy i zakłócenia.

---
## 2. Segmentacja (epoching)
- Wycięto **epoki czasowe −100 do 1400 ms względem bodźca**.
- **Korekcja bazowa**: odjęto średnią aktywność z okresu **−100 do 0 ms** (baseline), aby wyeliminować różnice w poziomie napięcia przed bodźcem.
Po co korekcja?
- EEG ma naturalne wahania napięcia (drift) niezwiązane z bodźcem.
- Odjęcie średniej wartości przedbodźcowej wyrównuje poziom napięcia w każdej epoce, co umożliwia porównanie odpowiedzi ERP między różnymi epokami i uczestnikami.

---
## 3. Usuwanie artefaktów okoruchowych
- Artefakty ruchów oczu (EOG) usuwano metodą **adaptive Conventional Recursive Least Squares (CRLS)**, z wykorzystaniem 4 kanałów EOG.
- Metoda ta (Gómez-Herrero et al., 2006; Islam et al., 2016) adaptacyjnie usuwa wpływ mrugnięć i ruchów gałek ocznych z EEG.

---
## 4. Odrzucanie epok z dużymi amplitudami
- Epoki o amplitudzie przekraczającej **±75 μV** na elektrodach **Pz** lub mastoidach zostały odrzucone.
- Po odrzuceniu średnia liczba epok na kategorię (probe/irrelevant) wynosiła ok. **27,55 ± 2,89** na uczestnika.

---
## 5. Rereferencja i wybór kanału
- Elektroda **Pz** została **rereferencjonowana do zlinkowanych mastoidów** (M1 i M2) - bardzo popularne w badaniach P300/ERP,
- Do dalszych analiz użyto tylko **Pz** – jest to standardowe miejsce rejestracji P300, które charakteryzuje się największą amplitudą tego komponentu.