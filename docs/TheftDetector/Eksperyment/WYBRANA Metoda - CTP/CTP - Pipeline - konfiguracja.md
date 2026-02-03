### 1) Standardowy trial w CTP — kolejność i czasy
- **Czas S1:** 300 ms — 500 ms. (Rosenfeld: używane w opisie CTP). 
    Najlepiej: 350–400 ms (krótsze niż 500 ms ogranicza ryzyko ruchów, ale pozostawia wystarczająco długo by rozpoznać obraz).
- **ISI:** 1000–1500 ms (czasami do 2000 ms).
    Najlepiej: 1000–1500 ms (mniej czasu = krótsza sesja, ale trzeba zachować odstęp, żeby P300 z S1 się ujawnił).
- **Czas S2:** ~300 ms.
- **Target/nontarget stosunek (dla S2):** preferowany **~20% target / 80% nontarget** (tradycyjnie w oddball tasks rzadki target wywołuje P300.
[groups.psych.northwestern.edu](https://groups.psych.northwestern.edu/rosenfeld/documents/CTPPsychophysiology2008.pdf)

### 2) Ile powtórzeń (liczba triali)
- **Probe per participant:** 60–100 powtórzeń (im więcej, tym lepiej; jeśli sesje mogą być długie, 60–80 minimum).
- **Irrelevants:** co najmniej **3–6** różne irrelevants w danej kategorii, każdy powtarzany 40–100 razy (żeby łączna pula irrelevants była co najmniej ~3–5× większa od liczby probe-epok, ułatwiając bootstrap).
- Cel: **łączna liczba epok S1** ~500–1000, co daje sensowny SNR przy niskobudżetowej opasce (ale rozbij na bloki i rób przerwy).
- Użyj jedno zdjęcie (jeden widok) irrelevants i probe (powtarzające się).

### 3) Przykładowy kompletny pipeline
**Trial start**
  - (Fixation cross) 500 ms  — zalecane, by ustabilizować wzrok
  - S1 (probe lub irrelevant) — prezentacja 350–400 ms (obraz)
  - Odpowiedź na każdy S1 - 'Z' na klawiaturze
  - ISI (pusty ekran/fixation) 1000–1500 ms (losowo)
  - S2 (target/nontarget) — prezentacja 300 ms (np. numer/string)
  - Odpowiedź na S2: target -> naciśnij przycisk 'M' na klawiaturze, nontarget -> naciśnij przycisk 'N'
  - Krótki ITI (inter-trial interval) 500–800 ms
  - (co pewien czas, np. 5–10% triali): niespodziewany quiz o treść S1 (pop-quiz) — pytanie o to, który element był S1; to wymusza utrzymanie uwagi
**Trial end**