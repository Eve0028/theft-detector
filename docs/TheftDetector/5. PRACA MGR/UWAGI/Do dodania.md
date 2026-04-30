### Statystyczna walidacja i standardy kryminalistyczne
W zastosowaniach prawnych błąd polegający na fałszywym oskarżeniu niewinnej osoby (false positive) jest niedopuszczalny.[7, 8] Dlatego modele muszą spełniać rygorystyczne kryteria statystyczne.[3, 8]

Metryki oceny w CIT

| Metryka                     | Znaczenie w CIT                          | Wyniki referencyjne              |
| --------------------------- | ---------------------------------------- | -------------------------------- |
| AUC (Area Under Curve)      | Zdolność rozróżniania winny vs. niewinny | 0.85 – 0.92 [3]                  |
| Effect Size (d∗)            | Siła różnicy Probe vs. Irrelevant        | 1.59 (średnia), do 2.5 w CTP [3] |
| Czułość (Sensitivity)       | Wykrywalność osób posiadających wiedzę   | **85% – 100%** [4, 14]           |
| Specyficzność (Specificity) | Brak reakcji u osób niewinnych           | **90% – 100%** [6, 8]            |
Zgodnie ze **standardami Brain Fingerprinting**, diagnoza „Information Present” (Wiedza obecna) może zostać postawiona tylko wtedy, gdy statystyczna pewność (confidence level) przekroczy predefiniowany próg (np. 99%).[5, 8] Jeśli wynik nie spełnia tego kryterium, klasyfikowany jest jako „Indeterminate” (nieokreślony), co zdarza się w około 3% przypadków w badaniach terenowych FBI i CIA.[8]


### Liczba prób a stabilność modelu
Uczenie maszynowe na pojedynczych próbach (single-trial classification) jest ekstremalnie trudne ze względu na SNR bliski zeru.[22, 24, 31] Skuteczność modelu rośnie wraz z liczbą uśrednionych prób.[1, 10] Wskazuje się, że minimum 40–50 prób typu Probe jest niezbędne do uzyskania stabilnej reprezentacji fali P300, przy czym optymalna precyzja zostaje osiągnięta przy **około 60** próbach.[1] Model wewnątrzosobniczy musi zatem zebrać odpowiednią liczbę reakcji Target w zadaniu S2, aby móc rzetelnie ocenić bodźce S1.[1, 6]

### Filtr
- wybieramy tylko autoreject - iqr i z-score sa podatne na za bardzo zaszumione dane

### Rząd filtra iir i zakres
- iir order powinien być parzysty;
- stosowanie filtracji w obu kierunkach (tzw. _zero-phase filtering_) - aby uniknąć nieliniowej odpowiedzi fazowej ("przesunięcia" piku P300 na osi czasu);
- zakres możemy zacząć od 0.8 Hz - przez kompromis pomiędzy ratowaniem danych przed szumem (kiepski sprzęt) a poprawną rejestracją P300;


### Zbieranie danych - LSL
**Opóźnienie monitora i systemu (Hardware & Display Jitter):** Logujesz znaczniki czasu (markery) za pomocą protokołu LSL przy pojawieniu się bodźca S1 i S2. Pamiętaj jednak, że od momentu wysłania komendy "pokaż obraz" w skrypcie do faktycznego rozświetlenia pikseli na ekranie mija czas (zależny od częstotliwości odświeżania monitora, np. ok. 16 ms dla 60 Hz). Dodatkowo sama transmisja danych z czepka BrainAccess generuje minimalne opóźnienia.
- _Wskazówka do pracy:_ Warto krótko wspomnieć o tych opóźnieniach w ograniczeniach metody. Uzasadnij, że szerokie okno badawcze dla fali P300 (np. analiza w przedziale 300–600 ms po bodźcu) pozwala tolerować ten sprzętowy _jitter_ bez utraty kluczowych informacji.


### Odrzucanie epok na podstawie zaszumionej linii bazowej
Przy użyciu rozstępu międzykwartylowego IQR, k= 3?

W celu zachowania ścisłej spójności spektralnej i fazowej, epoki wyekstrahowane dla bodźca S2 zostały poddane identycznemu potokowi czyszczenia (w tym weryfikacji wariancji w oknie referencyjnym oraz globalnemu odrzucaniu artefaktów) jak epoki bodźca S1, przy użyciu tych samych hiperparametrów wyłonionych w procesie optymalizacji.