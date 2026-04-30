1. Filtering:
	- High-pass (Hz) -> [0.1, 0.3, 0.5, 0.7]
	- Low-pass (Hz) -> [12.0, 15.0, 20.0, 30.0]
	- IIR order -> [2, 3, 4]

2. Epoching:
	- S1 tmin (s) -> [-0.2]
	- S1 tmax (s) -> [1.0]
	- S1 detrend method (DC offset / Linear) -> ['DC offset']
	- S1 rejection method (iqr / zscore / autoreject) -> ['autoreject', 'iqr']
	- Search adaptive k (IQR or Z-score sensitivity) -> [1.5, 2.0, 2.5, 3.0]

3. P300 Window:
	- P300 window mode: individual (based on S2) / static -> ['individual', 'static']
	- Individual window parameters (used when mode = individual):
		- Margin ± (s) -> [0.10, 0.15, 0.20]
		- Peak search start (s) -> [0.25, 0.30]
		- Peak search end (s) -> [0.60, 0.70]
		- S2 ERP LP cutoff (Hz) (with 'None' smoothing possibility) -> [6.0, 8.0, 10.0, 12.0, 'None']
	- Static window parameters (used when mode = static):
		- Static window start (s) -> [0.25, 0.30]
		- Static window end (s) -> [0.60, 0.70, 0.80]

4. CTP-BAD
	- Amplitude method -> ['mean', 'baseline-to-peak', 'peak-to-peak (Rosenfeld)', 'peak-to-peak (Peak-Valley)']
	- Smoothing - smoothing is always applied when choosen amplitude method type is 'Peak' -> ['None', 'low-pass (butterworth)']
	- Epoch smooth LP (Hz) -> [6.0, 8.0, 10.0, 12.0]


### Optuna implementation
#### Typy parametrów - przeszukiwanych

1. 'int':
	- lp_cutoff: 12 - 30, step=3
2. 'float':
	- Epoching:
		- adaptive_k: 1.5 - 3.0, step=0.5
	- P300 Window:
		- window_margin: 0.10 - 0.20, step=0.05
		- peak_search_start: 0.25 - 0.35, step=0.05
		- peak_search_end: 0.60 - 0.75, step=0.05
		- static_window_start: 0.25 - 0.35, step=0.05
		- static_window_end: 0.60 - 0.80, step=0.05
	- CTP-BAD:
		- epoch_smooth_lp_hz: 6.0 - 12.0, step=2.0
3. Pozostałe 'options'


### Metryka optymalizacyjna - F-beta

Metryka **F-beta** to elastyczna wersja popularnego F1-score. Łączy ona w sobie dwie wartości:
- **Precyzję (Precision):** Jak bardzo możemy ufać modelowi, gdy mówi "to jest złodziej"? (Maksymalizacja precyzji minimalizuje fałszywe alarmy - False Positives).
- **Czułość (Recall):** Jak skutecznie model wyłapuje wszystkich faktycznych "złodziei" z grupy? (Maksymalizacja czułości minimalizuje przeoczenia - False Negatives).

Parametr $\beta$ decyduje o tym, która z tych wartości jest dla nas ważniejsza:
- **$\beta = 1.0$ (F1-Score):** Precyzja i Czułość są równie ważne.
- **$\beta > 1.0$ (np. $\beta = 2.0$):** Czułość staje się ważniejsza. Karzemy model za przeoczenie złodzieja (przydatne np. w badaniach przesiewowych na raka, gdzie wolisz fałszywy alarm niż przeoczenie guza).
- **$\beta < 1.0$ (np. $\beta = 0.5$):** Precyzja staje się ważniejsza. Karzemy model za fałszywe oskarżenia (False Positives). Waga $\beta = 0.5$ sprawia, że Precyzja ma dwukrotnie większy wpływ na ostateczny wynik niż Czułość.


### Pipeline wyliczania 'threshold' i 'metryk końcowych' - LOO (Leave One Out) CV (wewnątrz triala optuny)

1. Podział: Z grupy 12 uczestników wyciągamy jedną osobę (Zbiór Testowy). Pozostaje nam 11 uczestników (Zbiór Treningowy).
2. Trening (Szukanie progu): Algorytm sprawdza wyniki metody BAD (np. wskaźnik max_proportion) dla tych 11 osób. Testuje 201 różnych progów odcięcia (np. próg 0.70, 0.75, 0.80...).
3. Wybór metryki optymalizacyjnej: Algorytm wybiera ten próg, który na tych 11 osobach daje najwyższą wybraną metrykę (F-beta).
4. Testowanie: Mając ten "wyuczony" próg (np. 0.82), aplikujemy go na tej 1 odłożonej osobie. Jeśli jej wynik BAD wynosi 0.88 (powyżej progu), klasyfikujemy ją jako "Złodzieja". Sprawdzamy, czy decyzja zgadza się z prawdą (uzyskujemy True Positive, False Positive, True Negative lub False Negative).
5. Rotacja: Zwracamy osobę do puli, wyciągamy kolejną i powtarzamy cały proces od nowa (kroki 1-4). Powstanie 12 unikalnych modeli z potencjalnie 12 różnymi progami.
6. Agregacja: Na koniec zliczamy wszystkie poprawne i niepoprawne klasyfikacje dla tych 12 niezależnych testów, obliczając ostateczną czułość, swoistość i dokładność całego Twojego pipeline'u.

##### Szczegółowo
#### Krok 1: Przetwarzanie sygnału i ekstrakcja cechy (BAD)
Dla danej iteracji Optuny (czyli dla konkretnego, wylosowanego zestawu filtrów):
- Algorytm bierze nagranie `.fif` pierwszej osoby (np. „złodzieja”).
- Filtruje je, dzieli na epoki, czyści.
- Uruchamia metodę BAD (1000 iteracji bootstrap). Metoda ta losuje epoki _Probe_ i _Irrelevant_ **tylko i wyłącznie z nagrania tej jednej osoby**. Zlicza, w ilu przypadkach amplituda P300 dla _Probe_ była większa. Wynikiem jest jedna, pojedyncza liczba: `max_proportion` (np. `0.85`).
- Skrypt powtarza to dla każdego z 12 uczestników niezależnie.
#### Krok 2: Zebranie wektora wyników
Po przetworzeniu wszystkich nagrań, skrypt ma 12 niezależnych liczb (skalarów).
- Tworzy z nich prosty wektor, np.: `scores = [0.85, 0.40, 0.92, 0.35...]`.
- Ma też wektor prawdziwych etykiet: `labels = [1, 0, 1, 0...]` (gdzie 1 to winny, 0 niewinny). Na tym etapie kończy się praca z surowym sygnałem EEG.
#### Krok 3: Klasyfikacja w pętli LOOCV
Teraz wkracza walidacja krzyżowa. LOOCV operuje **tylko na tych 12 liczbach**.
1. Wyciąga wynik pierwszej osoby (np. `0.85`) do testowania. Zostaje 11 wyników.
2. Na tych 11 wynikach algorytm testuje różne progi odcięcia (0.0 do 1.0), szukając takiego, który da najwyższą metrykę F-beta (optymalny próg `thr`).
3. Mając wyuczony próg, patrzy na wyciągnięty wynik `0.85` i ocenia: „Czy `0.85` jest większe lub równe progowi?”. Zapisuje trafienie (lub pomyłkę).
4. Pętla wraca do początku, wyciągając kolejną z 12 liczb, i powtarza proces.


