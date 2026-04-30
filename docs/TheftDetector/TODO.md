- [x] Dodaj opcję w ERP Analysis - Peak-to-Peak (Peak-Valley) - Szuka po prostu **najwyższego (Max)** oraz **najniższego (Min)** punktu w całym wyznaczonym oknie czasowym (np. 300–800 ms), ignorując to, w jakiej kolejności wystąpiły. Odejmuje Min od Max.

- [x] Dodaj opcję możliwości wyznaczania indywidualnego okna P300:
**Jak to zrobić krok po kroku:**
1. Dla każdego uczestnika wyodrębnij sygnał wokół markera `S2_onset` (tylko dla Targetów).
2. Uśrednij te triale (wygenerujesz klasyczne P300 na zadanie ruchowe/uwagowe).
3. Znajdź czas maksymalnego szczytu (peaku) tego P300 na kanale Pz lub Cz (np. wypadnie w 420 ms).
4. Na podstawie tego czasu wyznacz zindywidualizowane okno czasowe dla zadania S1 (np. szczyt z S2 $\pm$ 150 ms, czyli od 270 ms do 570 ms).
5. Mając to wyznaczone okno, przesuń się na epoki `S1_onset` i uruchom swój algorytm Bootstrap (Peak-to-Peak) dokładnie w tym indywidualnie policzonym przedziale.

- [x] Próg odrzucania artefaktów (Artifact Rejection Threshold) - dodaj opcję automatycznego dostosowania do uczestnika:
	**Użyj zautomatyzowanego algorytmu, np. biblioteki `autoreject` w Pythonie.** * _Jak to działa:_ Zamiast wpisywać ręcznie "100 µV", przekazujesz wycięte epoki do pakietu `autoreject` (który bezbłędnie integruje się z MNE). Algorytm analizuje wariancję sygnału dla każdego uczestnika za pomocą uczenia maszynowego (cross-validation) i sam oblicza optymalny próg odcięcia uV, który zbalansuje liczbę odrzuconych artefaktów i zachowanych "dobrych" triali. Jest to metoda w pełni obiektywna.

- [x] Dodaj opcję filtracji dolnoprzepustowej (np. 6 Hz) w celu wygładzenia przed znalezieniem peaków (w analizie ERP).

- [ ] Dodaj opcję - wyboru ostatecznego kanału (Channel Selection):
- _Problem:_ Ustaliłaś, że analizujesz kanał Pz. Ale u badanego nr 4 elektroda na Pz akurat słabo przylegała do głowy (np. przez gęste włosy) i na tym kanale masz sam szum, podczas gdy kanał Cz wygląda pięknie. Czy możesz dla tego jednego uczestnika zmienić kanał do statystyki na Cz?
- _Rozwiązanie:_ Tak, ale **tylko jeśli zrobisz to według z góry założonej, ślepej reguły.**
- _Jak to działa:_ Wykorzystujesz zadanie S2 (Target/Nontarget). Piszesz w kodzie prosty warunek: program oblicza tzw. SNR (Signal-to-Noise Ratio) dla P300 na S2 na kanale Pz i na Cz. Następnie do analizy ważnych bodźców S1 (Probe/Irrelevant) automatycznie wybiera ten kanał, który miał wyższe SNR na zadaniu S2. Algorytm decyduje sam w sposób obiektywny.

- [x] Podsumowywując: Zrób tak aby była możliwość wybrania np. takiego pipelineu:
1. Wczytanie `.fif`
2.  Nałożenie filtru Aggressive (data rescue).
3. Wycięcie epok S2 (Target).
4. Automatyczne obliczenie progu `autoreject` dla epok S2 i wyrzucenie szumu.
5. Zlokalizowanie maksymalnego szczytu na uśrednionym S2 (np. na Pz i Cz) -> Wyznaczenie indywidualnego okna czasowego (np. szczyt ± 150 ms) oraz wybór lepszego z tych dwóch kanałów.
6. Wycięcie epok S1 (Probe / Irrelevants).
7. Zastosowanie obiektywnego `autoreject` dla epok S1.
8. Filtracja dolnoprzepustowa (np. 6 Hz) w celu wygładzenia przed znalezieniem peaków.
9. Ekstrakcja wartości Peak-to-Peak (Peak-Valley) w indywidualnie wyznaczonym wcześniej oknie dla wylosowanych przez Bootstrap prób.
10. Zwrócenie wartości _p-value_ określającej winę/niewinność badanego.


- [x] Usuwanie całego triala (odpowiedzi S1), jeśli odpowiedź na powiązane z nim zadanie S2 była niepoprawna lub przekroczyła określony czas reakcji
- [x] To samo usuwanie epok S1 na podstawie niepoprawnych S2 trzeba również zaimplementować przed obliczaniem okna P300 z S2

- [x] Zmień quick pipeline tak aby była dodatkowa opcja batch upload, która pozwala na dodawanie dwóch zbiorów plików (innocent i guilty). Na końcu dodaj dodatkowe podsumowanie i metryki
- [x] Dodaj zakładkę z automatycznym obliczaniem najlepszych parametrów preprocessingu sygnału dla BAD (grid search, Optuna)

- [ ] Dodaj parametr k do rejection method w "Quick Pipeline" -> "Step 2 — S2 Target Epochs"

- [x] W przypadku filtrów IIR: mają one nieliniową odpowiedź fazową. Oznacza to, że mogą one "przesunąć" pik P300 na osi czasu, zaburzając latencję! Aby tego uniknąć, musisz upewnić się, że stosujesz filtrację w obu kierunkach przód-tył (tzw. _zero-phase filtering_, często zaimplementowane pod maską standardowych funkcji w bibliotece `mne` lub jako `filtfilt` w `scipy`). Dodatkowo wybór order filtra powinien być dyskretny (konkretnych liczb a nie z przedziału nawet typu int; i tylko parzystych wartości w wyborze).

- [x] **25 czystych epok Probe** - Jeśli po preprocessingu (po odrzuceniu zaszumionych epok (np. przez `autoreject`) i usunięciu epok z błędnymi odpowiedziami na zadanie S2) zostaje Ci mniej niż 25 epok ze skradzionym przedmiotem, sygnał P300 będzie zbyt mocno zniekształcony przez losowy szum tła. Takie nagranie należy w całości odrzucić. W takim przypadku przy uczeniu Optuny - odrzuć cały trial.

- [x] Zastosuj **Metrykę F-beta** - przy wyznaczaniu thresholdu w LOOCV (metoda BAD).

- [x] Standardowa korekta linii bazowej wykorzystuje okno od -200 ms do 0 ms. Proces ten polega na odjęciu średniej z tego wycinka od całej reszty epoki. Jeśli jednak w tych konkretnych 200 milisekundach wystąpi nagły skok napięcia (np. badany przełknie ślinę), po odjęciu tej "brudnej" średniej cała fala P300 sztucznie przesunie się w dół (w stronę wartości ujemnych). Algorytmy odrzucające, takie jak wykorzystywany `autoreject`, muszą bezwzględnie brać pod uwagę wariancję w samym oknie _baseline_, a nie tylko w oknie bodźca. Zaszumiona baza dyskwalifikuje całą epokę. Sprawdź czy tak jest (jeśli nie, to popraw to we wszystkich miejscach).

- [x] Dodaj, że jak wybieramy przy szukaniu parametrów dla metody BAD jedną z metod "Peak" - to optuna musi użyć smoothing (filtru).

- [x] Sprawdź czy użycie LOO (Leave-One-Out) CV - jest dobrą metodą do szukania parametrów preprocessingu dla BAD

- [ ] Dodaj metody ML (SVM, LDA)
- [ ] Dodaj szukanie hyperparametrów przez Optunę dla ML