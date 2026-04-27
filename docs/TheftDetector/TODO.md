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


- [ ] Usuwanie całego triala (odpowiedzi S1), jeśli odpowiedź na powiązane z nim zadanie S2 była niepoprawna lub przekroczyła określony czas reakcji