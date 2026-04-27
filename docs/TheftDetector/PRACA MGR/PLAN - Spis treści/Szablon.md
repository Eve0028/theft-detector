1. Wstęp
	- O kryminalistyce, metodach wykrywania kłamstw, przejście do EEG (krótkie opisanie co i jak), do metod CTP/CIT bazujących na ERP (krótkie opisanie co i jak).

2. Cel i zakres pracy
	- Opisanie celu pracy (ocena skuteczności niskobudżetowego niskokanałowego sprzętu/czepka eeg w wykrywaniu informacji ukrytej - kłamstwa/kradzieży).
	- Zakres pracy - krótkie wprowadzenie to co zostało zrobione czyli: zebranie grupy 14 osób do badania, wykonanie eksperymentu typu „Mock Crime” (krótki opis) - razem z sesją eeg, preprocessing danych, podział na chunki, wykonanie testu statystycznego BAD oraz dodatkowo jako osobny klasyfikator wykorzystanie uczenia maszynowego (SVM i LDA - na ręcznie wyekstrachowanych cechach), itp.

3. Detekcja informacji ukrytej
	- Ewolucja metod detekcji kłamstwa.
	- Przejście do metod EEG (CIT/CTP) - ale bez szczegółów o EEG i historii/detalach CIT/CTP (będzie opisane w następnym rozdziale) - napomknięcie o badaniach w kierunku tych metod, dodanie cytowań.

4. Charakterystyka sygnału EEG i technologii pomiarowej
	- Opis czym jest EEG, jak je mierzymy, jakim sprzętem.
	- Fizyka i biomechanika sygnału EEG.
	- Porównanie systemów medycznych i konsumenckich (do przykładu BrainAccess).
	- Specyfikacja platformy BrainAccess Mini (co i jak możemy zmierzyć, ograniczenia).

5. ERP (event-related potential)
	- Czym jest ERP (jak je mierzymy itp.).
	- Czym jest P300/N200 itp.

6. Historia CIT/CTP
	- Ewolucja metod detekcji kłamstwa jeśli chodzi o wykorzystanie EEG, ich wady, zalety, różnice.
	- Na jakie markery ERP się patrzy (P300, N200 itp.) i dlaczego.
	- Wyniki klasyfikacji różnych badań.
	- Astepkty etyczne i prawne (neuroprawo).
	- Tutaj dużo cytowań.

7. Metodologia badania
	- Wprowadzenie: zdecydowanie się na metodę CTP (podział na zadania S1 i S2...) - i dlaczego.
	- Użyty sprzęt BrainAccess wypożyczony był od Politechniki Krakowskiej (Wydziału Matematyki i Informatyki), całość badania była przerowadzona w pomieszczeniu w budynku na terenie PK (ul. Warszawska).
	- Wybranie użytych elektrod (Fz, Cz, Pz) - opis przepinania elektrod czepka (zmiana drogi kabelków itp.).
	- Grupa badawcza (charakterystyka), szukanie grupy odbyło się przez komunikator Teams na grupie wydziałowej. Chętni zostali poinformowani o ogólnym opisie badania EEG, wymaganiach (wykluczenie implantów w głowie, ran na skórze głowy, zażywania leków psychotropowych itp); zebrani uczestnicy to studenci PK (ilość: 14). Dopisanie innych info: rozkład wieku, płci itp.
	- Opis przedmiotów użytych w badaniu - pluszaki (testy jakości sygnału wykazywały kiepski SNR i postanowiono użyć przedmiotów, które "podwyższą" segment P300, bardziej wpłyną na uwagę, będą bardziej charakterystyczne i różne pomiędzy sobą itp.); przedstawienie ich na zdjęciach; kradziony pluszak to "wilk". Opis wykonania zdjęć (jednolite tło itp.).
	- Każdy był badany osobno w innych godzinach/terminie. Czas badania to około 1h-1.5 (przygotowanie ok. 15-30min; sesja ok. 30min; tłumaczenie wstępnych wyników/procedur itp., ankiety ok. 15-30min).
	- **Struktura badania**:
		- Przed badaniem uczestnicy musieli się przygotować: nie spożywać kofeiny (min. 3h przed badaniem), bez makijażu czy kremów na czole, umyć skórę głowy dzień przed lub w dniu badania. 
		- Przygotowanie stanowiska (laptop, program, czyszczenie elektrod, przygotowanie roztworów czyszczących), podpisanie zgód RODO, wypełnienie ankiety wstępnej, wprowadzenie do badania (jak działa eeg itp.).
		- Jeśli osoba była z grupy "złodzieji" to przeprowadzenie "mock crime" (opis eksperymentu).
		- Przygotowanie linii środkowej głowy, przedziałku (dla dobrego przylegania ząbkowych elektrod Fz, Cz, Pz).
		- Przeczyszczenie czoła oraz elektrod izopropanolem i SigmaSpray'em.
		- Założenie czepka.
		- Sprawdzenie sygnału/impedancji na firmowym oprogramowaniu BrainAccess.
		- Instruktaż dla uczestnika co i jak mierzymy, pokazanie fal alpha, wytłumazenie artefaktów od mrugnięć/ruchów, prośba o odpowiednie zachowanie podczas sesji itp.
		- Instruktaż sesji eeg, klawiszy i sesja próbna (1-2min).
		- Odpalenie sesji głównej (5 bloków po 80 prób, kontrola przerw i stanu badanego, badany miał do dyspozycji butelkę wody).
		- Zdjęcie sprzętu, wypełnienie ankiety końcowej i debriefing.
		- Pokazanie wstępnych wyników na aplikacji Streamlit (stworzonej do wizualizacji) i wytłumaczenie całego pipeline'u preprocessingu, podziału na epoki oraz metody BAD (klasyfikacji).
	- Opis sesji pomiarowej/eeg - szczegóły, cały pipeline (zdjęcia/screeny) - schemat.
		[[CTP - Pipeline sesji - końcowy]]
	- Sesja pomiarowa/eeg - kod i strona techniczna, LSL, zapis do formatu .fif, zbieranie markerów, wykorzystanie anonimowych identyfikatorów dla uczestników, itp.
	- Opis ankiet wstępnej i końcowej (rozkład odpowiedzi tutaj?) - zostały uzupełniane na kartce aby nie zwiększać uczestnikom czasu przed ekranem.
	- Odrzucanie danych osób, które nie rozpoznały przedmiotu kradzieży (z grupy złodzieji) - dane z ankiety.

8. Pipeline i preprocessing danych:

	1. Opis, że bazujemy tylko na P300 nie włączamy N200 ze względu na:
		- Różnice w N200 pomiędzy winnymi a niewinnymi mogą czasami wynikać ze słabego doboru samych bodźców (różnic w ich fizycznym/wizualnym wyglądzie) - dobrane pluszaki (ich zdjęcia) mogłyby zakłamać wynik;
		- Wzmocniona fala N200 w odpowiedzi na bodziec Probe najsilniej pojawia się u osób o tzw. "wysokiej świadomości", czyli tych, które są mocno skupione na fakcie ukrywania informacji. Z ankiet końcowych wynika, że uczestnicy nie starali się ukryć informacji (pomimo powiedzenia im, że mają starać się ukryć informację kradzieży) - nie przekazano im żadnych informacji na temat counter-measures, sami równiez ich nie używali. Moje badanie nie ma na celu sprawdzenie radzenia sobie z counter-measures.
	
	2. Opis preprocessingu:
		1. **Aggressive filter (data rescue):**
			- **Noth filter** - **50 Hz**.
			- IIR Butterworth **0.5–30 Hz** (order 4) with steep rolloff. Designed to remove large slow-wave drift while preserving the P300 band. Wytłumaczyć dlaczego używamy takiego agresywniejszego filtra i dodać zdjęcie sygnału.
			- Wytłumaczenie na jakich częstotliwościach pojawia się P300, i dlaczego inne filtrujemy.
		2. **Epoching:** 
			- 200 ms - 1000 ms.
			- Wytłumaczyć co i jak, dlaczego się to stosuje (uśrednienie itp.). Usuwanie całego triala (odpowiedzi S1), jeśli odpowiedź na powiązane z nim zadanie S2 była niepoprawna lub przekroczyła określony czas reakcji
		3. **Baseline correction:**
			- Okres od -200 ms do 0 ms.
			- Wytłumaczyć co i jak, dlaczego się to stosuje.
		4. **Artifact Rejection:**
			- W Python możemy użyć algorytmu `autoreject` (dla mniej niż 4 kanałów, czyli w naszym przypadku: `get_rejection_threshold` computes an optimal global µV threshold via Bayesian optimization, applies it, and shows the computed threshold value) - opis działania metody;
			- dodanie, że zazwyczaj stosowana jest metoda ICA ale w naszym przypadku (niskiej ilości elektrod) nie jest możliwa i dlaczego.
		5. **Wybór ostatecznego kanału do analizy:**
			- Głównym kanałem do analizy statystycznej powinien być **Pz**. Wytłumaczyć dlaczego, co i jak (występowanie P300).
		6. **Wybieramy okno czasowe P300** - dopasowanie do uczestnika na bazie zadania S2 - targetu:
			 1. Takie same parametry filtrów, preprocessingu i epochingu robimy na S2; 
			 2. Epoch smoothing (peak-based methods): 
				- Smoothing applied to individual epochs before peak-based amplitude extraction (zero-phase Butterworth low-pass, with smoothing_lowpass_hz = **12 Hz**); Wytłumaczenie dlaczego używamy smoothing.
			3. Finds the positive peak on a user-selected channel (default Pz) within a configurable search window
			4. Returns `peak ± margin` (0.15 s) as the individualized time window
		7. **Metoda BAD (Bootstrapped Amplitude Difference)**
			I. Epoch smoothing (peak-based methods): 
				- Smoothing applied to individual epochs before peak-based amplitude extraction (zero-phase Butterworth low-pass, with smoothing_lowpass_hz = **12 Hz**)
			II. Obliczamy średnią amplitudę w tym oknie P300 dla każdego pojedynczego triala S1.
			III. Używamy iteracyjnego losowania ze zwracaniem (np. 10 000 iteracji): losujemy próbki z puli Probe i puli Irrelevant, obliczając:
			- mean (średnią), - signed average in the P300 window - best for noisy data (random spikes cancel out);
			- peak-to-peak (Rosenfeld) - positive peak in P300 window, then negative trough after it - robust against baseline drift & CNV, but sensitive to single sharp artifacts;
			- peak-to-peak (Peak-Valley) - global max minus global min in the window, regardless of temporal order. Simple and order-agnostic;
			- baseline-to-peak - max positive amplitude relative to zeroed baseline.
			IV. Budujemy rozkład różnic i sprawdzamy, czy w co najmniej 80% przypadków różnica dla Probe jest istotnie statystycznie większa niż dla Irrelevant.
			V. Porównujemy wyniki wszystkich (4) metod/sposobów BAD.
		8. **Dodanie metod uczenia maszynowego: SVM i LDA (Liniowa Analiza Dyskryminacyjna)** - ze względu na niezbyt optymistyczne wyniki z BAD:
			- Model wewnątrzosobniczy (Intra-subject) - trenujemy model na reakcjach z zadania S2 (Target/Nontarget), a następnie używamy go do predykcji na zadaniu S1 (Probe/Irrelevant).
			- Cechy:
				- **Dla modelu SVM:** wektor złożony z **wyekstrahowanych współczynników falkowych** dla całego okna fali P300 (np. po zastosowaniu DWT/WPT dla pasm 0.5–16 Hz) - standaryzacja zbioru. Falki **Daubechies (np. db4 lub db8)** oraz **Symlets (np. sym4 lub sym8)**; Poziom dekompozycji: **5**.
				- **Dla modelu LDA:** ten sam **zestaw współczynników falkowych** oraz dorzucamy podstawowe statystyki (np. **stosunek amplitudy do latencji, peak-to-peak**), ale **stosujemy na nim algorytm selekcji cech (np. PCA, Stepwise LDA lub korelacje)**. Redukujemy zestaw z kilkudziesięciu/kilkuset parametrów falkowych do ok. 15-25 najsilniejszych cech i dopiero wtedy uczymy model LDA.
		9. Obliczenie wyników metryk.
	
9. Porównanie wyników wszystkich modeli + wyników innych badań.
   - Zestawienie wyników „guilty/innocent” w zależności od przyjętego progu ufności statystycznej dla wszystkich metod BAD, i AUC (dla ML).

10. Przedstawienie jakości pobranych danych i ograniczeń sprzętowych:
	- Przedstawienie statystyk, mean, std itp. pomiarów. Wykresy.
    - Kiepska jakość - pomimo starań: 
		- czyszczenie elektrod i czoła izopropanolem (izopropanol ok. 60/70% + woda destylowana),
	    - zmniejszenie impedancji SigmaSpray'em - przetarcie elektrod płaskich i czoła (opis co to jest i do czego),
	    - odpowiednie zaciśnięcie czepka (choć układ zapięć w samym czepku nie był najlepszy, kiepskie dopasowanie z tyłu głowy - ale zdecydowałam się nie zmieniać architektury czepka),
	    - minimalny ruch uczestników przy badaniu (+odpowiedni instruktaż),
	    - nawet na głowie bez włosów jakość była tak samo kiepska :')...
	- Porównanie czepka niskobudżetowego do profesjonalnego - krótka wzmianka (było wcześniej).
    - Przedstawienie innych możliwości wystąpienia kiepskiej jakości danych:
	    - Kiepskie metody pomiarowe?
	    - Inne podejście do eksperymentu "mock crime" (być może uczestnicy byli zbyt mało "wciągnięci" w kradzież -> zbyt mała motywacja do ukrywania informacji o kradzieży przedmiotu -> wpływ na komponent P300 - zbyt mało wyraźny).
    - Analiza błędów i korelacje z ankietą końcową - badanie wpływu deklarowanego zmęczenia, komfortu, czynników dystrakcyjnych itp.

11. Porównanie różnych wykresów dla grupy eksperymentalnej i kontrolnej:
	- Grand average (dla epok), porównanie S1/S2, itp.
	- Odpowiedzi z ankiet.
	
12. Architektura kodu w Pythonie? 

13. Wnioski dotyczące praktycznego wykorzystania BrainAccess Mini poza ścisłym rygorem szkolnym.