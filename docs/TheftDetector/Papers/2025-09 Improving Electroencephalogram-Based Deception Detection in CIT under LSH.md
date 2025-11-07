Badanie używa: prawdopodobnie **3-Stimulus Protocol (3SP)** + **BAD** (Bootstrapped Amplitude Difference); oraz bez CM

**Problem:** Tradycyjna metoda Statystycznej Różnicy Amplitudy (BAD, bootstrapped amplitude difference) jest mniej skuteczna w warunkach **niskiej heterogeniczności bodźców (LSH)**. Taka sytuacja ma miejsce, gdy **bodźce nieistotne (IR) niosą ze sobą znajomość lub wewnętrzne znaczenie dla podejrzanego**, co jest typowe dla realistycznych scenariuszy śledczych. Niska heterogeniczność zmniejsza różnicę amplitudy P300 między bodźcem dowodowym (PR) a bodźcem nieistotnym (IR), co negatywnie wpływa na dokładność metody BAD.

**Cel:** Badanie miało na celu zwiększenie zdolności wykrywania oszustwa w CIT opartym na P300, szczególnie w warunkach LSH, poprzez zastosowanie uczenia maszynowego (ML) i głębokiego uczenia (DL)

**Test Ukrytej Informacji (CIT):**
- W eksperymencie wzięło udział 67 zdrowych dorosłych, ostatecznie analizowano dane od 60 uczestników (30 w grupie winnej, 30 w grupie niewinnej).
- Przeprowadzono symulację przestępstwa (mock crime mission) trwającą około 30 minut, a następnie CIT (90 minut).
- Scenariusz LSH: Scenariusz pozorowanego przestępstwa został zaprojektowany tak, aby stworzyć warunki niskiej heterogeniczności. Uczestnicy z obu grup mieli styczność ze wszystkimi bodźcami, które miały być użyte w CIT, z wyjątkiem Bodźca Docelowego (TR).
- Uczestnicy winni zostali poinstruowani, by potajemnie ukraść przedmiot z określonego pudełka;
- Sygnały EEG rejestrowano z 30 elektrod (system 10–20) z częstotliwością próbkowania 1024 Hz.
- Dane zostały przefiltrowane (filtr pasmowo-zaporowy 59–61 Hz, pasmowo-przepustowy 0.1–50 Hz) i podzielone na epoki od -200 ms do 1500 ms po bodźcu.
- Do analizy BAD i ML użyto 46 segmentów na typ bodźca dla P300, oraz 44 epoki na typ bodźca dla cech ML.

**Kluczowe wyniki:**
1. **Zwycięska Metoda:** Najwyższą dokładność wykrywania oszustw osiągnął model Głębokiego Uczenia (DL) **EEGNet**.

2. **Dokładność:** EEGNet, wykorzystujący **nowatorską strategię augmentacji danych**, uzyskał dokładność **86,67%**. To najwyższy wynik klasyfikacji w tym badaniu.

3. **Wydajność w LSH:** Model DL przewyższył metodę **BAD** (która osiągnęła 73,5% dokładności) o ponad 13%. W warunkach LSH metoda BAD była szczególnie słaba w identyfikacji uczestników winnych, osiągając zaledwie **50% czułości**. EEGNet był w stanie skutecznie sklasyfikować prawie wszystkich uczestników niewinnych (**97,00% specyficzności**) i wykazał znacznie wyższą czułość dla grupy winnej (**76,67%**).

4. **Znaczenie Augmentacji:** Augmentacja danych okazała się kluczowa, ponieważ poprawiła dokładność EEGNet z 81,67% do 86,67%, redukując ryzyko przeuczenia przy ograniczonym rozmiarze zbioru danych.

5. **Inne Cechy ERP:** Oprócz składowej P300, analiza sugeruje, że składowa **N200** również może być ważną cechą w wykrywaniu oszustw w CIT, co było uwzględniane w modelach Uczenia Maszynowego (ML).

**Wnioski i Perspektywy:**
Model EEGNet z augmentacją danych jest obiecującym narzędziem do wykrywania subtelnych reakcji poznawczych w rzeczywistych warunkach śledczych. Przyszłe prace powinny koncentrować się na **opracowaniu modeli z mniejszą liczbą elektrod** w celu zastosowań terenowych oraz na wykorzystaniu **Wyjaśnialnej Sztucznej Inteligencji (XAI)**, aby zweryfikować decyzje klasyfikacyjne i zwiększyć wiarygodność DL w kontekstach śledczych.