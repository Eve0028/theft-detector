### **Classification CIT** vs **Complex Trial Protocol (CTP)**

Oba protokoły wykorzystują potencjały wywołane zdarzeniem (ERP), zwłaszcza składową **P300** (lub późniejsze składowe, takie jak P300-MERMER), która jest indukowana, gdy rzadki i istotny bodziec (oddball) jest prezentowany badanemu.

### Paradygmat Classification CIT (Brain Fingerprinting – BFP)

Paradygmat Classification CIT (Classification Concealed Information Test) jest ściśle związany z techniką **Brain Fingerprinting (BFP)**, wprowadzoną przez Farwella i Donchina.
#### Cel i Metoda
Classification CIT jest oparty na założeniu, że u osoby posiadającej ukrytą wiedzę (_guilty_ lub _information-present_, IP), bodziec **Sonda (Probe, P)** – zawierający ukrytą informację – powinien wywoływać reakcję P300 podobną do reakcji na bodziec **Cel (Target, T)**.

• **Rola bodźców:**
- **Sonda (P):** Rzadki bodziec zawierający ukrytą informację, znaną tylko osobie zaangażowanej w incydent.
- **Cel (T):** Rzadki bodziec, którego znaczenie jest _jawnie_ określone w instrukcjach, co gwarantuje, że wywoła on duży P300 u wszystkich badanych.
- **Nieistotny (Irrelevant, I):** Częsty bodziec, który nie ma znaczenia. Stosunek P:T:I wynosi zazwyczaj 1:1:4 lub 1:1:X.

• **Zadanie behawioralne:** Badani są instruowani, aby **zliczać targets** (cele) i **ignorować probes** (sondy) oraz _irrelevants_ (nieistotne), albo naciskać ten sam przycisk w reakcji na P i I, a inny w reakcji na T. Bodziec T jest stosowany głównie jako element wymuszający uwagę.

• **Metoda analizy (BCD):** Klasyfikacja odbywa się za pomocą metody **Bootstrapped Correlation Difference (BCD)**. Metoda ta wykorzystuje **korelację krzyżową** (double-centered correlation) pomiędzy przebiegami fal ERP:
- Porównuje się korelację przebiegów P i T z korelacją przebiegów P i I.
- Jeśli **korelacja P-T jest wyższa niż P-I** w 90% lub więcej iteracji bootstrappingu, badany jest klasyfikowany jako **posiadający wiedzę** (IPC).

• **Mierzony komponent:** BFP często analizuje szerszą epokę, znaną jako **P300-MERMER** (Memory and Encoding Related Multifaceted Electroencephalographic Response), rozciągającą się od 300 ms do 1800 ms. Wykorzystanie całej morfologii fali, a nie tylko szczytowej amplitudy, ma zwiększać dokładność i pewność statystyczną.


### Paradygmat Complex Trial Protocol (CTP)

Complex Trial Protocol (CTP) został opracowany przez Rosenfelda i współpracowników jako **protokół drugiej generacji** P300-CIT. Głównym powodem jego stworzenia była **odporność na kontrmaneuwy** (Countermeasures, CMs), które okazywały się skuteczne w starszych protokołach P300-CIT (takich jak 3-Stimulus Protocol, 3SP, który był używany w początkowych badaniach Rosenfelda i strukturalnie przypomina Classification CIT w zakresie jednoczesnej prezentacji T, P i I).

#### Cel i Metoda

Kluczową innowacją CTP jest **czasowe rozdzielenie** zadania rozpoznawania ukrytej informacji od zadania dyskryminacji wymuszającej uwagę.

• **Struktura próby (Trial):** Każda próba CTP składa się z dwóch bodźców (S1 i S2) prezentowanych w odstępie około 1–1.5 sekundy:

- **Bodziec 1 (S1):** Prezentuje się **Sondę (P)** lub **Nieistotny (I)** bodziec
	- **Reakcja 1:** Badany wykonuje **proste potwierdzenie** (np. naciśnięcie przycisku "widziałem to"), bez podejmowania jawnej decyzji o istotności bodźca. Ma to na celu **minimalizację obciążenia kognitywnego** i zachowanie pełnych zasobów przetwarzania dla nieświadomego rozpoznania P.
- **Bodziec 2 (S2):** Prezentuje się **Cel (T)** lub **Nie-cel (Non-target, NT)** (często są to ciągi cyfr).
	- **Reakcja 2:** Badany wykonuje **jawną dyskryminację** (np. naciśnięcie lewego lub prawego przycisku) w celu utrzymania uwagi w trakcie całego protokołu.

• **Metoda analizy (BAD):** CTP zazwyczaj wykorzystuje metodę **Bootstrapped Amplitude Difference (BAD)**.
- Metoda ta koncentruje się na bezpośredniej różnicy **amplitudy P300** między _Probe_ (P) a _Irrelevant_ (I).
- Jeżeli 90% lub więcej powtórzonych, uśrednionych różnic P minus I jest pozytywnych (P > I), badany jest klasyfikowany jako posiadający wiedzę.

• **Odporność na CMs:** Badania wykazały, że CTP jest **odporny** na proste kontrmaneuwy fizyczne (jak ukryte naciskanie palcami) i mentalne, które łatwo osłabiały starsze protokoły CIT. Ta zwiększona odporność jest kluczowym argumentem przemawiającym za jego stosowaniem w kontekstach kryminalistycznych.

### Kluczowe Różnice Między Paradygmatami

| Cecha                    | Classification CIT (BFP)                                                                                      | Complex Trial Protocol (CTP)                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Generacja/Twórca**     | Pierwsza generacja (Farwell & Donchin).                                                                       | Druga generacja (Rosenfeld).                                             |
| **Główny cel**           | Klasyfikacja na podstawie _podobieństwa_ P do T.                                                              | Zapewnienie _odporności na kontrmaneuwy_.                                |
| **Struktura próby**      | Pojedynczy bodziec (P, I lub T) w serii _oddball_.                                                            | **Próba złożona (S1 + S2):** S1 (P lub I) + S2 (T lub NT).               |
| **Zadanie dla S1 (P/I)** | **Dyskryminacja/Zadanie Podwójne:** Badany musi jawnie odróżnić T od P/I.                                     | **Proste Potwierdzenie:** Identyczna reakcja na P i I ("widziałem to").  |
| **Rola Targeta (T)**     | **Szablon ERP** dla korelacji z Sondą (P).                                                                    | **Wymuszanie uwagi** w S2, oddzielone w czasie od S1.                    |
| **Metryka Analizy**      | **BCD** (Bootstrapped Correlation Difference) – korelacja P-T vs. P-I.                                        | **BAD** (Bootstrapped Amplitude Difference) – różnica amplitudy P vs. I. |
| **Wrażliwość na CMs**    | Wrażliwy, ponieważ CMs mogą zredukować różnicę P-I, choć BFP twierdzi, że jest odporny, używając P300-MERMER. | Znacznie bardziej **odporny** dzięki rozdzieleniu zadań S1/S2.           |
| **Mierzony komponent**   | Zazwyczaj **P300-MERMER** (300–1800 ms).                                                                      | Amplituda **P300** (często mierzona _peak-to-peak_).                     |

Podsumowując, Classification CIT (BFP) jest paradygmatem ukierunkowanym na **dokładną klasyfikację** poprzez użycie targetów jako wewnętrznego wzorca dla reakcji Probe i zastosowanie analizy korelacyjnej na pełnej fali P300-MERMER. Natomiast CTP jest ewolucją protokołu mającą na celu **skuteczne oddzielenie procesów kognitywnych** (rozpoznanie ukrytej informacji w S1 od dyskryminacji w S2), co czyni go znacznie bardziej **odpornym na próby oszustwa**.