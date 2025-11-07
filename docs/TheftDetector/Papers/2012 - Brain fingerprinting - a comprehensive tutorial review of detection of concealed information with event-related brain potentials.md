Metoda: **Classification CIT (BFP)**

Artykuł stanowi kompleksowy przegląd techniki **Brain Fingerprinting (BF)**, naukowej metody obiektywnego wykrywania ukrytej informacji przechowywanej w mózgu poprzez pomiar fal mózgowych (EEG). BF jest metodą obiektywną, naukową i nieinwazyjną, mierzącą poznawcze przetwarzanie informacji, a nie kłamstwa, stres czy emocje.

### Użyte metody i mechanizmy
Kluczowym potencjałem mózgowym wykorzystywanym w BF jest **P300-MERMER** (Memory and Encoding Related Multifaceted Electroencephalographic Response).
1. **Potencjał P300:** Jest to dobrze znany potencjał wywołany zdarzeniem (ERP), szczyt elektrycznie pozytywny (P) pojawiający się 300–800 ms po bodźcu (w zależności od jego złożoności). P300 występuje, gdy mózg rozpoznaje i przetwarza bodziec, który jest znaczący w danym kontekście.
2. **P300-MERMER:** Farwell odkrył, że w przypadku bardziej złożonych bodźców, P300 jest częścią większej, złożonej reakcji, P300-MERMER. Składa się ona ze szczytu pozytywnego P300, po którym następuje późny potencjał negatywny (Late Negative Potential, LNP), którego latencja może sięgać 1200 ms. P300-MERMER zapewnia większą ilość informacji i jest bardziej niezawodnie wykrywalny niż samo P300, co prowadzi do wyższej pewności statystycznej. P300-MERMER jest maksymalny w okolicy ciemieniowej (P300), ale LNP jest również widoczny w okolicy czołowej.

### Projekt eksperymentalny (Metoda trzech bodźców)
W teście BF prezentowane są na ekranie słowa, frazy lub obrazy, a reakcje mózgu są mierzone nieinwazyjnie za pomocą elektrod EEG umieszczonych na skórze głowy. Stosuje się trzy typy bodźców (w przybliżonym stosunku 1/6 sond, 1/6 celów, 2/3 irrelewantnych):

1. **Sondy (Probes):** Zawierają szczegóły przestępstwa znane tylko sprawcy i śledczym. Osoba badana **zaprzecza** znajomości tych informacji.
2. **Cele (Targets):** Zawierają informacje o przestępstwie, o których eksperymentator jest pewien, że osoba badana zna (np. nazwisko ofiary ujawnione w instrukcjach testu). Bodźce te stanowią **standard** reakcji mózgu osoby badanej na istotną i znaną informację. Wywołują P300-MERMER u wszystkich badanych.
3. **Irrelewantne (Irrelevants):** Zawierają błędne, ale wiarygodne szczegóły przestępstwa. Są nieistotne i nie wywołują P300-MERMER. Stanowią **standard** reakcji na nieistotną informację.

**Procedura:** Osoba badana jest instruowana, aby nacisnąć jeden przycisk w odpowiedzi na cele, a inny na wszystkie pozostałe bodźce (sondy i irrelewantne). W ten sposób badany musi przeczytać i ocenić każdy bodziec, a zadanie behawioralne jest identyczne dla sond i irrelewantnych. Czas prezentacji bodźca to 300 ms, a interwał między bodźcami to 3000 ms.

### Analiza Danych

Celem analizy jest określenie, czy reakcje mózgu na sondy są bardziej podobne do reakcji na cele (informacja obecna), czy do reakcji na irrelewantne (informacja nieobecna).

Wykorzystuje się algorytm statystyczny oparty na **bootstrappingu na korelacjach**. Metoda ta oblicza prawdopodobieństwo i pewność statystyczną, uwzględniając zmienność reakcji w pojedynczych próbach. Algorytm porównuje korelację między falą sond a falą celów z korelacją między falą sond a falą irrelewantnych.

### **Kryteria Decyzyjne:**

• **Informacja obecna:** Stwierdzana, gdy pewność statystyczna wynosi ponad 90%.
• **Informacja nieobecna:** Stwierdzana, gdy pewność statystyczna wynosi ponad 70%.
• **Nieokreślony (Indeterminate):** Jeśli żadne kryterium nie jest spełnione, nie dokonuje się klasyfikacji.

### Wyniki i Dokładność

BF konsekwentnie wykazuje niezwykle wysoką dokładność w badaniach laboratoryjnych i terenowych, pod warunkiem ścisłego przestrzegania protokołów naukowych (20 standardów naukowych BF).
##### Wskaźniki Błędu i Dokładność
W testach i zastosowaniach terenowych prowadzonych w FBI, CIA, Marynarce Wojennej USA oraz w innych miejscach, **BF osiągnął 0% błędów**: nie stwierdzono fałszywych pozytywów ani fałszywych negatywów.
• **100% ustaleń było poprawnych**.
• **W przypadku użycia P300-MERMER:** nie stwierdzono przypadków nieokreślonych.
• **W przypadku użycia samego P300:** odsetek wyników nieokreślonych wyniósł 3%. Wynik nieokreślony nie jest uznawany za błąd.

### Badania terenowe i zastosowanie sądowe

BF został zastosowany w rzeczywistych sprawach kryminalnych, a jego dowody zostały uznane za dopuszczalne w sądzie.
1. **Sprawa Jamesa B. Grindera:** Test BF wykazał, że w mózgu podejrzanego znajdowały się szczegóły morderstwa (wynik: "informacja obecna", 99.9% pewności). Grinder przyznał się do winy.
2. **Sprawa Terry’ego Harringtona:** Testy wykazały "informację nieobecną" w stosunku do przestępstwa i "informację obecną" w stosunku do alibi (pewność >99%). Świadek oskarżenia odwołał swoje zeznania po przedstawieniu mu wyników BF, a Harrington został ostatecznie uniewinniony. Sąd Okręgowy w Iowa uznał dowody BF za wystarczająco wiarygodne, aby zezwolić na ich dopuszczenie.
3. **Badania FBI i CIA/US Navy:** W badaniu agentów FBI BF osiągnął 100% dokładności w wykrywaniu wiedzy istotnej dla FBI. W badaniach CIA i Marynarki Wojennej USA wykryto wiedzę dotyczącą ćwiczebnych scenariuszy szpiegowskich, medycyny wojskowej oraz realnych przestępstw, również z 100% dokładnością.

#### Odporność na środki zaradcze
	Brain Fingerprinting okazał się wysoce odporny na środki zaradcze. Pomimo zaoferowania nagrody w wysokości 100 000 USD za oszukanie testu w sprawach karnych, nikomu się to nie udało. BF prawidłowo wykrył wszystkich badanych, którzy próbowali stosować środki zaradcze. Badania, które wykazały podatność na środki zaradcze, dotyczyły alternatywnych technik, które nie spełniały standardów naukowych BF.