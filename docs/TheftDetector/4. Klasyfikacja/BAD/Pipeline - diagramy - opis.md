# Pipeline przetwarzania danych - diagramy

---

## Diagram 1 - Przetwarzanie jednego uczestnika

Wykonywany raz per uczestnik: w każdym trialu Optuny i podczas końcowej ewaluacji.

```mermaid
flowchart TD
    A(["Surowe EEG - jeden uczestnik"]) --> S1

    subgraph S1["1. Filtracja"]
        F1["Notch 50 Hz (zakłócenie sieciowe)\n+ Butterworth HP i LP\nParametry: hp_cutoff | lp_cutoff | iir_order"]
    end

    S1 --> S2

    subgraph S2["2. Wyznaczenie indywidualnego okna P300 z zadania S2 (target)"]
        E2["Epoki S2 (bodźce docelowe)\ntmin / tmax | korekcja bazowa | detrend\n→ odrzucenie artefaktów (autoreject lub IQR)\n→ odrzucenie epok z niestabilną linią bazową"]
        E2 --> ERP["Uśrednienie ocalałych epok → ERP na kanale Pz\n(opcjonalne wygładzenie dolnoprzepustowe ERP)"]
        ERP --> PEAK["Szukanie szczytu P300 na Pz\nw oknie czasowym: peak_search_start – peak_search_end\n→ czas szczytu P300 [s]"]
        PEAK --> WIN["Okno indywidualne\n= czas szczytu P300 ± margines"]
    end

    subgraph S3["3. Wybór okna P300"]
        WIN --> M3{"Tryb okna P300"}
        M3 -- "individual" --> WI["Okno indywidualne\n= czas szczytu P300 ± margines\n(różne dla każdej osoby)"]
        M3 -- "static" --> WS["Okno stałe\n= window_start – window_end\n(identyczne dla wszystkich)"]
    end

    S3 --> S4

    subgraph S4["4. Epoki S1 - bodźce probe i irrelevant"]
        E4["Wycinanie epok wokół bodźców S1\n→ usunięcie prób z błędną odpowiedzią lub przekroczonym RT\n→ odrzucenie artefaktów (ten sam tryb co S2)\n→ czyste epoki: probe + irrelevant"]
    end

    S4 --> S5

    subgraph S5["5. Test BAD (bootstrap) - decyzja dla uczestnika"]
        B1["Dla każdej czystej epoki S1:\noblicz amplitudę EEG w oknie P300\nMetoda: mean | baseline-to-peak | peak-to-peak\n(opcjonalne wygładzenie LP przed obliczeniem)"]
        B1 --> B2["Bootstrap (N iteracji):\nlosuj 1 epokę probe i 1 epokę irrelevant\nmax_prop = odsetek iteracji,\ngdzie amplituda(probe) > amplituda(irrelevant)"]
        B2 --> DEC{"max_prop\n≥ guilty_threshold?"}
        DEC -- "Tak" --> G(["GUILTY"])
        DEC -- "Nie" --> IN(["INNOCENT"])
    end
```



### Opis

Pipeline przetwarza dane jednego uczestnika od surowego sygnału EEG do binarnej decyzji klasyfikacyjnej. Ten sam kod jest wywoływany zarówno w każdym trialu optymalizacji Optuny (Diagram 2), jak i podczas końcowej ewaluacji (Diagram 3).

**Krok 2 - dlaczego korzystamy z zadania S2?**
Każdy uczestnik reaguje inaczej na bodźce - szczyt fali P300 pojawia się w różnym czasie po bodźcu (zwykle 300–600 ms). Zamiast przyjmować stałe okno dla wszystkich, wyznaczamy je indywidualnie: uśredniamy odpowiedzi na bodźce docelowe (S2 target), szukamy szczytu P300 na kanale Pz, a następnie budujemy okno czasowe jako `czas szczytu ± margines`. Zapewnia to precyzyjniejszy pomiar amplitudy P300 per uczestnik.

**Krok 5 - co mierzy test BAD?**
`max_prop` (max_proportion) to odsetek iteracji bootstrapu, w których amplituda EEG po bodźcu „probe" (skradziony przedmiot) przekroczyła amplitudę po bodźcach „irrelevant" (neutralne przedmioty). Wartość bliska 1.0 oznacza, że mózg uczestnika konsekwentnie reaguje silniej na probe - co interpretujemy jako rozpoznanie ukrytej informacji.

**Pojęcia:**


| Termin                | Znaczenie                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| S1                    | Główne zadanie eksperymentu: uczestnik odpowiada na probe i irrelevant jednakowo, ale mózg „złodzieja" reaguje silniej na probe |
| S2 (target)           | Dodatkowe rzadkie bodźce wplecione w sesję; użytkownik musi zareagować - służą wyłącznie do kalibracji indywidualnego okna P300 |
| probe                 | Bodziec S1 z ukrytą informacją (skradziony przedmiot)                                                                           |
| irrelevant            | Neutralne bodźce S1 (inne przedmioty)                                                                                           |
| max_prop              | Kluczowy wynik BAD: proporcja bootstrapu (0–1)                                                                                  |
| RT                    | Czas reakcji (reaction time)                                                                                                    |
| peak_search_start/end | Granice przeszukiwania szczytu P300 w ERP (w sekundach po bodźcu)                                                               |
| margines              | Połówkowa szerokość okna P300 wokół wyznaczonego szczytu                                                                        |
| guilty_threshold      | Próg decyzyjny dla max_prop - wyznaczany przez LOOCV                                                                            |


---

## Diagram 2 - Optymalizacja Optuna TPE (200 triali)

Szuka kombinacji parametrów o najwyższym ROC-AUC na wszystkich uczestnikach.
Każdy trial = pełny przebieg Diagramu 1 dla każdego uczestnika.

**Ustawienia sesji:** Mirror S2→S1 | LOOCV | F-β = 0.5 | N_bootstrap = 1000 | 200 triali

```mermaid
flowchart TD
    START(["Wyjściowo N=12 uczestników (6 innocent + 6 guilty)\nOdrzucono 1 guilty: zbyt mało czystych epok probe po odrzuceniu artefaktów\n→ N=11 (6 innocent + 5 guilty)\n\nMirror S2→S1: TAK - S2 używa tych samych ustawień co S1\nLOOCV: TAK | F-β = 0.5 | N_bootstrap = 1000"]) --> OPTLOOP

    subgraph OPTLOOP["Pętla Optuna: 200 triali TPE"]

        TPE["TPE proponuje zestaw parametrów\n(uczy się na historii poprzednich triali)"] --> SPACE

        subgraph SPACE["Przestrzeń przeszukiwania"]
            direction LR
            PF["Filtracja\nhp_cutoff: {0.1, 0.3, 0.5, 0.7} Hz  [cat]\nlp_cutoff: 12–30 Hz, krok 3  [int]\niir_order: {2, 3, 4}  [cat]"]
            PE["Epoching S1\nrejection: {autoreject, iqr}  [cat]\nadaptive_k: 1.5–3.0, krok 0.5  [float]\n(tylko gdy rejection = iqr)"]
            PP["Okno P300\nmode: {individual, static}  [cat]\nmargines: 0.10–0.20 s, krok 0.05  [float]\npeak_search: 0.25–0.35 s → 0.60–0.75 s  [float]\nwygładzenie ERP: {6, 8, 10, 12, None} Hz  [cat]\n- static: window_start 0.25–0.35, window_end 0.60–0.80 s"]
            PB["CTP-BAD\nmetoda amplitudy: {mean, b2peak, p2p-R, p2p-V}  [cat]\nsmoothing: {None, LP-Butterworth}  [cat]\nsmoothing_lp: 6–12 Hz, krok 2  [float]"]
            PF ~~~ PE
            PE ~~~ PP
            PP ~~~ PB
        end

        SPACE --> ENFORCE["Reguła spójności:\njeśli metoda amplitudy = Peak  i  smoothing = None\n→ wymuszono smoothing = Low-pass Butterworth"]

        ENFORCE --> RUN

        subgraph RUN["Uruchom Diagram 1 dla wszystkich uczestników"]
            R1["Uczestnik 1 → max_prop₁"]
            R2["Uczestnik 2 → max_prop₂"]
            RN["… Uczestnik N → max_propₙ"]
        end

        RUN --> SCORE

        subgraph SCORE["Ocena trialu"]
            SC1["ROC-AUC - sweep 201 progów (0 → 1) na wszystkich N uczestnikach\n= pole pod krzywą TPR(FPR)  ← cel optymalizacji Optuny"]
            SC1 --> LOO["LOOCV - metryki pomocnicze (nie optymalizowane):\ndla każdego uczestnika i:\n  sweep progów na N-1 pozostałych → próg maksymalizujący F-β(0.5)\n  zaklasyfikuj uczestnika i tym progiem\n→ Accuracy | Sensitivity | Specificity | F-β"]
        end

        SCORE --> REG["Optuna rejestruje AUC trialu\n→ TPE aktualizuje model bayesowski"]
        REG --> NEXT{"Trial 200?"}
        NEXT -- "Nie" --> TPE
        NEXT -- "Tak" --> BEST
    end

    BEST["Wybierz trial z MAX ROC-AUC\n→ najlepsza konfiguracja"]
```



### Opis

Optuna TPE (Tree-structured Parzen Estimator) to bayesowska metoda optymalizacji. W odróżnieniu od prostego Grid Search, po każdym trialu TPE buduje probabilistyczny model zależności między parametrami a wynikiem - i proponuje kolejne kombinacje z wyższym oczekiwanym AUC. Dzięki temu 200 triali eksploruje przestrzeń bardziej efektywnie niż równoważna siatka.

**Dlaczego ROC-AUC jako cel, a nie Accuracy?**
ROC-AUC mierzy zdolność `max_prop` do odróżnienia grup guilty/innocent niezależnie od wyboru konkretnego progu decyzyjnego. Accuracy zależy od progu - a ten jest zawsze dobierany per fold LOOCV, więc AUC jest bardziej stabilną i uczciwszą miarą do optymalizacji.

**Mirror S2→S1** oznacza, że epoki S2 (kalibracyjne) są wycinane i filtrowane tymi samymi parametrami co epoki S1 (główne) - gwarantuje to spójność preprocessing u między obiema ścieżkami.

**Pojęcia:**


| Termin      | Znaczenie                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------------ |
| TPE         | Tree-structured Parzen Estimator - algorytm bayesowski wybierający następny punkt przeszukiwania |
| [cat]       | Parametr kategoryczny: losowany z podanej listy wartości                                         |
| [int]       | Parametr całkowity z krokiem: np. lp_cutoff ∈ {12, 15, 18, …, 30} Hz                             |
| [float]     | Parametr zmiennoprzecinkowy z krokiem: np. margines ∈ {0.10, 0.15, 0.20} s                       |
| adaptive_k  | Próg odrzucania artefaktów IQR/z-score: wyższe k = mniej agresywne odrzucanie                    |
| b2peak      | baseline-to-peak: amplituda szczytu P300 względem wyzerowanej linii bazowej                      |
| p2p-R       | peak-to-peak (Rosenfeld): szczyt dodatni minus dolina ujemna po nim                              |
| p2p-V       | peak-to-peak (Peak-Valley): globalny max minus globalny min w oknie                              |
| F-β (β=0.5) | Ważona miara F: β < 1 premiuje specyficzność (ochronę niewinnych) ponad czułość                  |
| LOOCV       | Leave-One-Out Cross-Validation - patrz Diagram 3                                                 |


---

## Diagram 3 - Końcowa ewaluacja LOOCV (Quick Pipeline)

Uruchamiany oddzielnie z zamrożonymi parametrami i N_bootstrap = 10 000.
Logika LOOCV jest **identyczna** jak w ocenie każdego trialu Optuny (Diagram 2) - różnica polega wyłącznie na liczbie iteracji bootstrapu (10 000 vs 1 000).

```mermaid
flowchart TD
    START(["Zamrożona najlepsza konfiguracja z Optuny\nN_bootstrap = 10 000 | F-β = 0.5"]) --> RUN

    subgraph RUN["Uruchom Diagram 1 dla każdego uczestnika (N = 11)"]
        RA["Uczestnik 1 → max_prop₁"]
        RB["Uczestnik 2 → max_prop₂"]
        RN["… Uczestnik N → max_propₙ"]
        RA ~~~ RB
        RB ~~~ RN
    end

    RUN --> NOTE["Diagram 1 NIE jest tu ponownie uruchamiany\nLOOCV operuje wyłącznie na gotowych wartościach max_prop"]

    NOTE --> LOOCV

    subgraph LOOCV["Pętla LOOCV - N rund, jedna per uczestnik (i = 1 … N)"]
        L1["Zbiór treningowy: N-1 uczestników bez uczestnika i\n→ sweep 201 progów na N-1\n→ wybierz próg maksymalizujący F-β(0.5)"]
        L1 --> L2["Zaklasyfikuj uczestnika i wybranym progiem:\nmax_propᵢ ≥ próg → GUILTY\nw przeciwnym razie → INNOCENT"]
    end

    LOOCV --> AGG

    subgraph AGG["Metryki końcowe (N predykcji LOO)"]
        AG["Macierz pomyłek: TP | FP | FN | TN\nAccuracy    = (TP + TN) / N\nSensitivity = TP / (TP + FN)  - czułość, wykrycie winnych\nSpecificity = TN / (TN + FP)  - swoistość, ochrona niewinnych\nF-β(0.5) z predykcji LOO\nROC-AUC (resubstitution - niezależne od progów LOO)"]
    end
```



### Opis

Procedura LOOCV jest identyczna z tą w ocenie trialu Optuny. Jedyna różnica to N_bootstrap: **1 000 w Optunie** (szybciej, wyższa wariancja) vs **10 000 tutaj** (wolniej, stabilniejszy wynik `max_prop`). Oznacza to, że wyniki LOOCV z tego kroku mogą nieznacznie różnić się od tych z najlepszego trialu Optuny - to jest zamierzone.

**Pojęcia:**


| Termin             | Znaczenie                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| LOOCV              | Leave-One-Out CV: N foldów, w każdym jeden uczestnik jest walidacyjny, reszta treningowa                    |
| sweep 201 progów   | Dla każdego możliwego progu od 0 do 1 (201 równomiernych kroków) sprawdź F-β na zbiorze treningowym         |
| resubstitution AUC | AUC liczone na wszystkich N wynikach bez podziału na foldy - mierzy separowalność grup, nie zależy od progu |
| Sensitivity        | Czułość = TP / (TP+FN): odsetek rzeczywiście winnych, których poprawnie wykryto                             |
| Specificity        | Swoistość = TN / (TN+FP): odsetek rzeczywiście niewinnych, których poprawnie oczyszczono                    |


---

## Diagram 4 - Najlepsza konfiguracja i wyniki

Najlepsza konfiguracja z 200 triali Optuny. Pokazane metryki pochodzą z oceny **tego konkretnego trialu** (N_bootstrap = 1000 | LOOCV | F-β = 0.5 | Mirror S2→S1) - nie z oddzielnego uruchomienia Quick Pipeline (Diagram 3).

```mermaid
flowchart LR
    subgraph BEST_CFG["Najlepsza konfiguracja (trial wygrywający)"]
        direction TB

        subgraph F["Filtracja"]
            F1["High-pass:  0.3 Hz\nLow-pass:   24 Hz  ← wartość int krokowa, krok 3 Hz\nRząd IIR:   3\nNotch:      50 Hz"]
        end

        subgraph EP["Epoching S1 i S2"]
            EP1["Okno:       −0.2 s – 1.0 s po bodźcu\nDetrend:    DC offset (odjęcie średniej linii bazowej)\nOdrzucanie: autoreject\n            (adaptive_k nie jest próbkowane -\n             autoreject wyznacza próg automatycznie)\nMirror S2→S1: TAK"]
        end

        subgraph P300["Okno P300: individual (indywidualne)"]
            P1["Przeszukiwany przedział: 0.25 – 0.75 s\nMargines ±: 0.15 s\nWygładzenie ERP S2: 10 Hz LP\n→ okno = czas szczytu P300 ± 0.15 s (różne per uczestnik)"]
        end

        subgraph BAD["CTP-BAD"]
            B1["Metoda amplitudy: peak-to-peak (Peak-Valley)\n= globalny max minus globalny min w oknie P300\nWygładzenie epok S1: Low-pass 10 Hz\n(wymuszone automatycznie dla metod Peak)"]
        end

        F1 ~~~ EP1
        EP1 ~~~ P1
        P1 ~~~ B1
    end

    BEST_CFG --> RESULTS

    subgraph RESULTS["Wyniki LOOCV (N = 11 | β = 0.5)"]
        direction TB

        subgraph MET["Metryki"]
            M1["ROC-AUC:      0.9333\nAccuracy:     72.7%  (8 / 11 poprawnych)\nSensitivity:  66.7%  (4 / 6 guilty wykrytych)\nSpecificity:  80.0%  (4 / 5 innocent oczyszczonych)\nF-β (β=0.5): 0.7692"]
        end

        subgraph INTERP["Interpretacja"]
            I1["β = 0.5 → specyficzność ważniejsza od czułości\n= wolimy nie oskarżać niewinnych kosztem nieznalezienia winnego\n\nAUC = 0.93 → max_prop dobrze separuje grupy\nguilty / innocent niezależnie od progu\n\n2 osoby guilty niezaklasyfikowane → FN\n1 osoba innocent sklasyfikowana jako guilty → FP"]
        end

        M1 ~~~ I1
    end
```



**Macierz pomyłek (N = 11):**


|                           | Pred. GUILTY | Pred. INNOCENT |
| ------------------------- | ------------ | -------------- |
| **Actual GUILTY** (n=6)   | 4 - TP       | 2 - FN         |
| **Actual INNOCENT** (n=5) | 1 - FP       | 4 - TN         |


### Opis

**Jak interpretować wyniki przy N = 11?**
AUC = 0.93 to silny sygnał, że `max_prop` faktycznie rozróżnia grupy. Jednak bezpośrednie metryki (Sensitivity = 66.7%, Accuracy = 72.7%) są umiarkowane - przy N = 11 jedna pomyłka zmienia Accuracy o ~9 p.p., więc wyniki należy traktować ostrożnie. Ważne jest też to, że F-β(0.5) premiuje Specificity: z 5 niewinnych uczestników 4 zostało poprawnie oczyszczonych (FP = 1).