### **1. Wstęp**

Zaczynamy bardzo ogólnie i płynnie przechodzimy do głównego tematu pracy.

- **1.1. Ewolucja metod detekcji kłamstwa w kryminalistyce:** Krótkie historyczne wprowadzenie. Tradycyjne poligrafy (wariografy) mierzące reakcje autonomicznego układu nerwowego (pot, tętno) i ich wady (np. podatność na stres). Potrzeba stworzenia metod obiektywnych.
    
- **1.2. Neurofizjologiczne podstawy wykrywania informacji ukrytej:** Płynne przejście z układu autonomicznego na ośrodkowy układ nerwowy (mózg). Krótka zajawka, że mózg reaguje automatycznie i inaczej na znane sobie bodźce (tzw. "ukrytą wiedzę"), co można zmierzyć za pomocą elektroencefalografii (EEG).

### **2. Cel i zakres pracy**

Czyli co i po co zostało zrobione.

- **2.1. Główny cel badawczy:** Stwierdzenie, że celem jest ocena skuteczności i walidacja niskobudżetowego, niskokanałowego systemu EEG (BrainAccess Mini) w wykrywaniu informacji ukrytej (kradzieży) przy użyciu specyficznego paradygmatu badawczego.
    
- **2.2. Zakres zrealizowanych prac:** Wypunktowanie wszystkiego, co zostało zrobione. Zebranie grupy badawczej (14 osób), zaprojektowanie i przeprowadzenie eksperymentu "Mock Crime", pozyskanie sygnałów EEG, preprocessing, analiza statystyczna metodą BAD oraz wykorzystanie modeli uczenia maszynowego (SVM, LDA) na ręcznie wyekstrahowanych cechach.

### **3. Charakterystyka sygnału EEG i technologii pomiarowej**

Teoria sprzętowa i neurofizjologiczna.

- **3.1. Fizyka i biomechanika sygnału EEG:** Krótkie wyjaśnienie, skąd biorą się fale mózgowe (aktywność postsynaptyczna neuronów kory mózgowej) i dlaczego sygnał na skórze głowy ma tak niską amplitudę (mikrowolty) – tłumienie przez czaszkę i płyn mózgowo-rdzeniowy.
    
- **3.2. Potencjały wywołane (ERP):** Wyjaśnienie, czym są ERP i jak je wydobywamy z szumu EEG (uśrednianie epok). Zdefiniowanie komponentów P300 (związany z uwagą i rozpoznaniem) oraz krótkie wspomnienie o N200.
    
- **3.3. Metodologia pomiaru - systemy medyczne a konsumenckie:** Zestawienie drogiego sprzętu badawczego (np. 64-kanałowe czepki żelowe) z rozwiązaniami typu "wearables". Wskazanie wad tych drugich (większy szum, ruchy, sucha skóra).
    
- **3.4. Specyfikacja platformy BrainAccess Mini:** Dane o użytym sprzęcie. Informacje o częstotliwości próbkowania (250 Hz), typie elektrod (suche, ząbkowe) oraz fizycznych ograniczeniach (np. problem z dobrym dopasowaniem czepka z tyłu głowy).

### **4. Paradygmaty badawcze w detekcji ukrytej wiedzy**

Teoria eksperymentu (miejsce na dużo cytowań m.in. Rosenfelda i Farwella).

- **4.1. Od klasycznego CIT do CTP:** Ewolucja metodologii. Czym różni się standardowy Concealed Information Test (CIT) od Complex Trial Protocol (CTP). Dlaczego CTP stosuje zadania naprzemienne (S1 i S2).
    
- **4.2. Wykorzystanie markerów ERP w klasyfikacji bodźców:** Wyjaśnienie terminologii: Probe (przedmiot skradziony), Irrelevant (obojętny), Target (cel w S2). Dlaczego oczekujemy wyższej amplitudy P300 po bodźcu Probe tylko u grupy "złodziei".
    
- **4.3. Przegląd dotychczasowych badań i skuteczność klasyfikacji:** Zestawienie wyników innych naukowców (Accuracy/AUC) z wykorzystaniem P300 w detekcji kłamstwa.
    
- **4.4. Aspekty etyczne i prawne (Neuroprawo):** Krótka dyskusja o tym, czy można zmuszać oskarżonych do badań EEG, temat tzw. "brain fingerprinting" w sądach i prywatności myśli.

### **5. Metodologia eksperymentu**

Jak badanie wyglądało w praktyce na PK.

- **5.1. Scenariusz "Mock Crime" i projekt bodźców wizualnych:** Dlaczego użyte zostały pluszaki (poprawa SNR), przedstawienie "wilka" (Probe) oraz sposób wykonania zdjęć (jednolite tło).
    
- **5.2. Grupa badawcza:** Charakterystyka 14 studentów PK. Rekrutacja via Teams, kryteria wykluczenia (implanty, leki, rany).
    
- **5.3. Przygotowanie stanowiska i sprzętu (Setup):** Lokalizacja (PK ul. Warszawska). Logistyka: zmiana drogi kabli w czepku na układ Fz, Cz, Pz. Procedury czyszczące: robienie przedziałka, izopropanol, SigmaSpray (walka z wysoką impedancją).
    
- **5.4. Procedura badawcza i struktura sesji EEG:** Chronologia badania (1-1.5h). Restrykcje przed badaniem (brak kofeiny, czysta głowa). Dokładny podział sesji (5 bloków x 80 prób), struktura pojedynczego triala (0.5s fiksacja -> S1 -> ISI -> S2) oraz logowanie zdarzeń (klawisze Z, M, N).
    
- **5.5. Ankietyzacja i debriefing:** Omówienie ankiet (wstępnej i końcowej) uzupełnianych na kartce, odrzucanie z analizy osób z grupy "złodziei", które nie rozpoznały przedmiotu (Probe).

### **6. Przetwarzanie sygnałów i ekstrakcja cech (Pipeline)**

Jak z surowego, zaszumionego zapisu EEG otrzymano czyste dane gotowe do klasyfikacji/przewidywań.

- **6.1. Odrzucenie komponentu N200 – uzasadnienie metodologiczne:** Argumentacja, dlaczego analizujemy tylko P300. Wyjaśnienie, że silne N200 pojawia się głównie u osób stosujących tzw. _countermeasures_ (środki zaradcze), do których uczestnicy w tym badaniu nie byli instruowani.
    
- **6.2. Preprocessing sygnału EEG:**
    
    - **6.2.1. Filtrowanie (Filtering):** Techniczne uzasadnienie użycia filtru Notch (50 Hz) do usunięcia zakłóceń sieciowych oraz "agresywnego" filtru IIR Butterwortha (0.5–30 Hz, rzędu 4), który ratuje dane przed wolnymi dryftami izolinii, zachowując pasmo P300 (głównie delta i theta).
        
    - **6.2.2. Epokowanie (Epoching) i korekta linii bazowej:** Zdefiniowanie okna czasowego (od -200 ms do 1000 ms), usunięcie z analizy epok po błędnej odpowiedzi na bodziec S2 (brak uwagi) oraz wyjaśnienie procesu _baseline correction_ (-200 do 0 ms).
        
    - **6.2.3. Odrzucanie artefaktów (Artifact Rejection):** Dlaczego w systemie 3-kanałowym nie można użyć popularnej analizy składowych niezależnych (ICA). Opis zastosowania algorytmu `autoreject` w Pythonie (optymalizacja bayesowska progów odrzucania).
        
- **6.3. Ekstrakcja okna P300:** Wybór kanału Pz jako głównego źródła sygnału. Opis metody wygładzania epok (_epoch smoothing_ – low-pass 12 Hz) przed znalezieniem piku, aby uniknąć błędnego łapania szumu jako amplitudy P300.

### **7. Klasyfikacja i analiza statystyczna**

- **7.1. Metoda Bootstrapped Amplitude Difference (BAD):** Dokładny opis algorytmu. Wyjaśnienie iteracyjnego losowania ze zwracaniem (np. 10 000 iteracji) oraz zdefiniowanie budowy rozkładu różnic. Porównanie różnych metryk pomiaru amplitudy (Mean, Peak-to-Peak, Baseline-to-Peak).
    
- **7.2. Uczenie Maszynowe – model wewnątrzosobniczy (intra-subject):**

	- **7.2.1. Konstrukcja modelu:** Koncepcja trenowania modeli na zadaniu wspierającym S2 (nauka indywidualnego wzorca P300 uczestnika) i testowania na zadaniu głównym S1 (Probe/Irrelevant).
    
    - **7.2.2. Ekstrakcja cech falkowych:** Zastosowanie dyskretnej transformaty falkowej (np. DWT, Daubechies/Symlets na 5 poziomie dekompozycji) do wydobycia cech z pasm niskich częstotliwości (delta/theta).
        
    - **7.2.3. Selekcja cech i klasyfikatory:** Opis działania modeli SVM i LDA. Omówienie redukcji wymiarowości dla LDA (np. algorytmy PCA lub Stepwise), aby uniknąć tzw. "przekleństwa wymiarowości" przy małym zbiorze epok.

### **8. Architektura i implementacja oprogramowania**

Stos technologiczny (tech stack) i struktura kodu.

- **8.1. Akwizycja i synchronizacja danych:** Użycie protokołu Lab Streaming Layer (LSL) do synchronizacji logowania klawiszy (markerów S1/S2) z surowym strumieniem EEG. Zapis do formatu `.fif`.
    
- **8.2. Środowisko analityczne i biblioteki:** Omówienie użycia MNE-Python do preprocessingu EEG oraz `scikit-learn` do modelowania ML. Krótki zarys struktury skryptów (pipeline'u).
    
- **8.3. Wizualizacja i aplikacja interaktywna:** Opis narzędzia stworzonego w frameworku Streamlit służącego do interaktywnej eksploracji preprocesingu, uśrednionych przebiegów (Grand Average) i wyników BAD.

### **9. Wyniki badań**

Część analityczna bez nadmiernej interpretacji (interpretacja będzie w podsumowaniu).

- **9.1. Ocena jakości surowych danych:** Statystyki opisowe SNR (Signal-to-Noise Ratio). Analiza wpływu przygotowania skóry (izopropanol, SigmaSpray) na obniżenie impedancji w sprzęcie z suchymi/ząbkowymi elektrodami.
    
- **9.2. Analiza wizualna potencjałów wywołanych (ERP):** Wykresy _Grand Average_ (uśrednione przebiegi). Zestawienie różnic w amplitudzie dla bodźców Target vs Nontarget (S2) oraz Probe vs Irrelevant (S1).
    
- **9.3. Skuteczność statystycznej detekcji kłamstwa (Metoda BAD):** Wyniki klasyfikacji w zależności od ustalonego progu ufności bootstrap (np. 70%, 80%, 90%). Wskazanie odsetka fałszywych alarmów (False Positives) oraz trafnych detekcji (True Positives).
    
- **9.4. Skuteczność modeli Uczenia Maszynowego:** Porównanie wydajności modeli SVM oraz LDA. Przedstawienie metryk takich jak dokładność (Accuracy), czułość (Sensitivity), swoistość (Specificity) oraz pole pod krzywą ROC (AUC). Porównanie tych wyników z metodą BAD.


### **10. Dyskusja wyników**

- **10.1. Analiza ograniczeń aparaturowych i metodycznych:** Refleksja nad niedoskonałościami. Krytyka fizycznego dopasowania czepka (suche elektrody, ułożenie). Dyskusja nad psychologicznym aspektem eksperymentu (czy kradzież "wilka" wyzwoliła odpowiednio silne P300).
    
- **10.2. Zmienne behawioralne i wpływ czynników zewnętrznych:** Wnioski z ankiet końcowych (Exit Survey). Jak deklarowane zmęczenie, dyskomfort czepka lub spadek koncentracji wpłynęły na degradację sygnału.
    
- **10.3. Skuteczność BrainAccess Mini na tle literatury:** Zderzenie wyników (BAD, ML) z badaniami wykorzystującymi profesjonalny sprzęt medyczny w podobnych paradygmatach.

### **Podsumowanie**

- **Realizacja celu badawczego i główne wnioski:** Ostateczna odpowiedź na cel pracy: czy BrainAccess Mini poradził sobie w warunkach detekcji P300?
  Np.: "Eksperyment wykazał, że urządzenie BrainAccess Mini pozwala na rejestrację komponentu P300, jednak ze względu na wysoką podatność na artefakty ruchowe i zmienną impedancję elektrod suchych, obecnie generuje zbyt duży odsetek błędnych klasyfikacji (False Positives), aby mogło być bezpiecznie stosowane w rygorystycznym paradygmacie detekcji kłamstwa."
    
- **Perspektywy dalszych badań:** Co można by poprawić w przyszłości (np. zastosowanie elektrod żelowych, zmiana procedury _Mock Crime_ na bardziej stresującą, testowanie głębokich sieci neuronowych na większej próbie).