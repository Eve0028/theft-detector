### 2. Wykorzystanie Fz do kontroli artefaktów

Elektroda **Fz** (frontal/czołowa) leży najbliżej oczu. W związku z tym jest najbardziej podatna na artefakty elektrookulograficzne (EOG), czyli mrugnięcia i ruchy gałek ocznych.

- W neuroinformatyce wykorzystujemy tę właściwość jako atut. Ponieważ nie masz dedykowanych elektrod EOG ani wystarczającej liczby kanałów do przeprowadzenia analizy ICA, elektroda Fz może pełnić rolę "strażnika".
    
- Jeśli zauważysz potężny skok napięcia na elektrodzie Fz, który lekko "rozlewa" się na Cz, a najsłabiej widoczny jest na Pz, wiesz, że to zjawisko oczne, a nie załamek P300. Prawdziwe P300 ma odwrotny gradient (jest najsłabsze z przodu, a najsilniejsze z tyłu).
    

### 3. Gradient topograficzny jako dowód w pracy dyplomowej

Jednym z najczęstszych pytań recenzentów przy badaniach na sprzęcie niskobudżetowym jest: _"Skąd pani wie, że to, co pani mierzy, to faktycznie P300, a nie przypadkowy szum?"_.

- Posiadanie elektrod Fz, Cz i Pz w linii środkowej głowy (_midline_) pozwala Ci na wygenerowanie wykresu, który obroni się sam.
    
- Jeśli pokażesz uśrednione przebiegi (Grand Average) dla wszystkich trzech elektrod i udowodnisz, że amplituda reakcji na skradziony przedmiot rośnie wzdłuż osi Fz < Cz < Pz, dostarczysz twardy, neurofizjologiczny dowód na poprawność swojego eksperymentu. Zdecydowanie zalecałabym dodanie takiego wykresu do rozdziału z wynikami!
    

### 4. Zasilenie modeli Machine Learning (SVM i LDA)

O ile metoda BAD wymaga prostej statystyki, o tyle modele uczenia maszynowego doskonale radzą sobie z wielowymiarowością.

- Jeśli do algorytmów SVM i LDA jako wektor cech "wpuścisz" współczynniki falkowe wyekstrahowane nie tylko z Pz, ale również z Fz i Cz, dostarczysz klasyfikatorowi cennych informacji przestrzennych.
    
- Model będzie w stanie samodzielnie "nauczyć się", jak rozkłada się sygnał na całej głowie i oddzielić wzorzec kłamstwa od szumu sprzętowego.