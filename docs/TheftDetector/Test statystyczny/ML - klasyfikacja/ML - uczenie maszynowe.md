Zastosowanie metod uczenia maszynowego w analizie sygnałów EEG do wykrywania ukrytej wiedzy: Model wewnątrzosobniczy w paradygmacie predykcji wstecznej S2-S1

Rozpoznawanie śladów pamięciowych i ukrytych informacji za pomocą analizy bioelektrycznej aktywności mózgu stanowi jedno z najbardziej zaawansowanych wyzwań współczesnej neuronauki obliczeniowej. Elektroencefalografia (EEG), oferując milisekundową rozdzielczość czasową, pozwala na bezpośredni wgląd w procesy kognitywne towarzyszące rozpoznawaniu bodźców o znaczeniu krytycznym, co czyni ją narzędziem znacznie bardziej obiektywnym i odpornym na manipulacje niż tradycyjne metody poligraficzne. Kluczowym elementem tych systemów jest test ukrytej wiedzy (Concealed Information Test – CIT), który opiera się na detekcji specyficznych reakcji neuronalnych na bodźce znane jedynie osobie posiadającej wiedzę o danym zdarzeniu. Wprowadzenie zaawansowanych algorytmów uczenia maszynowego (Machine Learning – ML) umożliwiło automatyzację tego procesu, jednak największą barierą pozostaje niestacjonarność sygnału EEG i zmienność wewnątrzosobnicza między różnymi sesjami pomiarowymi. Niniejszy raport szczegółowo analizuje zastosowanie modeli ML w układzie wewnątrzosobniczym, ze szczególnym uwzględnieniem rygorystycznego paradygmatu uczenia modelu na danych z sesji późniejszej (S2) i testowania go na sesji wcześniejszej (S1). Takie podejście, znane jako predykcja wsteczna, pozwala na rzetelną ocenę stabilności „odcisków mózgu” (brainprints) oraz zdolności algorytmów do generalizacji wiedzy w warunkach czasowej regresji parametrów neuronalnych.

Neurofizjologiczne i kognitywne fundamenty detekcji wiedzy

Podstawą skuteczności systemów EEG-CIT jest identyfikacja komponentów potencjałów wywołanych (Event-Related Potentials – ERP), które manifestują się w odpowiedzi na bodźce istotne (probes) w porównaniu do bodźców neutralnych (irrelevant). Centralną rolę odgrywa tutaj załamek P300, endogenna fala pojawiająca się zazwyczaj między 300 a 600 ms po prezentacji bodźca. Amplituda P300 odzwierciedla intensywność procesów uwagi oraz proces aktualizacji pamięci roboczej, co zostało potwierdzone w metaanalizach obejmujących 54 badania eksperymentalne, wykazujących ogromną siłę efektu wynoszącą d∗=1,59.

Analiza przestrzenna sygnału wskazuje, że procesy rozpoznawania ukrytej wiedzy nie są zlokalizowane w jednym punkcie, lecz angażują szerokie sieci korowe. Szczególne znaczenie mają obszary ciemieniowe (elektroda Pz) oraz skroniowe (elektroda T7), które wykazują najwyższą czułość w wykrywaniu kłamstwa i posiadania wiedzy. Badania wykorzystujące rezonans magnetyczny i lokalizację źródeł EEG sugerują również zaangażowanie prawego zakrętu kątowego (AG) oraz prawego dolnego zakrętu czołowego (IFG) w procesy generowania odpowiedzi kłamliwych, co stanowi fundament dla doboru cech przestrzennych w modelach ML.

|Komponent ERP|Latencja (ms)|Znaczenie w CIT|Lokalizacja dominująca|
|---|---|---|---|
|N200|200 – 300|Wykrywanie niedopasowania i konfliktów|Czołowo-centralna|
|P300|300 – 600|Rozpoznanie bodźca, alokacja uwagi|Pz (ciemieniowa)|
|LPC (Late Positive Complex)|> 600|Epizodyczne przypominanie sobie|Regiony ciemieniowe|


Wykorzystanie paradygmatu Rapid Serial Visual Presentation (RSVP) pozwala na prezentację bodźców na granicy świadomości, co utrudnia badanym stosowanie przeciwdziałań (countermeasures) mających na celu oszukanie testu. W takich warunkach systemy oparte na P300 wykazują niemal 100-procentową skuteczność w wykrywaniu tożsamości osób na podstawie bodźców podprogowych, co podkreśla potencjał EEG jako narzędzia do penetracji ukrytych warstw pamięci.

Architektury przetwarzania sygnału i ekstrakcji cech

Przekształcenie surowego sygnału EEG w dane użyteczne dla klasyfikatorów uczenia maszynowego wymaga wieloetapowego potoku przetwarzania. Ze względu na słabą amplitudę sygnału mózgowego i wysoki poziom szumu, wstępne przetwarzanie ma krytyczne znaczenie dla stabilności modelu wewnątrzosobniczego.

Preprocessing i redukcja artefaktów

Standardowy proces rozpoczyna się od filtrowania pasmowoprzepustowego (Bandpass Filtering – BPF), najczęściej w zakresie od 0,5 Hz do 35 Hz, co eliminuje dryft linii izoelektrycznej oraz zakłócenia sieciowe (50/60 Hz). Następnie stosuje się zaawansowane metody separacji źródeł, takie jak analiza składowych niezależnych (Independent Component Analysis – ICA), w celu usunięcia artefaktów ocznych (EOG) i mięśniowych (EMG), które mogłyby zostać błędnie zinterpretowane przez algorytm jako aktywność kognitywna.] W badaniach nad stabilnością cross-session wykazano, że czyszczenie danych może poprawić dokładność klasyfikacji nawet o 8,4% w porównaniu do surowych sygnałów.

Zaawansowana ekstrakcja cech

W modelach wewnątrzosobniczych stosuje się ekstrakcję cech w trzech głównych domenach:

1. **Domena czasowa:** Wykorzystuje się amplitudę szczytową, pole pod krzywą (AUC), latencję oraz parametry Hjortha (aktywność, mobilność, złożoność). Parametry te pozwalają na uchwycenie dynamiki fali P300.
2. **Domena częstotliwości i czasu-częstotliwości:** Gęstość widmowa mocy (PSD) w pasmach delta, theta, alpha i beta dostarcza informacji o stanie pobudzenia kognitywnego. Dyskretna transformata falkowa (DWT) jest szczególnie ceniona za zdolność do analizy niestacjonarnych sygnałów EEG.
3. **Domena nieliniowa i łączność:** Entropia (np. Multi-scale Entropy, Fuzzy Entropy) mierzy złożoność sygnału, a wskaźniki Phase-Locking Value (PLV) i koherencja falkowa (WC) opisują synchronizację między różnymi obszarami mózgu.

Szczególnie obiecujące wyniki daje fuzja różnych typów entropii, umożliwiająca wielowymiarową charakterystykę sygnału, co znacząco podnosi odporność modelu na zmienność sesyjną.

Metody uczenia maszynowego w scenariuszach wewnątrzosobniczych

Wybór algorytmu ML w testach CIT zależy od balansu między dokładnością a interpretowalnością neuronalną. W badaniach dominują dwa podejścia: klasyczne algorytmy oparte na ręcznie wyselekcjonowanych cechach oraz modele głębokiego uczenia.

Klasyczne klasyfikatory i ich stabilność

Maszyny wektorów nośnych (SVM) z jądrem RBF oraz liniowa analiza dyskryminacyjna (LDA) są powszechnie stosowane ze względu na ich skuteczność przy małych zbiorach danych. W testach wewnątrzosobniczych SVM potrafi osiągnąć dokładność rzędu 82,74%, a w niektórych optymalizowanych konfiguracjach z wykorzystaniem Empirical Mode Decomposition (EMD) nawet 99,44%. Jednakże LDA, mimo dobrej separowalności wewnątrz sesji, wykazuje wyższą degradację wydajności w testach między sesjami (spadek o ok. 6,29%) w porównaniu do algorytmu k-najbliższych sąsiadów (KNN), który traci jedynie 2,52% dokładności.

Głębokie i hybrydowe modele neuronowe

Głębokie sieci neuronowe (Deep Learning), w tym 1D-CNN oraz architektury takie jak DeepSleepNet, wykazują wyższą odporność na brak równowagi klas i szum w porównaniu do klasycznych modeli statystycznych. Najnowocześniejsze podejścia wykorzystują modele hybrydowe, łączące Transformerów z kwantowymi sieciami neuronowymi (QNN). Transformer służy do modelowania długozakresowych zależności czasowych w sygnale EEG, podczas gdy warstwy kwantowe, wykorzystując superpozycję i splątanie w przestrzeni Hilberta, wykrywają subtelne, nieliniowe korelacje cech. Model ten na zbiorze BCIAUT-P300 osiągnął dokładność 0,921, co czyni go jednym z najskuteczniejszych narzędzi w diagnostyce opartej na P300.

| Algorytm        | Dokładność (Mean) | F1-Score | Specyfika zastosowania                      |
| --------------- | ----------------- | -------- | ------------------------------------------- |
| SVM (RBF)       | 82,74%            | 0,830    | Stabilna klasyfikacja małych próbek         |
| XGBoost         | 98,00%            | 0,975    | Biometryczna identyfikacja osób             |
| KNN             | 81,20%            | 0,805    | Najwyższa stabilność cross-session          |
| CNN             | 81,03% (Median)   | 0,790    | Automatyczna ekstrakcja cech przestrzennych |
| Transformer-QNN | 92,10%            | 0,915    | Wykrywanie nieliniowych wzorców P300        |
|                 |                   |          |                                             |
