Metoda: **CIT (Ogólny protokół)**

Artykuł stanowi przegląd naukowych badań (2017–2022) nad wykrywaniem kłamstw za pomocą EEG, koncentrując się na **potencjale wywołanym P300** w odpowiedzi na wzrokowe bodźce w postaci znanych i nieznanych twarzy.

### Użycie ML

**Metodologia:**
1. **Protokół:** Najczęściej stosowaną metodą jest **Test Ukrytej Informacji (CIT)** (78%).
2. **Przetwarzanie Wstępne:** We wszystkich pracach używano **Filtra Pasmowo-Przepustowego (BPF)**.
3. **Ekstrakcja Cech:** Najczęściej używaną metodą jest **Transformata Faleckowa (WT)** (34%).
4. **Klasyfikacja:** Najczęściej stosowanymi algorytmami są **LDA, SVM i MLFFNN**.

**Najlepszy wynik** w binarnej klasyfikacji danych EEG w kontekście P300 w odpowiedzi na rozpoznawanie twarzy został osiągnięty przez Bablani et al.:
• **Dokładność:** **96.8%**.
• **Metody:** Wykorzystano Transformację Faleckową (WT) do ekstrakcji cech oraz klasyfikator SVM, którego parametry zostały zoptymalizowane za pomocą algorytmu BAT. Użyto również algorytmu binarnego BAT do selekcji kanałów EEG, co pozwoliło na usunięcie niefunkcjonalnych kanałów (znajdujących się w płacie potylicznym), zwiększając wydajność systemu.