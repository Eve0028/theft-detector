### 2. Zapobieganie "Double Dipping" (Podwójnemu Zanurzaniu)

Dlaczego okno czasowe musi być wyznaczane na innym bodźcu (S2) niż ten, który podlega testowaniu (S1)? W neuroinformatyce wybieranie okna czasowego (ROI - Region of Interest) na podstawie tych samych danych, na których potem liczy się statystykę, to poważny błąd.

- **Kriegeskorte, N., et al. (2009). _Circular analysis in systems neuroscience: the dangers of double dipping._ Nature Neuroscience.** Klasyczna praca, która zrewolucjonizowała podejście do analizy EEG/fMRI. Autorzy pokazują, że dostosowywanie okna analitycznego tam, "gdzie widać największą różnicę" między warunkami eksperymentalnymi, drastycznie sztucznie pompuje wyniki (fałszywe alarmy). Aby tego uniknąć, należy używać niezależnych danych kalibracyjnych (ang. _independent functional localizer_) – w Twoim przypadku to uśredniona epoka dla S2 (Target).
    
- **Brooks, J. L., et al. (2017). _Data-driven region-of-interest selection without inflating Type I error rate: Safe data-driven ROI selection._ Psychophysiology.** Badacze potwierdzają, że wybór okien czasowych w badaniach nad ERP (w tym P300) może krytycznie wpływać na wnioski wyciągane z badania, a stosowanie wyboru bazującego bezpośrednio na analizowanych danych drastycznie zwiększa błąd pierwszego rodzaju (Type I error). Wybór okna P300 za pomocą danych kalibracyjnych/zautomatyzowanych algorytmów zabezpiecza przed tym zjawiskiem.


Zautomatyzowane podejście "ślepe" (Blind / Automated Pipeline)

**Meixner, J. B., & Rosenfeld, J. P. (2014). _A mock terrorism application of the P300-based concealed information test: Predicting exact weapons and specific targets._** W nowszych pracach z laboratorium Rosenfelda stosuje się wystandaryzowane algorytmicznie okna poszukiwań. Zdefiniowany wcześniej algorytm po prostu "przeczesuje" dozwolone okno i samodzielnie ekstrahuje pik P300.
