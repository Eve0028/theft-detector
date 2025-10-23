Nowy protokół testu ukrytej informacji oparty na P300, nazwany **Complex Trial Protocol (CTP)**, został opracowany w celu zwiększenia dokładności i odporności na środki zaradcze (CMs) w porównaniu do starszych metod.

**CTP** - ale z tymi samymi typami reakcji na S1 i S2.
W tym badaniu stosowano w S2 te same typy slajdów co w S1. Choć taski są oddzielone w czasie (S1 i S2) - to tu i tu są stosowane np. obrazy przedmiotów.

## Krótkie podsumowanie

**Uczestnicy i Procedura (Badanie Główne):**
• **Grupa Winna (Eksperymentalna):** 12 studentów z zaawansowanej klasy psychofizjologii (18–22 lata).
• **Grupa Niewinna (Kontrolna):** 12 osób z ogólnej populacji (18–62 lata).
• **Przebieg 3 Tygodni:** Uczestnicy z grupy winnej brali udział w trzech blokach, po jednym na tydzień. Tydzień 1 i 3 były warunkami "winny" (bez CMs). Tydzień 2 był warunkiem stosowania środków zaradczych (CMs)

**Środki Zaradcze (Countermeasures - CMs):** Środki zaradcze polegały na przekształceniu bodźców nierelatywnych (irrelevants) w ukryte cele (probe), aby wywołać P300 i zmniejszyć różnicę między P300 sondy/probe a P300 nierelatywnego. Uczestnikom polecono skojarzenie specyficznych, ukrytych reakcji behawioralnych (np. wciskanie palca, ruszanie palcem u nogi) z każdym z czterech bodźców nierelatywnych i wykonanie tych reakcji _przed_ naciśnięciem przycisku "Widziałem to"

**Pomiary i Analiza Danych (EEG/ERP):**
• **Rejestracja:** EEG rejestrowano elektrodami Ag/AgCl w Fz, Cz i Pz (analiza w Pz).
• **P300:** Mierzono metodą od szczytu do szczytu (p-p), która jest uznawana za najczulszą. Algorytm wyszukiwał maksymalną pozytywność (500 do 800 ms) i maksymalną negatywność (od latencji P300 do 1300 ms).
• **Diagnoza Indywidualna:** Stosowano **Metodę Różnicy Amplitudy Bootstrapped**. Standardowa metoda porównywała P300 sondy/probe ze średnią P300 wszystkich nierelatywnych (Iall). Bardziej rygorystyczna metoda, użyta po raz pierwszy, porównywała P300 sondy/probe z **Maksymalnym Nierelatywnym (Imax)** P300. Diagnoza była pozytywna przy 90% pewności.
• **Wykrywanie CMs za pomocą CR:** Wprowadzono algorytm kwalifikujący trafienie (RT-qualified hit) w oparciu o czas reakcji: jeśli CR nierelatywnego był znacząco wyższy niż CR sondy, diagnozowano użycie CMs

**Główne Wyniki:**
1. **Odporność na CMs (ERP):** W przeciwieństwie do starszych protokołów, CTP zachował diagnostyczną różnicę P300 między sondą a bodźcami nierelatywnymi (irrelevants) nawet w Tygodniu 2 (CM). Co więcej, użycie CMs nie tylko zwiększyło P300 nierelatywnych (irrelevants), ale także spowodowało wzrost P300 sondy (nowe zjawisko), co zrekompensowało próbę oszustwa.
2. **Wykrywanie CMs (CR):** Użycie CMs zostało skutecznie wykryte u wszystkich badanych w Tygodniu 2 poprzez znaczący wzrost czasu reakcji na pierwszy bodziec (S1).
3. **Wysoka Dokładność Diagnostyczna:** Indywidualne wskaźniki trafień były bardzo wysokie. W najbardziej rygorystycznym teście (P300 sondy/probe vs. Maksymalny P300 nierelatywnego/irrelevant, z kwalifikacją CR) wskaźnik trafień w Tygodniu 2 (CM) wyniósł **83%**, a w tygodniach bez CMs **92%**
4. **Niski Wskaźnik Fałszywych Pozytywów:** Wskaźnik fałszywych pozytywów w grupie niewinnej był bliski zeru (0–8%), a wskaźniki dyskryminacji A0 (Grier, 1971) osiągnęły wartość **0.95 do 0.98**.
5. **Replikacja:** Wyniki zostały potwierdzone w badaniu replikacyjnym, wykorzystującym bardziej realistyczne warunki uczenia się CMs, utrzymując wysoką dokładność (np. 100% trafień P vs. Iall w Tygodniu 2)


---
## EEG – Analiza P300 (CIT / Oddball)

### 1. Charakter reakcji: wiele pików P300

"Prior to the run, the participantwas alerted thatmissing more than one of these check-ups would result in test failure. This tended to discourage simpleCMs such as vision blurring. The detailed trial events diagrammed in Figure 1 indicate a probe–target trial. Also shown is a hypothetical ERP channel. Note that because this diagram is of a probe–target trial, an early P300 in response to the probe is shown, followed by a later P300 in response to the
target.We emphasize that the later P300 is of interest only in this first report to establish that the target did indeed function as a target normally does (forcing attention and eliciting a P300), but the key variable of interest with respect to concealed information detection is the response (or lack of same) to the first probe or irrelevant stimulus."

W niektórych typach prób (np. **probe–target trials**) w jednym przebiegu mogą pojawić się **dwa wyraźne komponenty P300**:
- **Pierwszy P300** – generowany w odpowiedzi na bodziec *probe*, czyli informację znaczącą lub rozpoznawaną przez uczestnika (np. znajomy przedmiot lub szczegół scenariusza).
- **Drugi P300** – generowany w odpowiedzi na bodziec *target*, wymagający reakcji (np. naciśnięcia przycisku).

Zazwyczaj analizie poddaje się **pierwszy komponent P300**, ponieważ to on odzwierciedla proces rozpoznania informacji utajonej. Drugi komponent służy jedynie potwierdzeniu, że bodziec docelowy prawidłowo wymusza uwagę i wywołuje typową odpowiedź poznawczą.

---

### 2. Pomiar amplitudy: metoda Peak-to-Peak (p–p)

"The algorithm searched within a window from 500 to 800 ms for the maximally positive segment average of 100 ms. The midpoint of the maximum positivity segment defined P300 latency. After the algorithm finds the maximum positivity, it searches from this P300 latency to 1300 ms for the maximum 100-ms negativity. The difference between the maximum positivity and negativity defines the p-p measure."

Amplituda P300 może być obliczana metodą **peak-to-peak (p–p)**.  
W tym podejściu:
1. Wyszukuje się w sygnale EEG (zazwyczaj w kanale Pz) **maksimum dodatnie** w oknie 500–800 ms po bodźcu.  
   - Wykorzystywany jest średni sygnał z 100 ms segmentów.  
   - Środek segmentu o największej średniej dodatniej wartości definiuje **latencję P300**.
2. Następnie, w oknie latencja–1300 ms, wyszukuje się **maksimum ujemne** (największą negatywność).
3. Amplituda
$$
A_{P300} = V_{\text{max\_pos}} - V_{\text{max\_neg}}
$$
Metoda ta eliminuje wpływ powolnych dryfów i zapewnia bardziej stabilny pomiar między uczestnikami.

---

### 3. Analiza i obsługa błędów (Error Handling)

"Standard analyses of variance (ANOVAs) were run to determine group effects. Any within-subject tests with > 1 df resulted in our use of the Greenhouse–Geisser (GG) corrected value of probability, p(GG), and the associated epsilon (e) value. All error trials (as well as artifact trials) were discarded and replaced so that analyses were done only on error free trials. (An error occurred when the subject pressed the wrong buttonFin terms of the instructions to a given stimulus.) This was also true for the within-subject analyses described in the next paragraph."

W analizie danych EEG z eksperymentów typu CIT obowiązują następujące zasady:
- **Eliminacja błędnych prób:**  
  Wszystkie próby, w których uczestnik nacisnął nieprawidłowy przycisk lub nie zareagował zgodnie z instrukcją, są **odrzucane**.
- **Eliminacja artefaktów:**  
  Przebiegi z artefaktami (np. ruchy oczu, napięcie mięśniowe, zakłócenia > ±75 μV) są **usuwane i zastępowane** nowymi próbami, aby zachować równą liczbę powtórzeń w każdej kategorii.
- **Analiza statystyczna:**  
  Dla efektów wewnątrzosobniczych stosuje się **korekcję Greenhouse–Geisser (GG)**, raportując wartość *p(GG)* oraz współczynnik *ε*.
---

### 4. Bootstrapping (within-individual)

"One thus bootstraps these distributions, in the bootstrap variation used here, as follows: A computer programgoes through the combined probe–target and probe nontarget set (all single sweeps) and draws at random, with replacement, a set of n1 waveforms. It averages these and calculates P300 amplitude from this single average using the maximum segment selection method as described above for the p-p index. Then a set of n2 waveforms is drawn randomly with replacement from the irrelevant set, from which an average P300 amplitude is calculated. The number n1 is the actual number of accepted probe (target and nontarget) sweeps for that subject, and n2 is the actual number of accepted irrelevant sweeps for that subject multiplied by a fraction (about .23 on average across subjects in the present report), which reduces the number of irrelevant trials to within one trial of the number of probe trials. The calculated irrelevant mean P300 is then subtracted from the comparable probe value, and one thus obtains a difference value to place in a distribution that will contain 100 values after 100 iterations of the process just described. Multiple iterations will yield differing (variable) means and mean differences due to the sampling-with-replacement process."

Metoda **bootstrapowa** służy do oceny różnic P300 między kategoriami (*probe* vs *irrelevant*) w obrębie jednego uczestnika.  
Procedura:
1. **Losowanie z powtórzeniami**:
   - Z puli pojedynczych prób *probe* losuje się *n₁* sygnałów.
   - Z puli *irrelevant* losuje się *n₂* sygnałów (zazwyczaj ~23% wszystkich, by liczebność była zbliżona).
1. **Średni przebieg**:
   - Dla każdej puli oblicza się średnie ERP i amplitudę P300 metodą *peak-to-peak*.
1. **Różnica amplitud**:
$$
\Delta A = A_{probe} - A_{irrelevant}
$$
2. **Iteracja procesu**:
   - Kroki 1–3 są powtarzane 100–1000 razy.
   - Powstaje rozkład różnic amplitud, z którego oblicza się miarę wykrycia (np. ROC AUC).

Metoda ta pozwala oszacować **stabilność i istotność różnic indywidualnych**, unikając błędów wynikających z małej liczby prób.

---
### 5. Interpretacja ogólna
- **Wielokrotne komponenty P300** → mogą wynikać z dwóch bodźców w jednym przebiegu (S1 i S2).  
- **Peak-to-peak** → zapewnia precyzyjny pomiar amplitudy niezależny od dryfów sygnału.  
- **Eliminacja błędnych prób** → zwiększa wiarygodność wyników statystycznych.  
- **Bootstrapping** → umożliwia estymację detekcji winy na poziomie jednostkowym.

---

## Analiza jakościowa ERP (opis wyników)

- **Różnica target vs. nontarget** nie ma znaczenia dla detekcji ukrytej informacji (CIT); oba typy odpowiedzi wykazują podobny pierwszy komponent P300 po bodźcach probe i irrelevant.  
- **Drugi (późniejszy) P300**, wyraźniejszy w odpowiedziach na bodźce *target*, wynika z wymagań zadania poznawczego, a nie z procesów związanych z ukrywaniem informacji.
- **Najważniejsze zjawisko:** w tygodniach 1 i 3, gdy uczestnicy **nie stosowali kontr-środków (Countermeasures, CM)**, amplituda P300 dla bodźców *probe* była **większa niż dla irrelevantów** — co potwierdza skuteczność CIT.
- W **tygodniu 2**, kiedy część uczestników próbowała stosować **ukryte kontr-środki**, pojawiły się wyraźne komponenty P300 również w odpowiedziach na *irrelevanty*, co wskazuje, że:
  - uczestnicy próbowali świadomie „maskować” reakcje,
  - metoda CIT może być częściowo podatna na takie działania.
- Mimo to, amplituda P300 dla *probe* w tygodniu 2 **nie uległa redukcji**, a nawet była większa niż w tygodniu 1 — efekt detekcji utrzymał się mimo prób kamuflażu.
- W tygodniu 3 zauważono pewne **efekty upływu czasu**, jednak różnice *probe–irrelevant* pozostały potencjalnie diagnostyczne.

---
## ERPs: Analiza ilościowa danych grupowych (ANOVA)

### Opis analizy:
Przeprowadzono **trójczynnikową analizę wariancji (ANOVA 2×2×3)** dla średnich amplitud P300 z czynnikami:
1. **Typ bodźca:** probe vs. irrelevant  
2. **Rodzaj prób:** target vs. nontarget  
3. **Czas:** trzy tygodnie (Week 1–3)
### Wyniki statystyczne:
- **Efekt typu bodźca (probe vs. irrelevant)** – bardzo silny: *F(1,11) = 62.1, p < .001*  
  → Potwierdza, że *probe* wywoływały istotnie większe P300 niż *irrelevants*.  
- **Efekt target vs. nontarget** – nieistotny: *F(1,11) < 1, p > .4*  
  → Brak różnic między próbami targetowymi i nontargetowymi.  
- **Efekt tygodnia (weeks)** – istotny: *F(2,22) = 12.3, p(GG) < .002, ε = .80*  
  → Amplitudy P300 zmieniały się w czasie eksperymentu.  
- **Interakcja probe–irrelevant × weeks** – istotna: *F(2,22) = 6.6, p(GG) < .008, ε = .93*  
  → Zmiana w amplitudach była różna dla probe i irrelevantów; wzrost irrelevantów w 2. tygodniu odpowiadał użyciu kontr-środków (CM).  
- Pozostałe interakcje (z udziałem target/nontarget) – nieistotne.

---
## ERPs: Analiza ilościowa danych indywidualnych (Hit Rates)

### Opis metody:
Dla każdego uczestnika obliczano **indywidualne wskaźniki detekcji** (hit rates) na podstawie **procedury bootstrap**.  
Porównywano średnią amplitudę P300 dla bodźców *probe* z średnią amplitudą wszystkich *irrelevants* w obrębie danego uczestnika.  
Analiza przeprowadzona była osobno dla trzech tygodni:
- **Week 1 i 3:** bez kontr-środków poznawczych (no-CM),
- **Week 2:** z kontr-środkami (CM week).
### Wyniki:
- **Skuteczność detekcji (Week 1 i 3):** 11 z 12 uczestników poprawnie wykrytych → **92% trafień**.  
- **Skuteczność (Week 2, CM):** 10 z 12 poprawnie wykrytych → niewielki spadek dokładności przy stosowaniu kontr-środków.  
- Jeden uczestnik miał **nietypowe latencje P300** oraz niskie amplitudy we wszystkich tygodniach. 

  Po dopasowaniu indywidualnego okna analizy („search window”), jego wyniki również osiągnęły trafną detekcję we wszystkich tygodniach.
### Wnioski:
- **Wysoka skuteczność detekcji (≈90–100%)** została utrzymana nawet przy stosowaniu kontr-środków poznawczych.  
- Zastosowanie **indywidualnego dopasowania parametrów P300 (latencji)** może zwiększyć dokładność analizy.  
- Jedna uczestniczka skutecznie „oszukała” test w tygodniu 2, jednak **jej profil czasów reakcji (RT)** ujawnił próbę stosowania kontr-środków.

---
## Reaction Time (RT): Analiza danych grupowych i indywidualnych

### Dane grupowe (Group Data)
W tygodniu 2 (z użyciem kontr-środków poznawczych, CMs) zaobserwowano istotny wzrost zarówno średniego RT, jak i jego wariancji, co wskazuje na większe obciążenie poznawcze i niestabilność procesów reakcji.
### Dane indywidualne (Individual Data)
**Wnioski:**
- W tygodniu z CMs (**Week 2**) średnie RT były wyraźnie **dłuższe**, a wariancja **większa**.  
- U trzech uczestników RT z tygodnia CM mieściły się w zakresie rozkładu z tygodnia bez CMs, lecz **wszyscy zostali poprawnie sklasyfikowani jako winni**.  
- U wszystkich badanych zaobserwowano **skrócenie RT** po zaprzestaniu stosowania CMs (Week 3).  
Zwiększona wariancja RT odzwierciedla **poznawczy koszt stosowania kontr-środków**

Wyniki te potwierdzają, że analiza RT jest **wiarygodnym wskaźnikiem prób świadomej manipulacji poznawczej**.

---
### ERPs: Quantitative Individual Data — Hit Rates Based on Probe vs. Largest Irrelevant (Imax)
W tej analizie zastosowano **najbardziej rygorystyczny sposób porównania** amplitud P300 w teście informacji ukrytej (CIT), polegający na zestawieniu amplitudy *probe P300* z **maksymalną amplitudą irrelevant P300 (Imax)** w ramach pojedynczego uczestnika. Metoda ta została dodatkowo zmodyfikowana w celu uwzględnienia **reakcji czasowych (RT)**, które wskazują na użycie kontr-środków poznawczych (CMs).
#### Algorytm diagnostyczny:
1. **Proste trafienie (Simple hit):**  
   Jeżeli amplituda *probe P300* była większa niż *Imax P300*, diagnozowano uczestnika jako „guilty”.  
   Dodatkowo, jeśli średni RT dla *irrelevants* był większy niż dla *probe*, notowano także użycie CM.
2. **RT-qualified hit:**  
   Jeśli nie uzyskano prostego trafienia, a niektóre *irrelevants* miały RT znacznie dłuższy niż *probe*, oznaczało to skuteczne użycie CM. Wówczas do porównania wybierano maksymalną *irrelevant P300* bez podwyższonego RT. Jeśli *probe P300* przewyższało tę wartość — diagnozowano „RT-qualified hit”.
3. **Pełne użycie CMs:**  
   Jeśli wszystkie *irrelevant RTs* były dłuższe od *probe RT*, interpretowano to jako pełne użycie kontr-środków. Wówczas porównywano *probe P300* z średnią wszystkich *irrelevants*, co dawało końcową diagnozę „RT-qualified hit”.
#### Uzasadnienie modyfikacji:
Autorzy argumentowali, że porównywanie *probe P300* z *irrelevant P300* nie ma sensu, jeśli dane RT wskazują, że bodziec *irrelevant* był objęty skutecznym CM. Dlatego zastosowano **RT-screening** jako filtr, aby zapewnić bardziej trafną interpretację aktywności P300. Mimo że zmniejsza to rygor analizy, metoda pozostaje konserwatywna i pozwala lepiej wykrywać próby manipulacji.
#### Wyniki i interpretacja:
- W tygodniach **bez CMs (Weeks 1 i 3)** prosta metoda (Step 1) była wystarczająca do prawidłowej klasyfikacji wszystkich uczestników jako „guilty”.  
- W tygodniu **z CMs (Week 2)** dokładność spadła, ale po zastosowaniu rozszerzonej analizy z uwzględnieniem RT (RT-qualified hits) **skuteczność diagnozy wzrosła**.  
- We wszystkich przypadkach wykryto użycie kontr-środków — RT w Week 2 były istotnie dłuższe niż w Week 1, co potwierdza ich aktywne stosowanie.  

Połączenie analizy ERP i RT umożliwia **rozróżnienie między „guilty despite CM use” a „noncooperative with CM use”**, co zwiększa wiarygodność wyników testu CIT.

---
## Grupa niewinna: fałszywe alarmy i błędy

**W skrócie:** Grupa niewinna nie reagowała różnicowo na bodźce związane z przestępstwem, fałszywe alarmy były niskie, a błędy pojawiały się głównie przy rozróżnianiu target vs. nontarget, a nie probe vs. irrelevant.

---
### Podsumowanie
**W skrócie:** CTP lepiej angażuje uwagę na probe dzięki odłożeniu decyzji target/nontarget i dodatkowo CM zwiększa koncentrację, co prowadzi do wzrostu P300 w porównaniu do starszych protokołów.
#### Dalsze badania
**Badanie alternatywnych strategii CM (countermeasures)**  
   - Warto sprawdzić, co się stanie, jeśli uczestnik **celowo opóźni reakcję „I saw it”** dla probe, aby ukryć różnice w czasie reakcji (RT).  
   - Można temu zapobiec poprzez **ograniczenie maksymalnego czasu reakcji (np. 1000 ms)**, co utrudnia świadome manipulowanie RT.  
   - Badania są tu niejednoznaczne – niektórzy autorzy wskazują, że przy dłuższych limitach (np. 1500 ms) manipulacja RT jest możliwa.  
   - W CTP czas reakcji (RT) służy nie do wykrywania informacji, lecz do **wykrywania użycia CM**.  
   - Próba celowego wydłużenia RT dla probe mogłaby **zwiększyć amplitudę P300**, co paradoksalnie ułatwiłoby wykrycie różnicy probe–irrelevant.