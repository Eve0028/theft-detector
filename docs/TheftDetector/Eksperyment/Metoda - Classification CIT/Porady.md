Opis Badania Mock Crime z Metodą Classification CIT

### I. Przygotowanie Scenariusza (Mock Crime)

1. Tworzenie Grup IP i IA (Winny/Niewinny)
	- **Grupa IP ("Winni"):** Uczestniczą w całym scenariuszu "mock crime" i celowo **kodują/zapamiętują** szczegóły stanowiące Sondy.
	- **Grupa IA ("Niewinni"):** Nie uczestniczą w krytycznym etapie "zbrodni" (np. otwieranie sejfu i oglądanie zawartości), a zatem **nie posiadają wiedzy** zawartej w Sondach.

### II. Kluczowe Procedury Przed Testem (Wywiad Wstępny)

W tej fazie należy spełnić kluczowe standardy BF (Standardy 8, 9, 10).

Kiedy pytać o znanie Sond (Probes)?
O posiadanie wiedzy (potencjalnie zanieczyszczającej) należy pytać badanych **przed rozpoczęciem testu EEG**.

1. **Potwierdzenie Wiedzy o Celach (Targets)**: Pokazuje Pani badanemu listę Celów i opisuje ich znaczenie w kontekście scenariusza, upewniając się, że **znają** oni te cele.

2. **Upewnienie się co do Znaczenia Sond (Probes)**: Należy opisać badanemu **znaczenie** każdej sondy w kontekście badanego zdarzenia (np. "Jeden z tych przedmiotów był w środku sejfu"), ale **nie ujawniać, która opcja jest poprawna**.

3. **Weryfikacja Kontaminacji (Standard 9)**: Należy zapytać badanego, czy **z jakiegokolwiek powodu niezwiązanego z mock crime** wie, który bodziec w danej grupie (Sonda + Nieistotne) jest relewantny.
	- **Jeśli badany zna Sondę z innego źródła:** Taki bodziec musi zostać **wyeliminowany** ze zbioru.
	- **Zalecenie dodatkowych Sond:** Zdecydowanie należy przygotować **dodatkowe Sondy** (więcej niż wymagane minimum sześciu), aby mieć zapas na wypadek konieczności wyeliminowania niektórych bodźców z powodu zanieczyszczenia.

### III. Sesja EEG (Procedura Testowania)

Procedura musi spełniać Standardy BF, aby zmusić badanych (szczególnie tych "winnych") do przetwarzania bodźców.

1. **Zadanie Behawioralne (Standard 11, 12):** Badany musi nacisnąć **jeden przycisk** w odpowiedzi na **Cele (Targets)**, a **drugi przycisk w odpowiedzi na WSZYSTKIE pozostałe bodźce** (Sondy i Nieistotne).
	- To jawne zadanie behawioralne jest **konieczne** w przypadku motywowanych badanych (np. w pracy magisterskiej, gdzie mogą próbować ukryć wiedzę), ponieważ zmusza ich do przeczytania i przetworzenia każdej sondy, zanim zdecydują, który przycisk nacisnąć.

2. **Brak Instrukcji „Kłamstwa” (Standard 11, 12):** Nie należy instruować badanych, aby „kłamali” lub „mówili prawdę”. Fakt, że badany "winny" wciska ten sam przycisk dla SOND co dla bodźców NIEISTOTNYCH, jest _behawioralnie_ fałszywą reakcją (zaprzeczeniem rozpoznania), ale test BF mierzy nieświadomą reakcję mózgu na rozpoznanie (P300), a nie zachowanie behawioralne.

3. **Liczba Prób:** W celu uzyskania statystycznie solidnych wyników dla każdego pojedynczego badanego, należy dążyć do zebrania co najmniej **100 prób z Sondami** oraz równej liczby prób z Celami.

4. **Rejestracja ERP:** Pomiary EEG powinny być rejestrowane z elektrody **Pz** (parietalna linia środkowa), gdzie P300 jest maksymalne.

### IV. Czy badacze zaklasyfikowani do niewinnych powinni wchodzić na miejsce zbrodni?

**Nie**, zgodnie ze standardami Classification CIT, **nie powinni**.

• **Cel IA:** Rolą grupy "niewinnej" jest dostarczenie wzorca dla reakcji mózgu, gdy **informacja jest nieobecna (IA)**, czyli gdy bodziec jest postrzegany jako naprawdę nieistotny (Irrelevant).

• **Ryzyko "Innocent-but-Informed":** Jeśli osoby niewinne wejdą na miejsce zbrodni i zobaczą szczegóły (probes), ale nie popełnią "przestępstwa", stają się **"niewinnymi, ale poinformowanymi"**. Taka wiedza, nawet jeśli jest "niewinna", może wywołać reakcję P300, ponieważ informacja jest **znana**. Rosenfeld i współpracownicy stwierdzili w jednym z badań, że uczestnicy "niewinni, ale poinformowani" byli **praktycznie nieodróżnialni** od tych, którzy faktycznie popełnili "mock crime", co prowadziło do wysokiego wskaźnika fałszywych pozytywów.

Dlatego, aby uzyskać wiarygodną próbę kontrolną (IA), badani muszą być **odizolowani** od informacji zawartej w Sondach, w celu potwierdzenia, że brak wiedzy prowadzi do braku reakcji P300 na sondy.

### V. Inne Wskazówki i Uwagi

• **Analiza Danych:** Należy użyć algorytmu klasyfikacji opartego na **bootstrappingu korelacji (BCD)** (Classification CIT), który porównuje podobieństwo fali Sondy do fali Celu (IP) lub fali Bodźca Nieistotnego (IA). Jest to kluczowy element, który odróżnia tę metodę od Comparison CIT i zapewnia wysoką ufność IA.

• **Pomiar P300-MERMER:** Dla uzyskania najwyższej możliwej ufności statystycznej (jak wykazano w Badaniu 2014), należy mierzyć nie tylko sam potencjał P300, ale cały **P300-MERMER** (P300 plus późna komponenta negatywna, LNP), co wymaga wydłużenia epoki analizy do co najmniej **1800 ms**

• **Wizualna Modality:** Prezentacja bodźców w **modalności wizualnej** (słowa/frazy na ekranie) jest zazwyczaj bardziej efektywna niż modalność słuchowa w protokole CTP.

• **Ekspertyza (Saliency):** Badania pokazują, że P300 jest bardziej efektywne, gdy używa się bodźców o dużej **osobistej istotności (saliency)**, takich jak autobiograficzne detale (np. imiona, urodziny), w porównaniu do detali nabytych incydentalnie w "mock crime". Warto rozważyć, czy detale "kradzieży" są wystarczająco istotne dla badanych. Jeśli badane detale są słabo zakodowane (incidentalnie), czułość testu może być niższa.

• **Ground Truth jest Wiedzą, nie Winą:** Proszę pamiętać, że Pani badanie wykrywa **obecność lub brak konkretnej informacji** w mózgu (IP lub IA), a nie to, czy badany jest winny, kłamie czy jest nieuczciwy.