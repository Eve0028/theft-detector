## Skrócone podsumowanie

Badanie terenowe porównywało dwie metody analizy potencjałów ERP (P300 i P300-MERMER) w wykrywaniu ukrytej informacji dotyczącej **poważnych przestępstw terrorystycznych** i tajnych operacji, w celu weryfikacji hipotezy standardów naukowych brain fingerprinting. Zastosowano **Klasyfikacyjny Test Ukrytej Informacji (Classification CIT)**, który spełnia 20 standardów naukowych (m.in. wykorzystuje korelacje odpowiedzi na cele i sondy) oraz **Porównawczy CIT (Comparison CIT)**, który ich nie spełnia (m.in. porównuje tylko amplitudy sond i nieistotnych bodźców).

## P300-MERMER

**P300-MERMER** to skrót od **Memory and Encoding Related Multifaceted Electroencephalographic Response**

Źródła wskazują, że P300-MERMER jest bardziej kompleksową reakcją mózgową niż sam P300, obejmującą również późniejszy potencjał negatywny (LNP):
1. **Zakres czasowy analizy (Epoka):**
	- **P300** to wczesna pozytywna fala ERP, tradycyjnie analizowana w przedziale **300–900 ms** po bodźcu.
	- **P300-MERMER** (lub P300 plus LNP – _Late Negative Potential_) obejmuje szerszy zakres czasowy, zazwyczaj **300–1500 ms** po bodźcu.
2. **Składowe:**
    - **P300-MERMER** jest równoważny sumie szczytowych amplitud **P300** i **LNP** (późnego potencjału negatywnego). Amplituda P300-MERMER jest często definiowana jako różnica między najwyższym napięciem w oknie P300 (300–900 ms) a najniższym napięciem w oknie LNP (900–1500 ms).
3. **Wpływ na skuteczność testu:**
    - Wcześniejsze badania (np. Farwell i Donchin, 1991), które w analizie korelacji bootstrapowej uwzględniały **tylko P300** (300–900 ms), miały 0% błędów, ale wykazywały **12,5% wyników nieokreślonych (indeterminates)**.
    - Późniejsze badania, w których uwzględniano **pełny P300-MERMER** (lub P300 plus LNP, 300–1500 ms), osiągnęły **0% błędów i 0% wyników nieokreślonych**. Sugeruje to, że włączenie pełnej odpowiedzi mózgu (P300-MERMER) do analizy przyczynia się do większej pewności statystycznej i rzadszego występowania wyników nieokreślonych.
4. **Zastosowanie w standardach naukowych:**
    - Standardy naukowe brain fingerprinting (standard 15) zalecają przeprowadzanie **dwóch analiz** w środowisku sądowym: jednej z użyciem **tylko P300** (dla pewności spełnienia standardu ogólnej akceptacji w środowisku naukowym) oraz drugiej z użyciem **P300-MERMER** (aby zapewnić najnowocześniejsze podejście).

Podsumowując, **P300-MERMER** to **rozszerzona miara ERP**, która oprócz składowej P300 uwzględnia również późny potencjał negatywny (LNP), co w kontekście Klasyfikacyjnego CIT przyczynia się do **dokładniejszej klasyfikacji wzorców odpowiedzi mózgowej** i zwiększenia pewności statystycznej wyników

## Metody

1. **Klasyfikacyjny CIT (Classification CIT):** Ta metoda spełnia naukowe standardy brain fingerprinting. Polega na **analizie korelacji** (przy użyciu metody _bootstrapping_) między pełnymi wzorcami odpowiedzi mózgowej ERP na sondy a wzorcami odpowiedzi na cele oraz na bodźce nieistotne.
	- **IP:** Jeżeli odpowiedź na sondę jest bardziej **podobna** do odpowiedzi na cele (znane i istotne).
	- **IA:** Jeżeli odpowiedź na sondę jest bardziej **podobna** do odpowiedzi na bodźce nieistotne (nieznane).
	- W przypadku braku wysokiej pewności statystycznej w obu kierunkach, wynik jest **nieokreślony (indeterminate)**.
	- **Kryterium:** Ustalono, że pewność statystyczna musi wynosić **90%** zarówno dla oznaczeń IP, jak i IA.

2. **Porównawczy CIT (Comparison CIT):** Ta metoda nie spełnia standardów BFSS i **ignoruje odpowiedzi na cele**. Używa _bootstrappingu_ do obliczenia prawdopodobieństwa, że **amplituda** odpowiedzi ERP na sondy jest **większa** niż amplituda odpowiedzi na bodźce nieistotne.
	- **IP:** Jeżeli prawdopodobieństwo, że sonda jest większa niż nieistotne, jest >90%.
	- **IA:** Jeżeli prawdopodobieństwo, że sonda jest większa niż nieistotne, jest <90%.
	- Ta metoda **nie ma kategorii nieokreślonej**

## Wyniki

**Klasyfikacyjny CIT** osiągnął **0% wskaźnika błędu** i medianę pewności statystycznej wynoszącą **99,6%** (dla IP 99,9%; dla IA 98,6%). Ponadto, był on wysoce **odporny na kontrargumenty**.

**Porównawczy CIT** osiągnął **6% wskaźnika błędu**, a mediana jego pewności statystycznej dla osób IA wyniosła **48,7%**, czyli **poniżej poziomu losowego** (szansa). Ponad połowa oznaczeń IA w Porównawczym CIT była **nieważna** statystycznie.

Wyniki te silnie wspierają hipotezę, że standardy brain fingerprinting zapewniają warunki wystarczające do uzyskania **mniej niż 1% błędu** i **powyżej 95% pewności statystycznej**, co czyni Klasyfikacyjny CIT warunkiem koniecznym dla wiarygodnego zastosowania tej metody w terenie


