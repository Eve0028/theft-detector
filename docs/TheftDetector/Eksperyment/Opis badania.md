## Detekcja ukrytej wiedzy metodą P300-BAD

Celem eksperymentu jest sprawdzenie skuteczności wykrywania „ukrytej wiedzy” za pomocą niskokanałowego, konsumenckiego systemu EEG. Badanie opiera się na analizie potencjału wywołanego **P300**, który pojawia się w mózgu w odpowiedzi na bodźce istotne lub znane poznawczo.

### 1. Podział na grupy (Procedura „Mock Crime”)
Uczestnicy są losowo przydzielani do jednej z dwóch grup:

- **Grupa eksperymentalna („Złodziej”):** Uczestnik wchodzi do pokoju, otwiera pudełko i „kradnie” znajdujący się w nim przedmiot (np. biżuterię). Musi go dokładnie obejrzeć i zapamiętać, a następnie schować do kieszeni.
    
- **Grupa kontrolna:** Uczestnik wchodzi do tego samego pokoju jedynie w celu podpisania listy obecności. Nie zagląda do pudełka i nie zna jego zawartości.

### 2. Sesja EEG i paradygmat badawczy
Po etapie wprowadzającym następuje właściwy pomiar EEG. Uczestnik zakłada lekki czepek lub opaskę i wykonuje zadanie komputerowe składające się z ok. **400 prób** (ok. 21 minut czystego pomiaru).

Każda pojedyncza próba (trial) składa się z prezentacji dwóch bodźców:
**Bodziec S1 (Obraz):**
- Wyświetlany przez 0,4 s. Może to być przedmiot skradziony (**Probe**) lub przedmioty neutralne (**Irrelevant**).
- **Zadanie:** Uczestnik zawsze naciska klawisz „Z” lewą ręką.
- **Cel:** Wywołanie reakcji mózgowej na rozpoznanie znanego przedmiotu u „złodzieja”.

**Bodziec S2 (Ciąg cyfr):**
- Pojawia się 1,0–1,5 s po obrazie.
- **Target:** Rzadki ciąg (np. „11111”) – wymaga naciśnięcia klawisza „M” prawą ręką.
- **Nontarget:** Częste ciągi (np. „22222”) – wymagają naciśnięcia klawisza „N” prawą ręką.
- **Cel:** Utrzymanie wysokiego poziomu zaangażowania uwagi i kontrola poprawności wykonywania zadania.

### 3. Struktura czasowa próby

|**Etap**|**Czas trwania**|**Działanie uczestnika**|
|---|---|---|
|**Punkt fiksacji (+)**|0,5 s|Skupienie wzroku|
|**Bodziec S1 (Obraz)**|0,4 s|Reakcja klawiszem "Z"|
|**Przerwa**|1,0 - 1,5 s|Oczekiwanie|
|**Bodziec S2 (Cyfry)**|do 1,0 s|Reakcja "M" (Target) lub "N" (Nontarget)|
|**Interwał między próbami**|0,5 - 0,8 s|Odpoczynek|

### 4. Hipoteza i analiza danych (Metoda BAD)

Głównym założeniem jest to, że u osób z grupy „złodziei” prezentacja przedmiotu skradzionego (S1-Probe) wywoła znacznie wyższą amplitudę fali **P300** (w przedziale 300–600 ms) niż przedmioty neutralne. Analiza statystyczna metodą **BAD** (Bootstrapped Amplitude Difference) pozwoli ocenić, czy różnica ta jest na tyle wyraźna, by przy użyciu taniego sprzętu EEG skutecznie odróżnić „sprawcę” od osoby niewinnej.