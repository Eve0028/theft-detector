Użycie **dodatkowych AUX electrodes**:
https://mind-monitor.com/forums/viewtopic.php?t=1379
https://mind-monitor.com/forums/viewtopic.php?t=1379&start=40
https://mind-monitor.com/FAQ.php#footnote-raw

## 1. Podstawowa zasada
- Muse S Athena ma **4 kanały AUX** dostępne przez **złącze USB-C**.
- Każdy kanał odpowiada konkretnym pinom USB-C (np. A8/B8, A10/B10, A3/B3, A2/B2).
- Do każdego kanału AUX można podłączyć **zewnętrzną elektrodę EEG** (snappy pad).
- Do poprawnego działania porzeba też **referencji (R)** i **biasu (B)** — te są zapewniane przez samą opaskę (wewnętrzne elektrody), więc nie trzeba ich dodawać osobno.

---
## 2. Lista elementów do kupienia
### a) Kabel / złącze USB-C z dostępem do pinów
- Zwykły przewód USB-C → USB-A **nie wystarczy**, bo nie ma fizycznego dostępu do pinów A8, A10, A3, A2.
- Potrzeba:
    - **płytkę breakout USB-C** (tzw. _USB-C breakout board with solder pads for all pins_), np. na AliExpress, Sparkfun, Adafruit albo Digikey.
    - Alternatywnie: **USB-C plug for DIY soldering** (trudniejsze, bo trzeba samemu lutować mikropiny).
- Ważne: połączyć **obie strony symetryczne** (np. A8 z B8), aby wtyczka działała niezależnie od orientacji.

### b) Przewody ekranowane (shielded cable)
- Polecane w wątku: **Digikey ARF2168-ND** (przewód koaksjalny).
- Dlaczego: sygnały EEG mają amplitudy rzędu **mikrowoltów**, więc bez ekranu przewód łapie zakłócenia z otoczenia.

### c) Złącza elektrod (snap connectors)
- **Snaps 13 mm (female)** → montujesz na przewodzie.
- Do tego: **żelowe elektrody z męskim snapem** (dostępne np. na Amazon: "EEG snap electrodes", "Ag/AgCl snap electrodes").

### d) Dodatki montażowe
- **Koszulki termokurczliwe** (strain relief, izolacja).
- **Klej na gorąco** (obudowanie lutów i wzmocnienie mechaniczne).
- Opcjonalnie: **obudowa drukowana 3D** na breakout board, żeby chronić lutowania.

---
## 3. Lutowanie i podłączenie
1. **Pin sygnału AUX** → lutuj żyłę sygnałową przewodu ekranowanego.
    - Kanały AUX w Muse S Athena (MS-03, USB-C):
        - Aux1 = A8/B8
        - Aux2 = A10/B10
        - Aux3 = A3/B3
        - Aux4 = A2/B2
2. **Ekran przewodu** → lutuj do masy USB-C (A1, B1, A12, B12).
3. **Drugi koniec przewodu** → żyła sygnałowa do snap female, ekran zabezpieczony termokurczką (nie podłącz go do snap!).
4. Snap → do elektrody żelowej → przyklejonej na głowie.

---
## 4. Co to daje
- Po podłączeniu 1–4 takich przewodów można dołożyć dodatkowe EEG np. na C3, C4, Cz, Pz.
- Muse S Athena staje się **8-kanałowym systemem EEG** (4 wbudowane + 4 AUX).
- Wszystkie sygnały są próbkowane synchronicznie (256 Hz).

---
## 5. Podsumowanie: lista zakupowa
1. **USB-C breakout board z dostępem do wszystkich pinów** (lub DIY USB-C plug).
2. **Przewód ekranowany** (np. Digikey ARF2168-ND, albo dowolny cienki przewód koaksjalny audio/mikrofonowy).
	https://www.digikey.ca/en/products/detail/amphenol-rf/A-1PA-113-01KB2/4271615
		Czy są odpowiednie do EEG, biorąc pod uwagę ich przeznaczenie to RF?
3. **Snaps 13 mm (female)** + **żelowe elektrody (male)**.
4. **Koszulki termokurczliwe** + **klej na gorąco**.

