## Ustawienia ogólne
- Sprzęt: Muse S Athena lub BrainAccess Standard Kit, PC z PsychoPy, LSL do wysyłania markerów.
- Wejście uczestnika: klawiatura.
  - `Z` — odpowiedź na S1 (lewa ręka) — stała odpowiedź (wszyskie S1).
  - `M` — odpowiedź "target" na S2 (prawa ręka).
  - `N` — odpowiedź "nontarget" na S2 (prawa ręka).
- Bodźce S1: obiekty/obrazy (1 × probe, 4 × irrelevants). Każdy obiekt to możliwość kilku obrazów (kilku widoków).
- Bodźce S2: napisy/numery (np. "111111" jako target; "222222","333333"… jako nontargety).
---
## Struktura triala (kolejność i czasy)
1. **Fixation cross** — 500 ms  
2. **S1 (probe lub irrelevant)** — obraz, **400 ms**  
   - Pozycje pliku obrazka są logowane (filename, typ: probe/irrelevantID).  
   - Reakcje `Z` zbierane od S1_onset do końca ISI (loguj RT względem S1_onset).
3. **ISI (pusty ekran / fixation)** — losowo **1000–1500 ms** (używaj równomiernego rozkładu; średnio 1250 ms)  
4. **S2 (target / nontarget)** — tekst (np. "111111"), **300 ms**  
   - Reakcje `M` lub `N` zbierane od S2_onset do +1000 ms (loguj RT względem S2_onset).
5. **ITI (inter-trial interval)** — losowo **500–800 ms** (średnio 650 ms)  
6. (Co pewien czas, np. 5–10% triali): **pop-quiz o S1** — pytanie o to, co było S1 w ostatnich trialach (wymusza uwagę).  
7. Na każdy znacznik zdarzenia wysyłaj event marker do LSL: fixation_onset, S1_onset (z etykietą probe/irrelevantID), S1_response, S2_onset (target/nontarget), S2_response, ITI_start.
---
## Liczba bodźców i powtórzeń
- **Probe**: 1 obraz, powtarzany **80 razy**.  
- **Irrelevants**: 4 różne obrazy, każdy powtarzany **80 razy** → 4 × 80 = **320 irrelevants**.  
- **Łączna liczba S1 triali**: 80 + 320 = **400 triali S1**.
- **S2 (target/nontarget)**: S2 pojawia się w każdym trialu S1; stosunek target/nontarget = **20% target / 80% nontarget** → około **80 targetów** i **320 nontargetów** (liczby zaokrąglone zgodnie z losowaniem).
- **Bloki**: podziel sesję na **5 bloków po 80 triali**
  - Po każdym bloku daj przerwę ~2 min (skrócić/ wydłużyć w zależności od zmęczenia).
---
## Szczegóły logowania (co zapisywać do CSV dla każdego triala)
- `participant_id, session_id, block, trial_index, S1_type (probe/irr_ID), S1_filename, fixation_onset_time, S1_onset_time, S1_response_key, S1_RT, ISI_dur, S2_type (target/nontarget), S2_onset_time, S2_response_key, S2_RT, ITI_dur, notes`
- Dodatkowo loguj LSL_marker_id i unix timestamp (precision ms or better).

---
## Synchronizacja z urządzeniem EEG
- Wyślij markery LSL przy S1_onset, S1_response, S2_onset, S2_response. Zapisz timestampy lokalne i LSL.  
- Sprawdź przed nagraniem opóźnienia systemowe (jitter) i wykonaj próbkę testową, aby upewnić się, że markery pojawiają się w strumieniu EEG zsynchronizowane.

---
### Harmonogram sesji i estymacja czasu
- Fixation: 0.5 s
- S1: 0.4 s
- średnie ISI: 1.25 s (zakres 1.0–1.5 s)
- S2: 0.3 s
- średnie ITI: 0.65 s (zakres 0.5–0.8 s)
- **Czas jednego triala (średnio)** = 0.5 + 0.4 + 1.25 + 0.3 + 0.65 = **3.10 s**
**Łącznie triali S1 = 400**
- Czas wszystkich triali = 400 × 3.10 s = **1240 s** = **20.67 minut**

#### Podział na bloki i przerwy
- **5 bloków × 80 triali** każdy.
- Po każdym bloku: **przerwa 2 min**
- Dodatkowo po 2–3 blokach warto dać dłuższą przerwę (3–4 min) jeśli uczestnik zgłasza zmęczenie.

#### Dodatkowe elementy sesji:
- Instrukcje i zgoda: **10 min**
- Mock crime / encoding (uczestnik zapoznaje się z probe): **5 min**
- Montaż i kontrola sprzętu, kalibracja, krótki baseline: **15 min**
- Przerwy między blokami: 4 przerwy × 2 min = **8 min**  (dla 5 bloków)
- Krótkie sprawdzenie sygnału i ewentualne korekty między blokami (wliczone w przerwy, + ~2 min dodatkowo w trakcie sesji) **(opcjonalnie)**
- Debriefing i zakończenie: **5 min**

#### Sumaryczny szacunkowy czas sesji EEG jednej osoby (konserwatywnie)
- Triali: 20.67 min
- Instrukcje + setup + mock crime + debrief + przerwy: 10 + 15 + 5 + 5 + 8 = **43 min**
**Razem:** 20.67 + 43 = **63.67 minut ≈ 64 minut**