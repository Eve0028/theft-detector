**Dryft napięcia stałego elektrody** (ang. _electrode DC offset drift_)

Obserwowany na wykresach powolny, jednokierunkowy trend linii bazowej to zjawisko o podłożu fizykochemicznym, znane jako dryf składowej stałej (DC). Zjawisko to występuje niezależnie od tego, czy stosuje się elektrody całkowicie suche, czy też wspomagane roztworami przewodzącymi. W przypadku systemów takich jak BrainAccess Mini, wykorzystujących pozłacane elektrody (typu _spike_ na owłosionej skórze głowy oraz płaskie na czole w roli elektrod Bias i Reference), zastosowanie preparatu SigmaSpray na elektrody płaskie modyfikuje dynamikę układu, jednak **nie eliminuje dryfu**.

Aby zrozumieć ten mechanizm, należy zdefiniować pojęcie potencjału półogniwa (ang. _half-cell potential_). Półogniwo to układ fizykochemiczny składający się z przewodnika elektronowego (metalu) kontaktującego się z przewodnikiem jonowym (elektrolicie). Na styku pozłacanej powierzchni elektrody płaskiej i skóry powstaje klasyczna granica faz. Wprowadzenie preparatu SigmaSpray – wysoce przewodzącego roztworu elektrolitycznego – sprawia, że przejmuje on rolę głównego elektrolitu, błyskawicznie obniżając impedancję i zastępując powolny proces gromadzenia się naturalnego potu.

Taki układ w dalszym ciągu zachowuje się jak miniaturowa bateria. Złoto jest metalem wysoce polaryzowalnym, co oznacza, że na granicy faz następuje nagromadzenie ładunków elektrycznych bez łatwej, odwracalnej wymiany jonowej z roztworem. Tworzy się tam elektryczna warstwa podwójna (warstwa Helmholtza), funkcjonująca jednocześnie jak kondensator i ciągłe źródło napięcia stałego. Napięcie to, osiągające wartości rzędu kilkudziesięciu do kilkuset miliwoltów (mV), jest o kilka rzędów wielkości potężniejsze niż właściwe, mikrowoltowe (µV) sygnały EEG.

Dryf jest nadal obecny mimo użycia SigmaSpray, ponieważ środowisko pomiarowe na czole pozostaje wysoce dynamiczne, a potencjał ulega ciągłym fluktuacjom z trzech głównych powodów:

1. **Parowanie i absorpcja:** Roztwór wodny z preparatu SigmaSpray stopniowo odparowuje do otoczenia, a część płynu jest powoli wchłaniana przez warstwę rogową naskórka (_stratum corneum_). Powoduje to ciągłą zmianę objętości i stężenia jonów w elektrolicie.
    
2. **Mieszanie z naturalnymi wydzielinami:** Aktywność gruczołów potowych i łojowych na czole sprawia, że nałożony sztuczny elektrolit miesza się z naturalnym potem, co nieustannie modyfikuje potencjał chemiczny roztworu na granicy faz.
    
3. **Fluktuacje impedancji:** Chociaż początkowa impedancja jest znacznie niższa i bardziej stabilna dzięki aplikacji płynu, jej wartość wciąż faluje w czasie pod wpływem wysychania, zmian temperatury i mikroruchów skóry względem płaskiej elektrody.
    

Te ciągłe zmiany fizykochemiczne są rejestrowane przez wzmacniacz jako potężna, wolnozmienna fala. Chociaż SigmaSpray drastycznie poprawia jakość połączenia i może przyspieszyć początkową stabilizację sygnału dla elektrod referencyjnych, potencjał półogniwa wciąż ulega powolnemu dryfowi.


Zjawisko dryfu składowej stałej (DC) jest znacznie bardziej uciążliwe, dynamiczne i osiąga większe amplitudy w przypadku elektrod suchych (takich jak złote elektrody w systemie BrainAccess) w porównaniu do tradycyjnych elektrod "mokrych" (np. chlorosrebrowych Ag/AgCl z żelem).

Wynika to z trzech kluczowych różnic fizykochemicznych na styku elektroda-skóra:

**1. Brak natychmiastowego bufora jonowego** W elektrodach mokrych żel przewodzący od razu tworzy stabilne, bogate w jony środowisko pomiarowe, które niweluje różnice właściwości naskórka. W elektrodach suchych początkowo nie ma elektrolitu. System "czeka", aż skóra pod metalem naturalnie wytworzy mikrowarstwę potu i wilgoci. W trakcie tego procesu (określanego jako czas stabilizacji – _settling time_), impedancja drastycznie spada, a potencjał półogniwa ulega powolnym, ale gigantycznym fluktuacjom, co rejestrowane jest jako potężny dryf linii bazowej.

**2. Polaryzowalność złota a wymiana jonowa** Klasyczne elektrody Ag/AgCl są niepolaryzowalne – reakcje chemiczne na ich powierzchni zachodzą łatwo, pozwalając na płynną wymianę jonów z żelem, co stabilizuje generowane napięcie stałe. Złoto jest natomiast metalem szlachetnym i wysoce polaryzowalnym. Nie wchodzi w odwracalne reakcje z jonami ze skóry. Interfejs elektroda-skóra zachowuje się w tym przypadku niemal wyłącznie jak kondensator (sprzężenie pojemnościowe). Akumulujące się ładunki elektryczne nie mogą zostać łatwo rozładowane przez reakcje chemiczne, co prowadzi do narastania i falowania wysokiego napięcia.



Obserwacja niemal liniowego, wznoszącego się trendu podczas 21-minutowej sesji pomiarowej jest zjawiskiem całkowicie normalnym i w pełni uzasadnionym prawami fizyki oraz elektrofizjologii.

Po pierwsze, fundamentalną rolę odgrywa tutaj **zjawisko ładowania pojemnościowego w połączeniu z prądem polaryzacji wzmacniacza**. Jak wyjaśniałam we wcześniejszych analizach, interfejs między złotą elektrodą a warstwą rogową naskórka (nawet przy obecności preparatów typu SigmaSpray) zachowuje się w ujęciu elektrycznym jako kondensator o znacznej impedancji. Wzmacniacze sygnałów biologicznych – w tym te najwyższej klasy, stosowane w systemach EEG – zawsze wykazują obecność tzw. wejściowego prądu polaryzacji (ang. _input bias current_). Jest to mikroskopijny, stały prąd upływu generowany przez układy elektroniczne urządzenia.

Gdy ten stały prąd przepływa przez strukturę o charakterze pojemnościowym (czyli styk elektroda-skóra), inicjuje proces powolnego, nieprzerwanego ładowania tego biologicznego „kondensatora”. Zgodnie z fundamentalnymi prawami elektrotechniki, ładowanie pojemności stałym prądem objawia się na wykresie jako liniowy przyrost napięcia w dziedzinie czasu. To właśnie z tego powodu rejestrowany sygnał systematycznie, przez wiele minut, wędruje w górę.

Po drugie, **kinetyka procesów fizjologicznych** na styku elektrody ze skórą ma charakter asymetryczny i wysoce długotrwały. Umieszczenie elektrody na skórze na okres 21 minut prowadzi do izolacji tego fragmentu naskórka, co skutkuje nieprzerwaną, jednokierunkową hydratacją (nawilżaniem) za sprawą naturalnej aktywności gruczołów potowych. Prowadzi to do powolnej, lecz monotonicznej zmiany stężenia jonów w utworzonym półogniwie. Zmiany te wymuszają ciągłe przesuwanie się potencjału elektrochemicznego w jednym kierunku, aż do momentu osiągnięcia fizjologicznego i termodynamicznego nasycenia (co często zajmuje ponad 30 minut).

**Dlaczego sygnał rośnie, zamiast maleć?** Kierunek dryfu (to, czy linia wznosi się w kierunku wartości dodatnich, czy opada w kierunku ujemnych) jest zjawiskiem deterministycznym, uwarunkowanym konkretną architekturą sprzętową i biologiczną. Zależy ściśle od:

1. Różnicy potencjałów chemicznych między konkretną elektrodą aktywną a elektrodą referencyjną (Reference).
    
2. Polaryzacji wewnętrznej samego wzmacniacza różnicowego w urządzeniu (w tym wspomnianego kierunku prądu _bias current_).