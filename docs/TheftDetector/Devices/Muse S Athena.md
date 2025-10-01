**Sprzęt**
https://eu.choosemuse.com/products/muse-s-athena?variant=54629657215311

**Dongle** (jeśli korzystamy z Windowsa do analizy pomiarów):  
[https://pl.farnell.com/silicon-labs/bled112-v1/bluetooth-module-v4-0-91dbm/dp/2930668?gross_price=true&utm_source=chatgpt.com](https://pl.farnell.com/silicon-labs/bled112-v1/bluetooth-module-v4-0-91dbm/dp/2930668?gross_price=true&utm_source=chatgpt.com "https://pl.farnell.com/silicon-labs/bled112-v1/bluetooth-module-v4-0-91dbm/dp/2930668?gross_price=true&utm_source=chatgpt.com")
- Czyli V1 z obsługą BLE i BGAPI (Interfejsy USB Host - emulacja portu Virtual COM)

### Specyfikacja
- **Sample Rate**: 256 Hz
- **EEG Channels**: 4 EEG channels + 4 amplified Aux channels
- **Reference Electrode Position**: FPz (CMS/DRL)
- **Channel Electrode Position**: TP9, AF7, AF8, TP10 (dry)
- **Aux Channel Connection**: Input range: 725uV AC signal (1.45mVp-p) with 1.45V DC offset
- **Charging Port**: USB-C (Upgraded)
### Pobranie sygnałów
1) Można skorzystać z darmowych narzędzi działających na Windows, Linux lub nawet MacOS (jest wsparcie dla Muse S ale nie jestem pewna co do wsparcia dla Muse S Athena - choć powinny być kompatybilne):
	- **BlueMuse** → potrzebny do utworzenia strumienia LSL (w Windows):  
		[https://github.com/kowalej/BlueMuse](https://github.com/kowalej/BlueMuse "https://github.com/kowalej/bluemuse")  
	- **LabRecorder** → do zapisu danych EEG do pliku:  
		[https://github.com/labstreaminglayer/App-LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder "https://github.com/labstreaminglayer/app-labrecorder")  
	- **pylsl** → opcjonalny — jeśli chcemy sterować eksperymentem, odbierać dane 'w kodzie':  
		[https://github.com/labstreaminglayer/pylsl/tree/main/src/pylsl/examples](https://github.com/labstreaminglayer/pylsl/tree/main/src/pylsl/examples "https://github.com/labstreaminglayer/pylsl/tree/main/src/pylsl/examples")

- Tutaj jest krótka instrukcja jak to działa (BlueMuse, pylsl):  
	[https://medium.com/@yang.zhao_22068/viewing-your-brain-with-muse-8f1584de00ad](https://medium.com/@yang.zhao_22068/viewing-your-brain-with-muse-8f1584de00ad "https://medium.com/@yang.zhao_22068/viewing-your-brain-with-muse-8f1584de00ad")

2) **Mind-monitor** - prostsza, lekko płatna (73 zł) apka/alternatywa do dostania się do surowych danych eeg (ale tylko na urządzenia mobilne) - na pewno wspiera Muse S i Muse S Athena:  
	https://mind-monitor.com/
	https://mind-monitor.com/FAQ.php#footnote-optics
https://mind-monitor.com/Technical_Manual.php


Użycie **dodatkowych AUX electrodes**:
https://mind-monitor.com/forums/viewtopic.php?t=1379
https://mind-monitor.com/forums/viewtopic.php?t=1379&start=40
https://mind-monitor.com/FAQ.php#footnote-raw