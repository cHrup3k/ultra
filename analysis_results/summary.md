# Analiza mic8.pdm — czujnik parkowania

W zapisie wykryto **31 krótkich, powtarzalnych zdarzeń ultradźwiękowych w okolicy 73 kHz**. To mocny kandydat na emisję czujnika. Sam jeden kanał bez nagrania kontrolnego z wyłączonym czujnikiem nie pozwala rozstrzygnąć, czy sygnał dociera akustycznie, czy jest zakłóceniem elektrycznym.

| Parametr | Wynik |
|---|---|
| Długość zapisu | 5.0001088 s |
| Liczba burstów | 31 |
| Pasmo zdarzeń | około 71.8–73.7 kHz (maksima pojedynczych krótkich FFT) |
| Maksimum uśrednionego widma zdarzeń | około 72.9 kHz |
| Odstęp między zdarzeniami | 152.8–165.9 ms; mediana 162.0 ms |
| Częstość powtórzeń | około 6.2 Hz |
| Długość impulsu, FWHM mocy | około 0.12–0.28 ms, mediana 0.19 ms |
| Wzrost mocy pasma 69–76 kHz | 14.5–17.4 dB względem mediany, po uśrednieniu 0.5 ms |

Pierwszy burst występuje przy 0.096236 s, ostatni przy 4.890220 s. Lista w `events.csv`. **Nie znaleziono jednoznacznego sweepu**: zdarzenia są krótkie, a rozdzielczość czasowo-częstotliwościowa nie wystarcza do wykluczenia niewielkiej modulacji wewnątrz pojedynczego burstu. Zmienność maksimum FFT między zdarzeniami nie dowodzi sweepu. Nie widać porównywalnie regularnej serii w paśmie 35–45 kHz.

Dodatkowo zarejestrowano silny, prawie stały ton **15 983.7 Hz**, od około **2.160 do 3.212 s** (1.051 s). Jego obwiednia jest około 41 dB ponad wcześniejszym tłem w paśmie ±500 Hz. Źródło tonu nie jest ustalone.

## Metoda i ograniczenia

Przyjęto nominalny zegar PDM **2.5 MHz** i kolejność bitów MSB-first, zgodnie z poprzednią analizą i dostępnym kodem przechwytywania. W folderze ultra jest wyłącznie mic8.pdm: brak osobnego potwierdzenia zegara, logu strat i wzorca czasu. Przy innym zegarze częstotliwości skalują się przez Fpdm/2.5MHz, a czasy odwrotnie.

PDM ±1 zdekodowano filtrem FIR Kaiser i decymacją ×10 do 250 kHz; pasmo analizy 0–100 kHz, zaporowe od 125 kHz, projekt 100 dB. To nie jest skalibrowana charakterystyka mikrofonu w ultradźwiękach. Poziomy odnoszą się do skali PDM, nie dB SPL. Rosnące tło wysokich częstotliwości jest zgodne z szumem kształtowanym PDM; samo w sobie nie stanowi detekcji emisji.

Spektrogram Hann: 2048 próbek (8.192 ms), krok 1.024 ms; kontrola krótkich zdarzeń: 256 próbek (1.024 ms), krok 0.256 ms. Podzielenie PSD każdego binu przez jego medianę czasową ujawnia serię około 73 kHz niewidoczną na zwykłej skali. Detektor: pasmo 69–76 kHz, średnia mocy 0.5 ms, maksimum ponad 6× medianę oraz prominencja ponad 4× medianę, minimalny odstęp 30 ms. Wyłączono pierwsze i ostatnie 20 ms z detekcji.

Długości z `events.csv` (~0.51 ms) są poszerzone przez uśrednianie detektora. Lepsze oszacowanie w tabeli powyżej pochodzi z obwiedni Hilberta pasma 65–80 kHz, wygładzonej przez 44 µs. Nadal obejmuje odpowiedź filtru i mikrofonu, więc nie jest dokładnym czasem sterowania przetwornikiem. Maksimum uśrednionej FFT około 72.9 kHz podano zaokrąglone; dopełnienie zerami nie zwiększa fizycznej rozdzielczości.

Źródło pozostało niezmienione; SHA256 zapisano w summary.json. Brak mapy strat uniemożliwia ocenę pełnej ciągłości zapisu. Przypisanie burstów do czujnika można sprawdzić przez porównanie nagrań z czujnikiem włączonym i wyłączonym przy niezmienionym torze PDM.

## Pliki i odtworzenie

- `ultrasonic_events.png`: wykryte bursty i powiększenie jednego zdarzenia.
- `burst_search.png`: obwiednia 16 kHz i spektrogramy względem tła.
- `spectrogram.png`, `spectrum.png`: pełne pasmo.
- `decoded_pcm.npz`, `mic8_250k_float.wav`: zdekodowany sygnał bez normalizacji.
- `summary.json`, `events.csv`: wyniki liczbowe.

Uruchom `analyze_ultra.py`, następnie `finalize_report.py`, albo notebook `../ultra_analysis.ipynb`. Wymagane numpy, scipy, matplotlib i nbformat; notebook dodatkowo IPython/Jupyter.
