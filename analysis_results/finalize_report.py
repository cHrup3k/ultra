from pathlib import Path
import json
import numpy as np
from scipy import signal, ndimage
import nbformat as nb
O=Path(__file__).resolve().parent
s=json.loads((O/'summary.json').read_text());d=np.load(O/'decoded_pcm.npz');fs=int(d['fs']);y=d['pcm']
z=signal.sosfiltfilt(signal.butter(4,[65000,80000],btype='bandpass',fs=fs,output='sos'),y)
e=ndimage.uniform_filter1d(abs(signal.hilbert(z))**2,size=11)
widths=[];specs=[];ff=np.fft.rfftfreq(2048,1/fs)
for event in s['ultrasonic_events']:
    p=int(event['time_s']*fs);p=p-100+np.argmax(e[p-100:p+100]);l=p;r=p
    while l>0 and e[l]>e[p]/2:l-=1
    while r<len(e)-1 and e[r]>e[p]/2:r+=1
    widths.append((r-l)/fs*1000)
    specs.append(abs(np.fft.rfft(y[p-128:p+128]*np.hanning(256),2048))**2)
sp=np.mean(specs,axis=0);m=(ff>68000)&(ff<77000)
s['fine_event_power_fwhm_ms']=widths;s['mean_event_spectrum_peak_hz']=float(ff[m][np.argmax(sp[m])])
(O/'summary.json').write_text(json.dumps(s,indent=2))
report='''# Analiza mic8.pdm — czujnik parkowania

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
'''
(O/'summary.md').write_text(report)
book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(report),nb.v4.new_code_cell("from pathlib import Path\nimport runpy\nfrom IPython.display import display, Image\nroot = Path.cwd()\nif not (root / 'analysis_results/analyze_ultra.py').exists():\n    root = root / 'ultra'\nrunpy.run_path(str(root / 'analysis_results/analyze_ultra.py'))\nrunpy.run_path(str(root / 'analysis_results/finalize_report.py'))\nfor name in ['ultrasonic_events', 'burst_search', 'spectrogram', 'spectrum']:\n    display(Image(filename=str(root / 'analysis_results' / (name + '.png'))))")])
book.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
nb.write(book,O.parent/'ultra_analysis.ipynb')
print(report)
