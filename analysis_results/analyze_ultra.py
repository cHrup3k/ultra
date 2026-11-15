from pathlib import Path
import os, json, hashlib, csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/ultra_mpl')
import numpy as np
from scipy import signal, ndimage
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'mic8.pdm'
FS_PDM=2500000
FS=250000
raw=SRC.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
bits=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='big')
x=bits.astype(np.float32)*2-1
n,beta=signal.kaiserord(100,25000/(FS_PDM/2)); n+=n%2==0
h=signal.firwin(n,112500,fs=FS_PDM,window=('kaiser',beta))
y=signal.resample_poly(x,1,10,window=h).astype(np.float32)
np.savez_compressed(OUT/'decoded_pcm.npz',pcm=y,fs=FS)
wavfile.write(OUT/'mic8_250k_float.wav',FS,y)
f,t,S=signal.spectrogram(y,FS,window='hann',nperseg=2048,noverlap=1792,mode='psd')
db=lambda a:10*np.log10(np.maximum(a,1e-25))
valid=(t>.02)&(t<len(y)/FS-.02)
avg=S[:,valid].mean(axis=1); peak=np.quantile(S[:,valid],.995,axis=1)
fig,axs=plt.subplots(2,1,figsize=(14,8))
for ax,lim in zip(axs,[(0,100),(20,65)]):
    m=(f>=lim[0]*1000)&(f<=lim[1]*1000)
    im=ax.pcolormesh(t,f[m]/1000,db(S[m]),shading='auto',cmap='magma',vmin=-155,vmax=-75)
    ax.set(ylim=lim,xlabel='Czas [s]',ylabel='Częstotliwość [kHz]')
fig.colorbar(im,ax=axs,label='PSD [dB re PDM ±1 / Hz]');fig.savefig(OUT/'spectrogram.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(13,4));ax.plot(f/1000,db(avg),label='Średnia PSD');ax.plot(f/1000,db(peak),label='Percentyl 99.5% ramek');ax.set(xlim=(0,100),xlabel='Częstotliwość [kHz]',ylabel='PSD [dB/Hz]');ax.grid();ax.legend();fig.tight_layout();fig.savefig(OUT/'spectrum.png',dpi=170);plt.close(fig)
peaks,_=signal.find_peaks(db(peak),prominence=6,distance=8)
peaks=peaks[(f[peaks]>20000)&(f[peaks]<100000)]
rank=peaks[np.argsort(peak[peaks])[::-1]][:20]
rows=[dict(hz=float(f[i]),mean_db=float(db(avg[i])),peak_percentile_db=float(db(peak[i])),contrast_db=float(db(peak[i]/avg[i]))) for i in rank]
(OUT/'spectral_candidates.json').write_text(json.dumps(rows,indent=2))
print('duration',len(bits)/FS_PDM,'candidates',rows[:12],flush=True)
np.savez_compressed(OUT/'spectrogram_data.npz',f=f,t=t,psd=S)
summary=dict(source=str(SRC),sha256=digest,bytes=len(raw),pdm_fs_nominal=FS_PDM,pcm_fs=FS,duration_s=len(bits)/FS_PDM,ones_percent=float(bits.mean()*100),fir_taps=n,candidates=rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
assert hashlib.sha256(SRC.read_bytes()).hexdigest()==digest

# Ton widoczny poniżej 20 kHz: dokładniejsza estymacja FFT i obwiednia.
fa,pa=signal.welch(y[int(2.3*FS):int(3.1*FS)],FS,nperseg=65536)
band=(fa>14000)&(fa<18000); ii=np.flatnonzero(band)[np.argmax(pa[band])]
delta=.5*(np.log(pa[ii-1])-np.log(pa[ii+1]))/(np.log(pa[ii-1])-2*np.log(pa[ii])+np.log(pa[ii+1]))
tone=float((ii+delta)*FS/65536)
z=signal.sosfiltfilt(signal.butter(4,[tone-500,tone+500],btype='bandpass',fs=FS,output='sos'),y)
env=np.sqrt(ndimage.uniform_filter1d(z*z,size=250))
baseline=float(np.median(env[(np.arange(len(y))/FS<2)&(np.arange(len(y))/FS>.1)]))
active=(env>max(baseline*10,float(np.max(env))*.2))&(np.arange(len(y))/FS>.02)&(np.arange(len(y))/FS<len(y)/FS-.02)
ids=np.flatnonzero(active); start=float(ids[0]/FS); end=float(ids[-1]/FS)
summary['tone']=dict(hz=tone,start_s=start,end_s=end,duration_s=end-start,amplitude_ratio_db=float(20*np.log10(np.median(env[active])/baseline)))
# Krótkie okna wychwytują bursty, które zanikłyby w średniej PSD.
fig,axs=plt.subplots(3,1,figsize=(14,10))
axs[0].plot(np.arange(len(y))[::125]/FS,env[::125]);axs[0].set(xlabel='Czas [s]',ylabel='RMS',title=f'Obwiednia tonu {tone:.1f} Hz (pasmo ±500 Hz)');axs[0].grid()
checks=[]
for ax,win in zip(axs[1:],[256,2048]):
    fq,tq,pq=signal.spectrogram(y,FS,window='hann',nperseg=win,noverlap=win*3//4)
    ref=np.median(pq,axis=1)
    relative=db(pq/np.maximum(ref[:,None],1e-25))
    m=(fq>=20000)&(fq<=100000); inner=(tq>.02)&(tq<len(y)/FS-.02)
    view=relative[m]; ax.pcolormesh(tq,fq[m]/1000,view,shading='auto',vmin=-5,vmax=15,cmap='magma')
    ax.set(xlabel='Czas [s]',ylabel='kHz',title=f'PSD / mediana czasowa każdego binu; okno {win/FS*1000:.3f} ms, skala −5…15 dB')
    # Moc całego pasma, stabilność i największy skok względem mediany.
    powers={}
    for lo,hi in [(20000,100000),(35000,45000),(38000,42000),(65000,80000)]:
        bp=pq[(fq>=lo)&(fq<=hi)].sum(axis=0)*(fq[1]-fq[0]); v=bp[inner]
        powers[f'{lo}-{hi}']=dict(max_over_median_db=float(db(v.max()/np.median(v))),p99_over_median_db=float(db(np.quantile(v,.99)/np.median(v))),max_time_s=float(tq[inner][np.argmax(v)]))
    checks.append(dict(window_ms=win/FS*1000,bands=powers))
fig.tight_layout();fig.savefig(OUT/'burst_search.png',dpi=170);plt.close(fig)
summary['burst_checks']=checks
fh,H=signal.freqz(h,worN=131072,fs=FS_PDM)
summary['filter_passband_error_db']=float(np.max(np.abs(20*np.log10(np.maximum(abs(H[fh<=100000]),1e-20)))))
summary['filter_stopband_attenuation_db']=float(-20*np.log10(max(abs(H[fh>=125000]))))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(tone=summary['tone'],checks=checks),indent=2))

# Lokalizacja powtarzalnych krótkich zdarzeń widocznych po wybieleniu PSD.
zb=signal.sosfiltfilt(signal.butter(4,[69000,76000],btype='bandpass',fs=FS,output='sos'),y)
eb=ndimage.uniform_filter1d(zb*zb,size=125)
base=float(np.median(eb[int(.02*FS):-int(.02*FS)]))
pk,_=signal.find_peaks(eb,height=base*6,distance=int(.03*FS),prominence=base*4)
pk=pk[(pk>.02*FS)&(pk<len(y)-.02*FS)]
events=[]
fig,axs=plt.subplots(3,1,figsize=(14,10))
axs[0].plot(np.arange(len(y))[::125]/FS,db(eb[::125]/base));axs[0].plot(pk/FS,db(eb[pk]/base),'rx');axs[0].set(xlabel='Czas [s]',ylabel='Moc / mediana [dB]',title='Kandydaci burstów 69–76 kHz');axs[0].grid()
for p in pk:
    a=max(0,p-1000);b=min(len(y),p+1000)
    ff0,pp=signal.welch(y[a:b],FS,nperseg=512)
    mm=(ff0>68000)&(ff0<77000); freq=float(ff0[mm][np.argmax(pp[mm])])
    l=p;r=p
    while l>0 and eb[l]>eb[p]/2:l-=1
    while r<len(y)-1 and eb[r]>eb[p]/2:r+=1
    events.append(dict(time_s=float(p/FS),peak_over_background_db=float(db(eb[p]/base)),frequency_hz=freq,power_fwhm_ms=float((r-l)/FS*1000)))
if len(pk):
    p=pk[np.argmax(eb[pk])];a=max(0,p-int(.008*FS));b=min(len(y),p+int(.008*FS))
    fq,tq,pq=signal.spectrogram(y[a:b],FS,nperseg=256,noverlap=240)
    mm=(fq>=60000)&(fq<=85000)
    axs[1].pcolormesh((tq+a/FS-p/FS)*1000,fq[mm]/1000,db(pq[mm]),shading='auto',cmap='magma');axs[1].set(xlabel='Czas względem maksimum [ms]',ylabel='kHz',title=f'Najsilniejszy kandydat przy {p/FS:.6f} s')
    axs[2].plot((np.arange(a,b)-p)/FS*1000,zb[a:b]);axs[2].set(xlabel='Czas względem maksimum [ms]',ylabel='Amplituda',title='Przebieg w paśmie 69–76 kHz')
fig.tight_layout();fig.savefig(OUT/'ultrasonic_events.png',dpi=170);plt.close(fig)
summary['ultrasonic_events']=events
summary['event_interval_s']=np.diff(pk/FS).tolist()
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'events.csv').write_text('time_s,peak_over_background_db,frequency_hz,power_fwhm_ms\n'+'\n'.join(','.join(str(v) for v in e.values()) for e in events))
print('EVENTS',json.dumps(events))
