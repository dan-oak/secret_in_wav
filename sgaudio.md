
## Imports


```python
import numpy as np
import scipy as sp
import scipy.io.wavfile
```

## Settings


```python
fname = 'brilliant.wav'
```


```python
rate, channels = sp.io.wavfile.read(fname)
channels = channels.copy()
rate, channels.shape
```




    (22050, (14595,))



## Encode


```python
msg = "super secret message!"
msglen = 8 * len(msg)
msglen
```




    168




```python
seglen = int(2 * 2**np.ceil(np.log2(2*msglen)))
segnum = int(np.ceil(channels.shape[0]/seglen))
segnum, seglen
```




    (15, 1024)




```python
if len(channels.shape) == 1:
    channels.resize(segnum*seglen, refcheck=False)
    channels = channels[np.newaxis]
else:
    channels.resize((segnum*seglen, channels.shape[1]), refcheck=False)
    channels = channels.T
channels.shape
```




    (1, 15360)




```python
channels.dtype
```




    dtype('int16')




```python
msgbin = np.ravel([[int(y) for y in format(ord(x), '08b')] for x in msg])
msgPi = msgbin.copy()
msgPi[msgPi == 0] = -1
msgPi = msgPi * -np.pi/2
```


```python
segs = channels[0].reshape((segnum,seglen))
```


```python
segs = np.fft.fft(segs)
M = np.abs(segs)
P = np.angle(segs)
print(M[0,:3])
print(P[0,:3])
```

    [ 39859.          18443.44842853  40294.31380713]
    [ 0.          2.8657574   2.70047085]



```python
dP = np.diff(P, axis=0)
dP[0,:5]
```




    array([ 0.        , -5.94549832, -5.66979862,  3.56063878,  3.56078846])




```python
segmid = seglen // 2
P[0,-msglen+segmid:segmid] = msgPi
P[0,segmid+1:segmid+1+msglen] = -msgPi[::-1]
for i in range(1, len(P)): P[i] = P[i-1] + dP[i-1]
```


```python
segs = (M * np.exp(1j * P))
```


```python
segs = np.fft.ifft(segs).real
channels[0] = segs.ravel().astype(np.int16)
```


```python
sp.io.wavfile.write('steg_'+fname, rate, channels.T)
```

## Decode


```python
msglen = 8 * 4
seglen = 2*int(2**np.ceil(np.log2(2*msglen)))
segmid = seglen // 2
```


```python
if len(channels.shape) == 1:
    x = channels[:seglen]
else:
    x = channels[:seglen,0]
x = (np.angle(np.fft.fft(x))[segmid-msglen:segmid] < 0).astype(np.int8)
x = x.reshape((-1,8)).dot(1 << np.arange(8 - 1, -1, -1))
''.join(np.char.mod('%c',x))
```




    'asdf'


