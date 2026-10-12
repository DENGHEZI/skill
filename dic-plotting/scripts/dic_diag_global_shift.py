# -*- coding: utf-8 -*-
import numpy as np, os
from PIL import Image
from scipy.signal import fftconvolve

def integ(a):
    I=np.zeros((a.shape[0]+1,a.shape[1]+1),dtype=np.float64)
    I[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    return I
def winstats(II,IS,k):
    A=II[:-k,:-k]; B=II[k:,k:]; C=II[:-k,k:]; D=II[k:,:-k]
    s=B-D-C+A
    if IS is not None:
        A2=IS[:-k,:-k];B2=IS[k:,k:];C2=IS[k:,:-k];D2=IS[:-k,k:]
        sq=B2-D2-C2+A2
        mean=s/(k*k); var=np.maximum(sq/(k*k)-mean**2,0); return mean,np.sqrt(var)
    return s/(k*k), None

d = r'speckle_in/新建文件夹 (5)'
files = sorted([f for f in os.listdir(d) if f.endswith('_1.tiff')])
DS=4
def load(p): return np.asarray(Image.open(p))[::DS,::DS].astype(np.float64)
ref=load(os.path.join(d,files[0]))
h,w=ref.shape; cy,cx=h//2,w//2; R=192
T=ref[cy-R:cy+R, cx-R:cx+R]; ky,kx=T.shape
ts=T.std(); tsum=T.sum(); Tm=T.mean()
print(f'参考模板 {T.shape} std={ts:.1f}')

for f in files[1:]:
    cur=load(os.path.join(d,f))
    SR=110; pad=SR
    y_off=cy-pad-R; x_off=cx-pad-R
    S=cur[y_off:y_off+ky+2*SR, x_off:x_off+kx+2*SR]
    corr=fftconvolve(S,T[::-1,::-1],'valid')
    Sm,Ss=winstats(integ(S),integ(S*S),ky)
    C=np.clip((corr-Sm*tsum)/np.maximum(Ss*ts*ky*kx,1e-6),-1,1)
    pk=np.unravel_index(np.argmax(C),C.shape)
    print(f'{f}: 全局位移 dy={pk[0]-SR:+.0f} dx={pk[1]-SR:+.0f} ds-px  峰值ZNCC={C[pk]:.3f}')
