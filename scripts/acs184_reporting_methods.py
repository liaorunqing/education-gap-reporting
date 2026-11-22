"""Finite-size sampling and compatible estimators, independently testable."""
import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.special import expit,logit
from scipy.stats import binom

def integer_allocation(N,score,m,lower=None):
    N=np.asarray(N,int);score=np.maximum(np.asarray(score,float),1e-12)
    lower=np.minimum(N,2) if lower is None else np.asarray(lower,int)
    assert 0<=m<=N.sum() and lower.sum()<=m and (lower<=N).all()
    out=lower.copy();weight=N*score
    # Continuous bounded allocation followed by objective-aware integer adjustment.
    def counts(t):return np.clip(t*weight,lower,N)
    hi=1.
    while counts(hi).sum()<m-1e-8:hi*=2
    t=brentq(lambda t:counts(t).sum()-m,0,hi,xtol=1e-13) if m>lower.sum() else 0.
    out=np.floor(counts(t)+1e-10).astype(int)
    while out.sum()<m:
        gain=np.where(out<N,weight**2/(np.maximum(out,1)*(out+1)),-np.inf)
        out[np.argmax(gain)]+=1
    while out.sum()>m:
        loss=np.where(out>lower,weight**2/(np.maximum(out-1,1)*np.maximum(out,1)),np.inf)
        out[np.argmin(loss)]-=1
    assert out.sum()==m and (out>=lower).all() and (out<=N).all()
    return out

def group_indices(g):return [np.flatnonzero(g==k) for k in range(int(np.max(g))+1)]

def strat_estimate(groups,n,T,rng):
    H=len(T);point=np.zeros(T.shape[1]);variance=point.copy();indices=[]
    for ix,nn in zip(groups,n):
        N=len(ix)
        if N==0:continue
        assert nn>=min(2,N)
        s=ix if nn==N else ix[rng.choice(N,int(nn),replace=False)]
        indices.extend(s);weight=N/H
        point+=weight*T[s].mean(0)
        if nn<N:variance+=weight**2*(1-nn/N)*T[s].var(0,ddof=1)/nn
    return point,variance,np.array(indices,int)

def two_wave(groups,m,T,prior_sd,rng):
    H=len(T);N=np.array([len(g) for g in groups]);n1=integer_allocation(N,np.ones(len(N)),max(int(.2*m),int(np.minimum(N,2).sum())))
    first=[];remaining=[];sd=[];point=np.zeros(T.shape[1])
    for k,(ix,n) in enumerate(zip(groups,n1)):
        s=ix if n==len(ix) else ix[rng.choice(len(ix),int(n),replace=False)]
        first.extend(s);point+=T[s].sum(0)/H
        keep=np.ones(len(ix),bool);keep[np.searchsorted(ix,s)]=False;remaining.append(ix[keep])
        v=T[s,0].var(ddof=1) if len(s)>1 else prior_sd[k]**2
        sd.append(np.sqrt((max(len(s)-1,0)*v+20*prior_sd[k]**2)/(max(len(s)-1,0)+20)))
    rem=np.array([len(g) for g in remaining]);lower=np.minimum(rem,2)
    assert lower.sum()<=m-n1.sum()
    n2=integer_allocation(rem,sd,int(m-n1.sum()),lower)
    var=np.zeros(T.shape[1]);selected=list(first)
    for ix,n in zip(remaining,n2):
        Nrem=len(ix)
        if Nrem==0:continue
        s=ix if n==Nrem else ix[rng.choice(Nrem,int(n),replace=False)]
        selected.extend(s);point+=Nrem/H*T[s].mean(0)
        if n<Nrem:var+=(Nrem/H)**2*(1-n/Nrem)*T[s].var(0,ddof=1)/n
    assert len(selected)==m and len(set(selected))==m
    # E[estimate | entire first wave] equals the same fixed finite-population mean.
    return point,var,np.array(selected,int)

def _conv(a,b,m):
    c=fftconvolve(a,b)[:m+1]
    return np.maximum(c,0)

def conditional_means(N,p,m):
    pmf=[binom.pmf(np.arange(min(int(n),m)+1),int(n),float(pp)) for n,pp in zip(N,p)]
    prefix=[np.ones(1)]
    for a in pmf:prefix.append(_conv(prefix[-1],a,m))
    suffix=[None]*(len(N)+1);suffix[-1]=np.ones(1)
    for j in range(len(N)-1,-1,-1):suffix[j]=_conv(pmf[j],suffix[j+1],m)
    prob=prefix[-1][m];assert prob>1e-8,(m,prob)
    ans=[]
    for j,a in enumerate(pmf):
        other=_conv(prefix[j],suffix[j+1],m);k=np.arange(len(a));valid=(m-k>=0)&(m-k<len(other))
        ans.append(np.dot(k[valid]*a[valid],other[m-k[valid]])/prob)
    return np.array(ans),float(prob)

def cps_setup(N,score,m,rho=0):
    N=np.asarray(N,int);score=np.maximum(np.asarray(score,float),1e-12)
    assert (N>0).all() and 0<m<N.sum()
    fraction=m/N.sum();floor=.1*fraction
    def target(t):return np.clip(t*score,floor,.95)
    hi=1.
    while np.dot(N,target(hi))<m:hi*=2
    t=brentq(lambda t:np.dot(N,target(t))-m,0,hi,xtol=1e-14)
    pi=(1-rho)*target(t)+rho*fraction
    lp=logit(pi)
    for it in range(60):
        shift=brentq(lambda c:np.dot(N,expit(lp+c))-m,-50,50)
        p=expit(lp+shift);mu,accept=conditional_means(N,p,m)
        error=float(np.max(abs(mu/N-pi)))
        if error<2e-10:break
        lp+=logit(pi)-logit(np.clip(mu/N,1e-12,1-1e-12))
    else:raise ArithmeticError(('CPS marginal calibration did not converge',error))
    return dict(N=N,p=p,pi=pi,m=m,marginal_error=error,calibration_iterations=it+1,acceptance=accept)

def cps_counts(setup,R,rng):
    chunks=[];n=0;attempts=0
    while n<R:
        B=min(200000,max(2000,int((R-n)/setup['acceptance']*1.1)))
        a=rng.binomial(setup['N'],setup['p'],size=(B,len(setup['N'])))
        keep=a[a.sum(1)==setup['m']];chunks.append(keep);n+=len(keep);attempts+=B
        assert attempts<10_000_000
    return np.concatenate(chunks)[:R]

def cps_estimate(groups,counts,pi,T,rng):
    chosen=[];probs=[]
    for ix,n,p in zip(groups,counts,pi):
        if n:
            selected=ix if n==len(ix) else ix[rng.choice(len(ix),int(n),replace=False)]
            chosen.extend(selected);probs.extend([p]*n)
    s=np.asarray(chosen,int);p=np.asarray(probs);H=len(T);n=len(s)
    values=T[s];point=np.sum(values/p[:,None],axis=0)/H
    c=n/(n-1)*(1-p)
    beta=np.sum(c[:,None]*values/p[:,None],axis=0)/c.sum()
    variance=np.sum(c[:,None]*(values-p[:,None]*beta)**2/p[:,None]**2,axis=0)/H**2
    return point,variance,s

def srs_greg(Y,F,s,H,Fmean=None):
    y=Y[s];f=F[s];fc=f-f.mean(0);yc=y-y.mean(0);den=np.sum(fc*fc,0)
    beta=np.divide(np.sum(fc*yc,0),den,out=np.zeros(Y.shape[1]),where=den>1e-20)
    point=y.mean(0)+beta*((F.mean(0) if Fmean is None else Fmean)-f.mean(0))
    variance=(1-len(s)/H)*(y-beta*f).var(0,ddof=1)/len(s)
    return point,variance

def srs_greg_cells(Y,Z,s,H,Zmean=None):
    x=Z[s];y=Y[s];xm=x.mean(0);xc=x-xm;ym=y.mean(0);yc=y-ym
    scale=np.maximum(xc.std(0),1e-12);design=xc/scale
    solution,_,rank,_=np.linalg.lstsq(design,yc,rcond=1e-10);coefficient=solution/scale[:,None]
    point=ym+((Z.mean(0) if Zmean is None else Zmean)-xm)@coefficient
    residual=y-x@coefficient
    variance=(1-len(s)/H)*residual.var(0,ddof=1)/len(s)
    return point,variance,int(rank)

def cps_greg_cells(Y,Z,s,p,H,Zmean):
    weights=1/p;wn=weights/weights.sum();x=Z[s];y=Y[s];xm=wn@x;ym=wn@y
    xc=x-xm;yc=y-ym;scale=np.maximum(np.sqrt(wn@(xc*xc)),1e-12)
    sol,_,rank,_=np.linalg.lstsq(xc/scale*np.sqrt(weights[:,None]),yc*np.sqrt(weights[:,None]),rcond=1e-10)
    coefficient=sol/scale[:,None];intercept=ym-xm@coefficient
    e=y-intercept-x@coefficient
    point=intercept+Zmean@coefficient+(e/p[:,None]).sum(0)/H
    n=len(s);c=n/(n-1)*(1-p);betadev=np.sum(c[:,None]*e/p[:,None],axis=0)/c.sum()
    variance=np.sum(c[:,None]*(e-p[:,None]*betadev)**2/p[:,None]**2,axis=0)/H**2
    return point,variance,int(rank)

def checks():
    import itertools
    rng=np.random.default_rng(739);N=np.array([3,4,2]);setup=cps_setup(N,[1.,2.,.5],4)
    units=np.repeat(np.arange(3),N);p=setup['p'][units]
    weights=[];members=[]
    for combo in itertools.combinations(range(9),4):
        mask=np.zeros(9,bool);mask[list(combo)]=True
        weights.append(np.prod(np.where(mask,p,1-p)));members.append(mask)
    pr=np.array(weights);pr/=pr.sum();members=np.array(members)
    marginal=pr@members
    assert np.max(abs(marginal-setup['pi'][units]))<1e-9
    T=np.column_stack([np.arange(9.)-2,np.ones(9)])
    estimates=(members@(T/marginal[:,None]))/9
    assert np.max(abs(pr@estimates-T.mean(0)))<1e-10
    for m in [3,4,7,9]:
        n=integer_allocation(N,[1,2,.5],m,lower=np.ones(3,int));assert n.sum()==m
    # Exhaustive one-stratum two-wave check including conditional variance.
    H=7;m1=2;m2=2;x=np.arange(7.)**2;est=[];vv=[]
    for a in itertools.combinations(range(H),m1):
        rest=np.setdiff1d(np.arange(H),a)
        for b in itertools.combinations(rest,m2):
            b=np.array(b);est.append((x[list(a)].sum()+len(rest)*x[b].mean())/H)
            vv.append((len(rest)/H)**2*(1-m2/len(rest))*x[b].var(ddof=1)/m2)
    assert abs(np.mean(est)-x.mean())<1e-10 and abs(np.var(est)-np.mean(vv))<1e-10
    constant=np.ones((100,2));_,v,_=strat_estimate([np.arange(100)],[30],constant,rng);assert (v==0).all()
    values=np.arange(60.).reshape(30,2);ix=np.arange(12)
    setup_uniform=cps_setup([30],[1.],12)
    pp,vv,ss=cps_estimate([np.arange(30)],[12],setup_uniform['pi'],values,np.random.default_rng(1))
    assert np.allclose(vv,(1-12/30)*values[ss].var(0,ddof=1)/12,rtol=1e-8)
    gg=[np.arange(20),np.arange(20,40)];tt=np.random.default_rng(4).normal(size=(40,2))
    _,_,selected=two_wave(gg,20,tt,[1.,1.],np.random.default_rng(5))
    first=selected[:4];changed=tt.copy();remaining=np.setdiff1d(np.arange(40),first);changed[remaining]=1000*np.random.default_rng(6).normal(size=(len(remaining),2))
    _,_,selected2=two_wave(gg,20,changed,[1.,1.],np.random.default_rng(5))
    assert np.array_equal(selected,selected2),'Unobserved labels changed adaptive selection'
    return dict(CPS_enumerated_marginal_error=float(np.max(abs(marginal-setup['pi'][units]))),CPS_HT_unbiased=True,two_wave_conditional_estimator_unbiased=True,two_wave_variance_exact_enumeration=True,constant_population_zero_variance=True,uniform_CPS_variance_equals_SRS=True,unsampled_label_perturbation_leaves_two_wave_selection_unchanged=True)
