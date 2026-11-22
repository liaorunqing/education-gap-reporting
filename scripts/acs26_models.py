"""Mature, fixed-parameter predictors and explicit working-model bootstrap."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit,digamma
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

class LogisticMixed:
 """Weighted Bernoulli random-intercept GLMM, Laplace integration."""
 def _calc(self,beta,lt,X,y,w,g,G):
  tau=np.exp(lt);iv=1/tau**2;xb=X@beta;u=np.zeros(G)
  for _ in range(80):
   p=expit(xb+u[g]);h=np.bincount(g,weights=w*p*(1-p),minlength=G)+iv
   step=(np.bincount(g,weights=w*(y-p),minlength=G)-u*iv)/h
   step=np.clip(step,-2,2);u+=step
   if np.max(np.abs(step))<1e-9:break
  p=expit(xb+u[g]);v=w*p*(1-p);h=np.bincount(g,weights=v,minlength=G)+iv
  ll=np.sum(w*(y*(xb+u[g])-np.logaddexp(0,xb+u[g])))-.5*np.sum(u*u*iv)-G*lt-.5*np.log(h).sum()
  Hx=np.column_stack([np.bincount(g,weights=v*X[:,j],minlength=G) for j in range(X.shape[1])])
  du=-Hx/h[:,None];t=v*(1-2*p)
  tx=np.column_stack([np.bincount(g,weights=t*X[:,j],minlength=G) for j in range(X.shape[1])])
  ts=np.bincount(g,weights=t,minlength=G)
  grad=X.T@(w*(y-p))-.5*np.sum((tx+ts[:,None]*du)/h[:,None],axis=0)
  dut=2*u*iv/h;gradt=np.sum(u*u*iv-1-.5*(ts*dut-2*iv)/h)
  return ll,grad,gradt,u,h
 def fit(self,X,y,w,domains):
  X=np.column_stack([np.ones(len(X)),X]);w=np.asarray(w,float);w=w/w.mean();y=np.asarray(y,float)
  self.domain_names,group=np.unique(domains,return_inverse=True);G=len(self.domain_names);N=len(X)
  self.center=X.mean(0);self.scale=X.std(0);self.center[0]=0;self.scale[0]=1;self.scale[self.scale<1e-8]=1
  X=(X-self.center)/self.scale
  def objective(b):
   ll,gb,gt,_,_=self._calc(b[:-1],b[-1],X,y,w,group,G)
   pen=.001*np.sum(b[1:-1]**2)/2
   grad=-np.r_[gb,gt]/N;grad[1:-1]+=.001*b[1:-1]
   return -ll/N+pen,grad
  b=np.zeros(X.shape[1]+1);b[0]=logit(np.average(y,weights=w));b[-1]=np.log(.2)
  r=minimize(objective,b,jac=True,method='L-BFGS-B',bounds=[(None,None)]*(len(b)-1)+[(np.log(.01),np.log(2))],options={'maxiter':300,'ftol':1e-10,'gtol':1e-6,'maxls':40})
  assert r.success, r.message
  self.beta=r.x[:-1];self.tau=float(np.exp(r.x[-1]));_,_,_,u,h=self._calc(self.beta,r.x[-1],X,y,w,group,G)
  self.effects={k:(float(a),float(1/b)) for k,a,b in zip(self.domain_names,u,h)}
  self.audit={'success':bool(r.success),'iterations':int(r.nit),'tau':self.tau,'gradient_max':float(np.abs(r.jac).max()),'method':'weighted Laplace GLMM, beta ridge .001, normalized likelihood weights; not published GMERF replication'}
  return self
 def predict_proba(self,X,domains):
  X=np.column_stack([np.ones(len(X)),X]);X=(X-self.center)/self.scale
  uv=[self.effects.get(k,(0,self.tau**2)) for k in domains];u=np.array([a for a,b in uv]);v=np.array([b for a,b in uv])
  nodes,ww=hermgauss(20);p=np.sum(expit((X@self.beta+u)[:,None]+np.sqrt(2*v)[:,None]*nodes)*ww,axis=1)/np.sqrt(np.pi)
  return np.column_stack([1-p,p])

AGRID=np.r_[0,np.logspace(-8,0,129)]
def fh_fit(y,D,x):
 """Batch REML on an identical fixed grid, re-estimating beta and A."""
 single=np.ndim(y)==1;y=np.atleast_2d(y);D=np.maximum(np.atleast_2d(D),1e-9)
 if len(D)==1 and len(y)>1:D=np.repeat(D,len(y),axis=0)
 mus=[];aud=[]
 for start in range(0,len(y),16):
  yy=y[start:start+16];dd=D[start:start+16];v=dd[:,None,:]+AGRID[None,:,None];w=1/v
  s0=w.sum(2);s1=(w*x).sum(2);s2=(w*x*x).sum(2);t0=(w*yy[:,None,:]).sum(2);t1=(w*yy[:,None,:]*x).sum(2)
  det=np.maximum(s0*s2-s1*s1,1e-20)
  b0=(t0*s2-t1*s1)/det;b1=(t1*s0-t0*s1)/det
  residual=yy[:,None,:]-b0[:,:,None]-b1[:,:,None]*x
  loss=np.log(v).sum(2)+(w*residual**2).sum(2)+np.log(det)
  k=loss.argmin(1);ii=np.arange(len(yy));A=AGRID[k];bb0=b0[ii,k];bb1=b1[ii,k]
  gamma=A[:,None]/(A[:,None]+dd);mu=gamma*yy+(1-gamma)*(bb0[:,None]+bb1[:,None]*x)
  vv0=s2[ii,k]/det[ii,k];vv1=-s1[ii,k]/det[ii,k];vv2=s0[ii,k]/det[ii,k]
  plugin=A[:,None]*dd/(A[:,None]+dd)+(1-gamma)**2*(vv0[:,None]+2*vv1[:,None]*x+vv2[:,None]*x*x)
  mus.append(mu);aud.extend([{'A':float(a),'beta0':float(c),'beta1':float(d),'plugin_variance':e} for a,c,d,e in zip(A,bb0,bb1,plugin)])
 mu=np.concatenate(mus)
 return (mu[0],aud[0]) if single else (mu,aud)

def smooth_variance(D,neff,x):
 single=np.ndim(D)==1;D=np.atleast_2d(D);nu=np.maximum(4,neff-1)
 X=np.column_stack([np.ones(len(x)),np.log(np.maximum(neff,2)),np.log(np.maximum(x*(1-x),1e-4))])
 center=X.mean(0);std=X.std(0);center[0]=0;std[0]=1;std[std<1e-8]=1;X=(X-center)/std
 penalty=np.diag([0,.1,.1]);inv=np.linalg.pinv(X.T@X+penalty)@X.T
 z=np.log(np.maximum(D,1e-9))-(digamma(nu/2)-np.log(nu/2))
 b=z@inv.T;out=np.exp(np.clip(b@X.T,-20,0))
 return out[0] if single else out

def fh_intervals(y,D,x,neff,B,seed):
 """Raw plug-in, beta/A refit bootstrap, and GVF/noisy-variance bootstrap."""
 rng=np.random.default_rng(seed);raw,ar=fh_fit(y,D,x);smooth=smooth_variance(D,neff,x);sm,ass=fh_fit(y,smooth,x)
 out={'FH_plugin_diagnostic':{'point':raw,'intervals':{}},'FH_parametric_bootstrap':{'point':raw,'intervals':{}},'FH_GVF_variance_bootstrap':{'point':sm,'intervals':{}}}
 for name,mu,A,Dwork,joint in [('FH_parametric_bootstrap',raw,ar,D,False),('FH_GVF_variance_bootstrap',sm,ass,smooth,True)]:
  theta=A['beta0']+A['beta1']*x+rng.normal(size=(B,len(x)))*np.sqrt(A['A'])
  obs=theta+rng.normal(size=(B,len(x)))*np.sqrt(Dwork)
  if joint:
   nu=np.maximum(4,neff-1);dhat=Dwork*rng.chisquare(nu,size=(B,len(x)))/nu
   ds=smooth_variance(dhat,neff,x)
  else:ds=np.repeat(np.asarray(Dwork)[None,:],B,axis=0)
  fitted,_=fh_fit(obs,ds,x);err=fitted-theta
  for level in [.8,.9,.95]:
   alpha=1-level;lo=mu-np.quantile(err,1-alpha/2,axis=0);hi=mu-np.quantile(err,alpha/2,axis=0)
   out[name]['intervals'][level]=(np.clip(lo,0,1),np.clip(hi,0,1))
 for level in [.8,.9,.95]:
  z=norm.ppf((1+level)/2);se=np.sqrt(np.maximum(ar['plugin_variance'],0))
  out['FH_plugin_diagnostic']['intervals'][level]=(np.clip(raw-z*se,0,1),np.clip(raw+z*se,0,1))
 return out,{'A_raw':ar['A'],'A_GVF':ass['A'],'D_raw_mean':float(np.mean(D)),'D_GVF_mean':float(np.mean(smooth)),'bootstrap_draws':B}
