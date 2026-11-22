"""Python publication figures; source tables retained in full."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullLocator
from acs183_reporting_data import OUT,FINAL,loadjson,dump,sha
from acs186_reporting_run import METHODS

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300,'axes.unicode_minus':True})
LABELS={'SRS_HT':'Direct HT','SRS_historical_cell':'Historical cells','SRS_HGB':'HGB + SRS','SRS_GREG':'GREG + SRS',
 'proportional_HGB':'Proportional','Neyman_HGB':'Historical Neyman','two_wave_HGB':'Two-wave',
 'CPS_coarse':'CPS: coarse','CPS_RF':'CPS: RF risk','CPS_HGB':'CPS: HGB risk','CPS_robust_HGB':'Robust CPS','SRS_GREG_cells':'GREG: cells + HGB',
 'Neyman_working_fine':'Fine working Neyman','CPS_working':'Working-risk CPS','CPS_working_GREG':'Working CPS + GREG','CPS_RF_GREG':'RF CPS + GREG'}
COLORS=['#596873','#357b78','#cc8b42','#7e6b9f','#ba6572']

def save(fig,name,dest):
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(dest/f'{name}.svg',bbox_inches='tight')
    fig.savefig(dest/f'{name}.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

def make(phase):
    src=OUT/f'analysis/{phase}';dest=OUT/f'figures/{phase}'
    df=pd.read_csv(src/'point_interval_results.csv');rp=pd.read_csv(src/'reporting_results.csv');nr=pd.read_csv(src/'normal_and_matched_reference.csv')
    context=pd.read_csv(src/'population_context.csv');summary=pd.read_csv(src/'equal_state_summary.csv')
    select=loadjson(OUT/'development_selection.json');selected=select.get('selected_learned_method') or select['selected_method'] or 'SRS_HGB';comp=select['mature_comparator'] or 'SRS_HGB'
    panel_methods=list(dict.fromkeys(['SRS_HGB','Neyman_HGB',comp,selected,'CPS_HGB']))
    # All confirmation states, fixed order; development uses full state-year labels.
    order=FINAL if phase=='final' else context.state.tolist()
    context=context.set_index('state').loc[order].reset_index()
    n=len(context);fig,axes=plt.subplots(1,2,figsize=(7.2,max(3.5,.30*n)),gridspec_kw={'width_ratios':[1,1]})
    cols=['foreign_weight_share','older_weight_share','mixed_nativity_household_share']
    a=100*context[cols].to_numpy();im=axes[0].imshow(a,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    axes[0].set_xticks(range(3),['Foreign-born\nshare','Age 60+\nshare','Mixed-nativity\nhouseholds']);axes[0].set_yticks(range(n),[x.upper() for x in context.state]);axes[0].set_title('a  Population composition (%)',loc='left',fontweight='bold')
    for i in range(n):
        for j in range(3):axes[0].text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',color='black')
    b=100*context[['raw_gap','standardized_gap']].to_numpy();limit=max(5,float(np.ceil(abs(b).max()/5)*5))
    im2=axes[1].imshow(b,cmap='RdBu',norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit),aspect='auto')
    axes[1].set_xticks([0,1],['Raw gap','Age/sex-standardized']);axes[1].set_yticks(range(n),[x.upper() for x in context.state]);axes[1].set_title('b  Foreign minus native gap (pp)',loc='left',fontweight='bold')
    for i in range(n):
        for j in range(2):axes[1].text(j,i,f'{b[i,j]:+.1f}',ha='center',va='center',color='white' if abs(b[i,j])>.6*limit else 'black')
    fig.colorbar(im,ax=axes[0],shrink=.7,label='Share (%)');fig.colorbar(im2,ax=axes[1],shrink=.7,label='Gap (pp)')
    fig.subplots_adjust(wspace=.4);save(fig,'figure1_population_context',dest)
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.3))
    for j,name in enumerate(panel_methods):
        x=summary[(summary.endpoint=='standardized')&(summary.method==name)].sort_values('m')
        a=rp[(rp.endpoint=='standardized')&(rp.method==name)&(rp.tau==.02)].groupby('m').agg(wrong=('wrong_count','sum'),claims=('claim_count','sum')).reset_index()
        color=COLORS[j%len(COLORS)];kw=dict(color=color,lw=1.8,label=LABELS[name])
        axes[0,0].plot(x.m,x.equal_state_claim_rate*100,**kw)
        axes[0,1].plot(a.m,100*a.wrong/a.claims.replace(0,np.nan),**kw)
        axes[1,0].plot(x.m,x.equal_state_coverage95*100,**kw)
        axes[1,1].plot(x.m,x.equal_state_width95*100,**kw)
    titles=['a  Reports produced','b  Errors among produced reports','c  95% interval coverage','d  Mean 95% interval width']
    for ax,title in zip(axes.ravel(),titles):
        assert all(b>0 for b in [300,1000,3000])
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xscale('log');ax.xaxis.set_minor_locator(NullLocator());ax.set_xticks([300,1000,3000],['300','1,000','3,000']);ax.set_xlabel('Sampled residential units');ax.grid(axis='y',alpha=.15)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_ylim(0,100);ax.set_ylabel('Percent')
    axes[0,1].set_ylim(0,25)  # Fixed before confirmation; zero origin, explicit probability scale.
    axes[1,1].set_ylim(bottom=0);axes[1,1].set_ylabel('Percentage points');axes[1,0].axhline(95,color='#222222',ls=':',lw=1)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.025))
    fig.subplots_adjust(hspace=.45,wspace=.3,bottom=.15);save(fig,'figure2_reporting_budget',dest)
    # Matched reference and risk transport; no selected favorable observations.
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.2))
    a=nr[(nr.endpoint=='standardized')&(nr.m==1000)&(nr.method==comp)].copy();a['label']=a.year.astype(str)+' '+a.state.str.upper();a=a.sort_values(['year','state']);xx=np.arange(len(a))
    axes[0].plot(xx,a.observed_claim_rate*100,color='#357b78',lw=2,label='Observed');axes[0].plot(xx,a.normal_claim_rate*100,color='#cc8b42',lw=1.6,ls='--',label='Same-case Gaussian reference')
    axes[0].set_xticks(xx,a.label,rotation=65,ha='right');axes[0].set_ylim(0,100);axes[0].set_ylabel('Reports produced (%)');axes[0].legend(fontsize=8);axes[0].set_title('a  How much needs a new explanation?',loc='left',fontweight='bold')
    rt=pd.read_csv(src/'risk_transport.csv');coarse=rt[rt.model=='coarse'][['year','state','risk_prediction_MSE']].rename(columns={'risk_prediction_MSE':'coarse_MSE'});rt=rt.merge(coarse,on=['year','state']);rt['ratio']=rt.risk_prediction_MSE/rt.coarse_MSE
    table=rt.pivot(index=['year','state'],columns='model',values='ratio')[['RF','HGB']];v=table.to_numpy();limits=max(1.1,float(np.nanmax(v)));ims=axes[1].imshow(v,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=min(.9,float(np.nanmin(v))),vcenter=1,vmax=limits))
    axes[1].set_xticks([0,1],['RF risk','HGB risk']);axes[1].set_yticks(range(len(table)),[f'{y} {s.upper()}' for y,s in table.index],fontsize=8)
    for i in range(len(table)):
        for j in range(2):axes[1].text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=8)
    axes[1].set_title('b  Residual-risk prediction MSE / coarse',loc='left',fontweight='bold');fig.colorbar(ims,ax=axes[1],shrink=.8,label='Ratio (<1 better)')
    fig.subplots_adjust(wspace=.48,bottom=.27);save(fig,'figure3_reference_and_transport',dest)
    if phase=='final':
        p=df[(df.endpoint=='standardized')&(df.m==1000)].copy();base=p[p.method==comp][['state','MSE']].rename(columns={'MSE':'base'});p=p.merge(base,on='state');p['gain']=100*(1-p.MSE/p.base)
        gains=p.pivot(index='state',columns='method',values='gain').loc[FINAL,METHODS];coverage=p.pivot(index='state',columns='method',values='coverage95').loc[FINAL,METHODS]*100
        fig,axes=plt.subplots(2,1,figsize=(7.2,7.2))
        im=axes[0].imshow(gains.to_numpy(),cmap='RdBu',vmin=-50,vmax=50,aspect='auto');im2=axes[1].imshow(coverage.to_numpy()-95,cmap='RdBu',vmin=-10,vmax=10,aspect='auto')
        for ax in axes:ax.set_yticks(range(8),[s.upper() for s in FINAL]);ax.set_xticks(range(len(METHODS)),[LABELS[m] for m in METHODS],rotation=35,ha='right',fontsize=8)
        for i in range(8):
            for j in range(len(METHODS)):
                value=gains.iloc[i,j];axes[0].text(j,i,f'{value:+.0f}',ha='center',va='center',fontsize=7,color='white' if abs(value)>30 else 'black')
                val=coverage.iloc[i,j];axes[1].text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=7,color='white' if abs(val-95)>6 else 'black')
        axes[0].set_title(f'a  MSE reduction (%) versus frozen {LABELS[comp]}',loc='left',fontweight='bold');axes[1].set_title('b  Empirical coverage (%) of nominal 95% intervals',loc='left',fontweight='bold')
        fig.colorbar(im,ax=axes[0],shrink=.8,extend='both',label='Gain (%), color saturated at ±50');fig.colorbar(im2,ax=axes[1],shrink=.8,extend='both',label='Coverage minus 95 (pp)')
        fig.subplots_adjust(hspace=.75,bottom=.13);save(fig,'figure4_independent_validation',dest)
    dump(f'figures/{phase}/figure_contract_and_QA.json',dict(backend='python',selected_from_development=selected,comparator_from_development=comp,
      evidence='All states and all16methods retained in source data; figure2 compact subset fixed by development; no box/scatter-only encoding',
      exports=['PDF vector','SVG editable text','PNG600dpi'],dimensions_inches=['7.2x3.5+','7.2x6.3','7.2x4.2','7.2x7.2'],
      source_files={p.name:sha(p) for p in src.glob('*.csv')},units='Percentage points explicitly distinguished from percent',
      normal_reference='Uses same-case moments; not external predictive validation',visual_QA_pending=True))
    print('FIGURES_WRITTEN',phase,dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev','final']);a=p.parse_args();make(a.phase)
