"""Python-only descriptive supplementary figure; all prespecified pilot cases retained."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acs_research215_2026-11-22'
PUBLIC_INPUTS=(ROOT/'data/sipp_pilot').is_dir()
if PUBLIC_INPUTS:OUT=ROOT/'data'
D=ROOT/'reproduced_figures' if PUBLIC_INPUTS else OUT/'figures'
D.mkdir(parents=True,exist_ok=True)
methods=['SRS_HT','SRS_prior_cells','SRS_prior_BA','proportional_prior_BA','working_Neyman_prior_BA']
labels=['Direct SRS','Prior cells + SRS','Prior BA + SRS','Prior BA + proportional','Prior BA + working Neyman']


def main():
    contract=dict(backend='python',status='Post-score descriptive figure, not a new confirmatory selection',
        conclusion='Observed prior-wave information helps precision, but follow-up is incomplete and first-order interval/report quality must be checked separately',
        archetype='quantitative grid',panels=['Prior household follow-up categories','All five methods/four planned scope-budget cases coverage','All five methods near-band reports'],
        units='Baseline household clusters for sampling; 2000 conditional simulation repetitions, not independent countries or people',
        axes='Retention/report panels full0-100%; coverage cell labels actual percentages, color explicitly shows95minusobserved from-5to20pp',
        uncertainty='Empirical fractions; conditional MC95 limits in source tables, no statistical-significance encoding',
        exclusions='Planned panel2021rotation excluded before matching; all primary and reported-only pilot methods/budgets shown',
        outputs=['PDF','SVG','PNG','TIFF'],size_inches=[7.2,6.5])
    ((D if PUBLIC_INPUTS else OUT)/'sipp_figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf8')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    if PUBLIC_INPUTS:
        counts=pd.read_csv(OUT/'sipp_pilot/S3a_retention.csv').households.astype(int).tolist()
    else:
        linked=pd.read_csv(OUT/'private_sipp/linked_cohort_private.csv.gz',usecols=['cluster','observed_for_pilot'])
        per=linked.groupby('cluster').observed_for_pilot.agg(['all','any'])
        counts=[int(per['all'].sum()),int((per['any']&~per['all']).sum()),int((~per['any']).sum())]
    assert sum(counts)==13054 and counts[0]==7938
    pd.DataFrame(dict(category=['All eligible members observed','Some members observed','No valid member outcome'],households=counts,
                      fraction=np.array(counts)/sum(counts))).to_csv(D/'S3a_retention.csv',index=False)
    point=pd.read_csv(OUT/'sipp_pilot/point_interval_results.csv')
    cov=np.array([[point[(point.scope==s)&(point.m==m)&(point.method==method)].coverage95.iloc[0] for s,m in
                   [('all_released',300),('all_released',1000),('reported_only',300),('reported_only',1000)]] for method in methods])*100
    pd.DataFrame(cov,index=labels,columns=['Released300','Released1000','Reported300','Reported1000']).to_csv(D/'S3b_coverage.csv')
    report=pd.read_csv(OUT/'sipp_pilot/reporting_results.csv');report=report[(report.scope=='all_released')&(report.m==1000)&np.isclose(report.tau,.02)].set_index('method').loc[methods]
    report.to_csv(D/'S3c_reports.csv')
    fig=plt.figure(figsize=(7.2,6.5));gs=fig.add_gridspec(3,1,height_ratios=[.7,2.0,1.9],hspace=.85)
    ax=fig.add_subplot(gs[0]);colors=['#487C90','#CB9C52','#D9DFE4'];left=0
    for count,color in zip(counts,colors):
        v=100*count/sum(counts);ax.barh(0,v,left=left,color=color,height=.46,edgecolor='white');
        if v>8:ax.text(left+v/2,0,f'{count:,}\n{v:.1f}%',ha='center',va='center',fontsize=7,color='white' if color==colors[0] else '#263746')
        left+=v
    ax.set_xlim(0,100);ax.set_yticks([]);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Prior households (%)',labelpad=2)
    ax.set_title('a  Historical frame does not establish complete follow-up',loc='left',fontweight='bold',fontsize=9,pad=19)
    ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['All adult outcomes','Partial adult outcomes','No valid adult outcome'])],loc='lower left',bbox_to_anchor=(-.01,1.02),ncol=3,frameon=False,fontsize=7)
    ax=fig.add_subplot(gs[1]);im=ax.imshow(95-cov,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=20),aspect='auto')
    ax.set_yticks(range(5),labels,fontsize=7);ax.set_xticks(range(4),['Released\n300','Released\n1,000','Reported only\n300','Reported only\n1,000'],fontsize=7)
    ax.set_title('b  Observed coverage of nominal 95% intervals',loc='left',fontweight='bold',fontsize=9,pad=12)
    for i in range(5):
        for j in range(4):ax.text(j,i,f'{cov[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='white' if 95-cov[i,j]>12 else '#172A38')
    cbar=fig.colorbar(im,ax=ax,pad=.025,fraction=.027);cbar.ax.tick_params(labelsize=6);cbar.set_label('95 − observed coverage (pp)',fontsize=7)
    ax=fig.add_subplot(gs[2]);left=np.zeros(5);rcolors=['#547F89','#B9745E','#E0E5E9']
    for col,color in zip(['correct','wrong','unresolved'],rcolors):
        vals=100*report[col].to_numpy()/2000;ax.barh(np.arange(5),vals,left=left,color=color,height=.65,edgecolor='white',linewidth=.3);left+=vals
    assert np.allclose(left,100)
    ax.set_yticks(range(5),labels,fontsize=7);ax.invert_yaxis();ax.set_xlim(0,120);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('All attempts (%)')
    ax.set_title('c  Reports near the fixed 2 pp boundary (1,000 clusters)',loc='left',fontweight='bold',fontsize=9,pad=22)
    for i,(_,r) in enumerate(report.iterrows()):ax.text(102,i,f'{100*r.correct/2000:.2f} / {100*r.wrong/2000:.2f}',va='center',fontsize=6.5)
    ax.text(102,-.85,'Correct / wrong (%)',fontsize=6.3)
    ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(rcolors,['Correct','Wrong','Unresolved'])],loc='lower left',bbox_to_anchor=(-.01,1.02),ncol=3,frameon=False,fontsize=7)
    ax.grid(axis='x',color='#BAC6CD',alpha=.25,linewidth=.5);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.29,right=.91,top=.93,bottom=.08)
    fig.text(.02,.017,'SIPP 2024→2025 retained cohort; prior weights/reference. Conditional sampling only; field roster and degree truth unverified.',fontsize=6.7)
    for ext in ['pdf','svg','png','tiff']:
        kwargs={'dpi':220 if ext=='png' else 600}
        if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(D/f'SupplementaryFigureS3.{ext}',**kwargs)
    plt.close(fig)
    print('SIPP_FIGURE_EXPORTED',counts)


if __name__=='__main__':main()
