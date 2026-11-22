"""Publication figures: a bounded educational-reporting argument, Python only."""
from pathlib import Path
import hashlib,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/acs_hssc199_2026-11-22';BASE=ROOT/'outputs/acs_reporting183_2026-11-21'
D=OUT/'submission/Figures';D.mkdir(parents=True,exist_ok=True);S=OUT/'source_data';S.mkdir(exist_ok=True)
contract=dict(backend='python',archetype='quantitative grid',claim='Mature allocation improves correct educational reporting in seven states while near-band SC remains mostly unresolved; all outcomes are displayed.',source='analysis/correct_report_yield_all_cases.csv',selection='All eight fixed confirmation states, all three original budgets; 2pp band; old preselected comparator and SRS, not chosen from new results',unit='2000 household sampling repetitions per state/method, not independent states or people',axis='All attempt proportions use full 0–100 percent extent; no cropped gain axis',uncertainty='Bars are empirical category proportions; no significance tests or MC intervals encoded in this descriptive panel; MC bounds in source tables',exports=['PDF vector','SVG editable','TIFF 600dpi','PNG preview'])
(OUT/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf8')
for old,new in [('figure1_population_context','Figure1'),('figure4_independent_validation','Figure3')]:
 for ext in ['pdf','svg','png']:shutil.copy2(BASE/f'figures/final/{old}.{ext}',D/f'{new}.{ext}')
for name in ['population_context.csv','point_interval_results.csv','reporting_results.csv','equal_state_summary.csv','frozen_comparator_comparison.csv','normal_and_matched_reference.csv','risk_transport.csv','observed_change_decomposition.csv','oracle_grouping_diagnostic.csv','matched_gap_and_reference_ESS.csv']:
 shutil.copy2(BASE/'analysis/final'/name,S/name)
d=pd.read_csv(OUT/'analysis/correct_report_yield_all_cases.csv');order='ct nh ia ks al sc az co'.split();methods=['SRS_HGB','Neyman_working_fine'];colors=['#537F85','#7394BC','#BB6851','#E5E9EE']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(7.2,7.0),sharey=True)
for col,(ax,m) in enumerate(zip(axes,[300,1000,3000])):
 labels=[];y=0
 for state in order:
  for method in methods:
   r=d[(d.state==state)&(d.m==m)&(d.method==method)&np.isclose(d.tau,.02)].iloc[0]
   vals=[100*(r.correct_advantage+r.correct_disadvantage)/r.R,100*r.correct_within/r.R,100*r.wrong/r.R,100*r.unresolved/r.R];assert abs(sum(vals)-100)<1e-9
   left=0
   for v,c in zip(vals,colors):ax.barh(y,v,left=left,color=c,height=.72,edgecolor='white',linewidth=.2);left+=v
   ax.text(102,y,f'{100*r.correct_yield:.1f}',va='center',fontsize=6.3)
   labels.append(state.upper()+(' random' if method=='SRS_HGB' else ' allocated'));y+=1
  y+=.55
 ax.set_title(f'{chr(97+col)}  {m:,} residential units',loc='left',fontsize=9,fontweight='bold',pad=12)
 yy=[i*2.55+j for i in range(8) for j in [0,1]];ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlim(0,118);ax.set_xticks([0,50,100]);ax.set_xlabel('All attempts (%)');ax.grid(axis='x',color='#ADB8C1',alpha=.25,linewidth=.5);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,labelsize=6.8);ax.spines['left'].set_visible(False);ax.text(102,-1.1,'Correct\n(%)',fontsize=6.3,ha='left')
fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(colors,['Correct direction','Correct within band','Wrong report','Unresolved'])],loc='upper center',ncol=4,bbox_to_anchor=(.52,1.0),frameon=False,fontsize=7)
fig.subplots_adjust(left=.14,right=.99,top=.91,bottom=.08,wspace=.15)
for ext in ['pdf','svg','png','tiff']:
 kw={'dpi':600 if ext=='tiff' else 220}
 if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
 fig.savefig(D/f'Figure2.{ext}',**kw)
plt.close(fig)
shutil.copy2(OUT/'analysis/correct_report_yield_all_cases.csv',S/'correct_report_yield_all_cases.csv')
for name in ['primary_all_state_reporting.csv','paired_correct_report_yield.csv','paired_yield_summary.csv']:shutil.copy2(OUT/'analysis'/name,S/name)
# Unchanged technical displays move to Supplementary Information.
for old,new in [('figure2_reporting_budget','SupplementaryFigureS1'),('figure3_reference_and_transport','SupplementaryFigureS2')]:
 for ext in ['pdf','png','svg']:shutil.copy2(BASE/f'figures/final/{old}.{ext}',OUT/f'{new}.{ext}')
print('FIGURES_READY',len(list(D.glob('*'))))
