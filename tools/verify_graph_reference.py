"""Verify new production numerics against BOTH original offline reference modules.
This is a method-parity check, not another policy-performance experiment.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,zipfile,datetime,importlib.util,argparse,tempfile
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study-zip',type=Path,required=True,help='Original MLEvolve_Variance_Focused_Offline_Study.zip')
parser.add_argument('--output',type=Path,default=Path('offline_reference_parity.json'))
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
workspace=tempfile.TemporaryDirectory(prefix='graph_reference_')
REF=Path(workspace.name)
with zipfile.ZipFile(args.study_zip) as archive:
 for filename in ('base_gp.py','anchor_kernel_ei.py'):
  (REF/filename).write_bytes(archive.read('variance_value_study/code/'+filename))
sys.path[:0]=[str(ROOT),str(REF)]
from base_gp import GP, design, ancestry_features, logei, OPS
from anchor_kernel_ei import predict as reference_b
from engine.parent_selectors.gp_graph_features import DecisionGraph, GP6_NAMES
from engine.parent_selectors.gp_graph_surrogate import GraphProposalSurrogate,GraphProposalSelector
from engine.parent_selectors.gp_surrogate import GP6Config
from engine.parent_selectors.base import ExpansionOption,OperatorPreview
from engine.parent_selectors.transition_store import TransitionRecord
from types import SimpleNamespace as NS
z=zipfile.ZipFile(args.study_zip)
base='variance_value_study/prior/absolute_ancestry_study/'
files=[x for x in z.namelist() if x.startswith(base+'processed/gp6') and x.endswith('.json')]
report=[]
for f in sorted(files):
 run=json.loads(z.read(f));rows=run['samples']
 eligible=[i for i,r in enumerate(rows) if sum(h['valid'] and h['available']<r['time'] for h in rows)>=5]
 if not eligible:continue
 # First and final evaluable point in every native-source run.
 indices=sorted(set([eligible[0],eligible[-1]]))
 raw=json.loads(z.read(base+'data/'+Path(f).name))['transitions']
 def timestamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
 raw_records=[TransitionRecord.from_dict({k:v for k,v in r.items() if not k.startswith('_')}) for r in sorted(raw,key=lambda r:timestamp(r['timestamp']))]
 def frozen(r):
  prefix=[h for h in raw_records if timestamp(h.timestamp)<r['time']]
  g=DecisionGraph(.3);g.update(prefix)
  result={**{'gp_'+name:float(v) for name,v in zip(GP6_NAMES,r['raw6'])},
          'parent_utility':float(r['baseline']),'incumbent_at_decision':float(r['incumbent']),
          **g.features(r['parent'],r.get('maximize',rows[0]['maximize']))}
  # Source ancestry must match the online prefix builder, not just new math.
  assert result['gp_primary_ancestry']==r['ancestry'],(run['run'],r['id'],result['gp_primary_ancestry'],r['ancestry'])
  return result
 hist=[]
 for r in rows:
  if r['valid']:
   state=frozen(r)
   hist.append((r,TransitionRecord(r['transition_id'],r['id'],r['parent'],r['child'],r['op'],tuple(r.get('references',[])),
     r['baseline'],r['outcome'],r['baseline'],r['outcome'],r['outcome']-r['baseline'],True,
     0.,1.,1.,None,None,'reference_check',state,datetime.datetime.fromtimestamp(r['available'],datetime.timezone.utc).isoformat(),
     {'proposal_closed':True})))
 for i in indices:
  query=rows[i];selected=[(h,t) for h,t in hist if h['available']<query['time']]
  old=[h for h,t in selected];records=[t for h,t in selected]
  cs=[query]+[c for c in run['pools'].get(query['id'],[]) if c['ancestry']]
  options=[]
  for c in cs:
   c=dict(c,time=query['time'],incumbent=query['incumbent'])
   options.append(ExpansionOption(c['parent'],c['op'],tuple(c.get('references',[])),frozen(c),NS(id=c['parent']),OperatorPreview(c['op'])))
  x=design(old,run,True);cx=design(cs,run,True);op=np.array([OPS.index(r['op']) for r in old]);cop=np.array([OPS.index(r['op']) for r in cs])
  ancestry=ancestry_features(old+cs,.5);n=len(old);g=ancestry[:n]@ancestry[:n].T;gc=ancestry[n:]@ancestry[:n].T
  y=(np.array([r['outcome'] for r in old])-np.array([r['baseline'] for r in old]))/run['scale']
  ra=GP(w=.3).fit(x,op,y,g);mu,sd,pr=ra.predict(cx,cop,gc)
  expected_a=(np.array([c['baseline'] for c in cs])+run['scale']*mu,run['scale']*sd,run['scale']*pr)
  rb=reference_b(x,[r['op'] for r in old],[r['ancestry'] for r in old],np.array([r['outcome'] for r in old]),
     np.array([t.selection_features['gp_graph_center'] for t in records]),cx,[r['op'] for r in cs],[r['ancestry'] for r in cs],
     np.array([o.state_features['gp_graph_center'] for o in options]),query['incumbent'],run_scale=run['scale'])
  for mode,expected in [('gp7_ancestry_logei',expected_a),('laplacian_anchor_logei',(rb.center,rb.std_allocation,rb.std_predictive_proxy))]:
   m=GraphProposalSurrogate(mode=mode,task_id=run['task'],config=GP6Config(max_optimizer_iterations=0 if mode.startswith('laplacian') else 25))
   # Compare numeric modules at the SAME offline frozen reference, not silently
   # relabel the study as native initialization replay.
   m.reference_frozen=True;m.utility_scale=run['scale'];m.absolute_loss_scale=run['absolute_loss_scale']
   m.draft_anchor=float(np.median(run['initial_seed_utilities']))
   m.fit(records);assert_allclose(m.x,x,rtol=1e-12,atol=1e-12)
   got=m.predict_quality(options)
   assert_allclose(got,expected,rtol=1e-7,atol=2e-9,err_msg=str((run['run'],i,mode)))
   if mode.startswith('gp7'):assert_allclose(m.hyperparameters,ra.hp,rtol=2e-6,atol=2e-8)
   selector=GraphProposalSelector(surrogate=m,records_provider=lambda:records);result=selector.select(NS(),options)
   assert not result.decision.fallback_used,result.decision.fallback_reason
   ref_log=logei(expected[0],expected[1],query['incumbent'])-np.log(run['scale'])
   assert result.selected_option.parent_node_id==options[int(np.argmax(ref_log))].parent_node_id
   report.append({'run':run['run'],'mode':mode,'decision_index':i,'training_count':n,'candidates':len(cs),
                  'max_prediction_abs_error':float(np.max(np.abs(np.asarray(got)-np.asarray(expected)))),
                  'same_selected_parent':True,'same_ancestry':True,'same_design':True})
print('VERIFIED',len(report),'model/decision cases;',len(set(r['run'] for r in report)),'runs; max abs error',max(r['max_prediction_abs_error'] for r in report))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
z.close();workspace.cleanup()
