"""Production preparation/CPU execution/parser/GP integration.

Only model responses are synthetic: setup uses a real local HTTP transport;
candidate responses use the normal code parser. No external LLM/GPU is used.
"""
import copy
import json
import math
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from config import _load_cfg, prep_agent_workspace
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.fixed_evaluation import FixedEvaluation, EvaluationError, InvalidPrediction, compute_metric, score_files, admit_prepared, _json, _write_json
from engine.search_node import Journal
from engine.parent_selectors.proposal_feedback import effective_proposals
from utils import serialize
from utils.evaluation_helpers import binary_probabilities, fit_binary_temperature


ADAPTER = '''import json
import pandas as pd

def prepare(data_dir, output_dir, seed):
    public = output_dir / 'public'
    public.mkdir()
    raw = pd.read_csv(data_dir / 'train.csv')
    raw['source_id'] = 'train:' + raw['id'].astype(str)
    raw['sample_id'] = 'v' + raw['id'].astype(str)
    raw['split'] = ['V' if i % 5 == 0 else 'T' for i in range(len(raw))]
    test = pd.read_csv(data_dir / 'test.csv')
    test['sample_id'] = test.id
    members = raw[['sample_id','source_id','split']].copy()
    tm = pd.DataFrame({'sample_id':test.sample_id, 'source_id':'test:' + test.id, 'split':'test'})
    pd.concat([members,tm]).to_csv(output_dir / 'members.csv',index=False)
    raw.loc[raw.split == 'T',['sample_id','x','label']].to_csv(public / 'train.csv',index=False)
    valid = raw.loc[raw.split == 'V'].copy()
    valid[['sample_id','x']].to_csv(public / 'validation.csv',index=False)
    test[['sample_id','id','x']].to_csv(public / 'test.csv',index=False)
    pd.read_csv(data_dir / 'sample_submission.csv').to_csv(public / 'sample_submission.csv',index=False)
    pd.DataFrame({'id':valid.sample_id,'label':valid.label,'__sample_id':valid.sample_id}).to_csv(output_dir / 'truth.csv',index=False)
    spec = dict(metric={'name':'log_loss','maximize':False,'labels':[0,1],'clip_eps':1e-15},
       id_columns=['id'],prediction_columns=['label'],truth_columns=['label'],prediction_kind='numeric',
       source_target_columns=['label'],value_range=[0,1],
       split={'strategy':'stratified','unit':'sample','seed':seed},
       official_sample_submission={'path':'sample_submission.csv'},
       metric_source='Evaluation: binary log loss',candidate_instructions='Numeric x; V prediction id is sample_id, test id is id; probability is for class 1.')
    (output_dir / 'protocol.json').write_text(json.dumps(spec))
'''

CANDIDATE = '''import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluation_helpers import binary_probabilities, fit_binary_temperature
t = pd.read_csv('input/train.csv')
v = pd.read_csv('input/validation.csv')
test = pd.read_csv('input/test.csv')
model = LogisticRegression(C=0.1).fit(t[['x']].iloc[::2], t.label.iloc[::2])
cal = t.iloc[1::2]
temperature = fit_binary_temperature(model.decision_function(cal[['x']]), cal.label, [1,2,3], 1e-6)
v_logits = model.decision_function(v[['x']])
test_logits = model.decision_function(test[['x']])
out = Path(os.environ['MLEV_EVALUATION_OUTPUT_DIR'])
np.savez(out / 'calibration_cache.npz', protocol_id=os.environ['MLEV_EVALUATION_PROTOCOL_ID'], node_id=os.environ['MLEV_NODE_ID'],
         calibration_logits=model.decision_function(cal[['x']]), calibration_labels=cal.label,
         validation_logits=v_logits, test_logits=test_logits, validation_ids=v.sample_id.to_numpy(dtype=str), test_ids=test.id.to_numpy(dtype=str))
pd.DataFrame({'id':v.sample_id,'label':binary_probabilities(v_logits,temperature,1e-6)}).to_csv(out / 'validation.csv',index=False)
Path('submission').mkdir(exist_ok=True)
pd.DataFrame({'id':test.id,'label':binary_probabilities(test_logits,temperature,1e-6)}).to_csv('submission/submission.csv',index=False)
print('Final Validation Score: 999.0')
'''


@pytest.fixture
def setup_server():
    state = SimpleNamespace(responses=[], requests=[], delay=0, delays=[])
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state.requests.append(request)
            import time
            time.sleep(state.delays.pop(0) if state.delays else state.delay)
            response = state.responses.pop(0) if state.responses else {'supported':True,'reasoning':'Independent examples; fixed binary log loss; T-only calibration.', 'code':ADAPTER}
            body = json.dumps({'id':'local-completion','object':'chat.completion','created':1,'model':'local-test',
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'tool_calls':[{'id':'setup','type':'function','function':{'name':'submit_evaluation_protocol','arguments':json.dumps(response)}}]}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
            try:self.wfile.write(body)
            except BrokenPipeError:pass
        def log_message(self, *args):pass
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    state.url = f'http://127.0.0.1:{server.server_port}/v1'
    yield state
    server.shutdown(); server.server_close(); thread.join()


@pytest.fixture
def fixed_cfg(tmp_path, setup_server):
    cfg = OmegaConf.merge(_load_cfg(use_cli_args=False), OmegaConf.load(Path(__file__).parents[1] / 'config/experiments/gp6.yaml'))
    cfg.data_dir = tmp_path / 'source'; cfg.data_dir.mkdir()
    x = np.linspace(-3,3,100)
    pd.DataFrame({'id':range(100),'x':x,'label':(np.sin(np.arange(100)) + x > 0).astype(int)}).to_csv(cfg.data_dir / 'train.csv',index=False)
    pd.DataFrame({'id':['t0','t1','t2'],'x':[-2,0,2]}).to_csv(cfg.data_dir / 'test.csv',index=False)
    pd.DataFrame({'id':['t0','t1','t2'],'label':[0,0,0]}).to_csv(cfg.data_dir / 'sample_submission.csv',index=False)
    cfg.log_dir = tmp_path / 'run/logs'; cfg.workspace_dir = tmp_path / 'run/workspace'
    cfg.exp_id = 'cpu-fixture'; cfg.exp_name = 'test_fixed_cpu'
    cfg.agent.code.model = 'local-test'; cfg.agent.code.base_url = setup_server.url; cfg.agent.code.api_key = 'local-test-only'
    cfg.agent.use_global_memory = False; cfg.coldstart.use_coldstart = False
    cfg.agent.use_stepwise_generation = False; cfg.agent.use_diff_mode = False; cfg.agent.use_aggregation = False
    cfg.agent.initial_drafts = 1; cfg.agent.search.num_drafts = 1
    cfg.agent.time_limit = 300; cfg.exec.timeout = 30
    cfg.evaluation.enabled = True; cfg.evaluation.setup_timeout_seconds = 60
    cfg.evaluation.protocol_dir = str(tmp_path / 'protocol')
    prep_agent_workspace(cfg)
    return cfg


@pytest.fixture
def model_responses(monkeypatch):
    state = SimpleNamespace(code=CANDIDATE,prompts=[])
    def generate(**kwargs):
        state.prompts.append(kwargs['prompt'])
        return 'Fit logistic regression and use T-only calibration.\n```python\n' + state.code + '\n```'
    def query(**kwargs):
        state.prompts.append(kwargs['system_message'])
        if kwargs['func_spec'].name == 'submit_code_review':
            return {'needs_revision':False,'reasoning':'The candidate follows the fixed protocol.','revised_code':None}
        return {'is_bug':False,'summary':'Completed; stdout score is diagnostic.', 'code_summary':'Logistic regression', 'metric':999}
    monkeypatch.setattr('agents.coder.base_coder.generate',generate)
    monkeypatch.setattr('agents.code_review_agent.query',query)
    monkeypatch.setattr('agents.result_parse_agent.query',query)
    return state


def test_cpu_candidate_generation_execution_scoring_gp_and_resume(fixed_cfg, setup_server, model_responses):
    agent = AgentSearch('Evaluation: binary log loss on positive-class probability.',fixed_cfg,Journal())
    interpreter = Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    assert len(agent.journal.nodes) == 1
    assert agent.search_start_time == agent.start_time
    first = agent.step(None,interpreter.run,execute_immediately=False)
    assert first.stage == 'draft' and first.pending_execution
    first = agent.execute_deferred_node(first,interpreter.run)
    assert first.is_valid and not first.is_buggy, first.term_out + str(first.analysis)
    assert 0 < first.metric.value < 1 and first.metric.maximize is False
    assert first.metric.value == first.evaluation_result['metric']
    model_responses.code = CANDIDATE.replace('C=0.1','C=1.0')
    child = agent.step(first,interpreter.run)
    assert child.stage == 'improve' and child.is_valid, child.analysis
    runtime = agent.phase1_runtime
    runtime.surrogate.fit(runtime.transition_store.records())
    assert runtime.surrogate.x.shape[1] == 6
    assert len(runtime.transition_store.records()) == 2
    assert all(r.valid for r in runtime.transition_store.records())
    assert first.evaluation_output_dir != child.evaluation_output_dir
    assert 'Final Validation Score: 999.0' in first.term_out
    assert 'Create my own train/val' not in agent.data_preview
    assert 'label' not in pd.read_csv(fixed_cfg.workspace_dir / 'input/validation.csv')
    assert (Path(first.evaluation_output_dir) / 'calibration_cache.npz').is_file()
    agent.capture_resume_state()
    restored = serialize.loads_json(serialize.dumps_json(agent.journal),Journal)
    resumed = AgentSearch('ignored',copy.deepcopy(fixed_cfg),restored)
    assert resumed.fixed_evaluation.protocol_id == agent.fixed_evaluation.protocol_id
    assert len(setup_server.requests) == 1
    assert resumed.best_node.metric.value == agent.best_node.metric.value
    review_prompts = [prompt for prompt in model_responses.prompts
                      if isinstance(prompt, dict) and 'Authoritative program result' in prompt]
    assert review_prompts
    assert all('submission_file' not in prompt['Authoritative program result']
               for prompt in review_prompts)
    assert first.evaluation_result['submission_file']
    # Code-review legitimately sees the adversarial stdout print in candidate code.
    assert 'Final Validation Score' not in json.dumps(model_responses.prompts[0])


def test_temperature_scalar_grid_and_extremes():
    logits=np.array([-2,-1,1,2,4]); labels=np.array([0,0,1,0,1]); grid=[1,2,3]
    temperature=fit_binary_temperature(logits,labels,grid)
    losses=[-np.mean(labels*np.log(binary_probabilities(logits,t))+(1-labels)*np.log1p(-binary_probabilities(logits,t))) for t in grid]
    assert temperature == grid[np.argmin(losses)]
    assert isinstance(temperature,float)
    np.testing.assert_allclose(binary_probabilities([-1000,1000]),[1e-6,1-1e-6])
    with pytest.raises(ValueError):binary_probabilities([-1000,1000],clip_eps=1e-30)
    with pytest.raises(ValueError):fit_binary_temperature([1],[0],[0])
    with pytest.raises(ValueError):binary_probabilities([[1,2]])
    assert math.isfinite(fit_binary_temperature([-1000,1000],[1,0],[1,2,3]))


@pytest.fixture
def raw_prepared(fixed_cfg, tmp_path):
    root=tmp_path / 'adapter-output'; root.mkdir()
    namespace={}; exec(ADAPTER,namespace)
    namespace['prepare'](fixed_cfg.data_dir,root,fixed_cfg.agent.seed)
    (root / 'evaluate.py').write_text(ADAPTER)
    return root


def test_independent_metric_definitions_and_label_order():
    y=np.array([[0,0],[1,0],[0,1],[1,1]])
    p=np.array([[.1,.4],[.4,.2],[.35,.8],[.8,.6]])
    assert compute_metric(y,p,{'name':'roc_auc','average':'macro'}) == 1
    assert compute_metric(y,p,{'name':'roc_auc','average':'micro'}) == .96875
    assert compute_metric([[0,0],[0,0]],[[0,0],[3,4]],{'name':'rmse'}) == 2.5
    assert compute_metric(np.zeros((2,2)),np.expm1([[1,0],[3,4]]),{'name':'rmsle'}) == pytest.approx((math.sqrt(5)+math.sqrt(8))/2)
    loss={'name':'log_loss','labels':['z','a'],'clip_eps':1e-15}
    assert compute_metric([['z'],['a']],[[.9,.1],[.1,.9]],loss) == pytest.approx(-math.log(.9))
    assert compute_metric([['z'],['a']],[[.1,.9],[.9,.1]],loss) == pytest.approx(-math.log(.1))
    for labels in ([0,1],['0','1']):
        assert compute_metric([['0'],['1']],[[.1],[.9]],dict(loss,labels=labels)) == pytest.approx(-math.log(.9))
    assert compute_metric([['2'],['1']],[[.9,.1],[.1,.9]],dict(loss,labels=[2,1])) == pytest.approx(-math.log(.9))
    assert compute_metric([['NA'],[''],['Москва'],['a b']],[['NA'],[''],['москва'],['a b']],{'name':'accuracy'}) == .75
    assert compute_metric([[0],[1],[2]],[[0],[2],[2]],{'name':'quadratic_kappa','labels':[0,1,2]}) == pytest.approx(.8)
    with pytest.raises(InvalidPrediction):
        compute_metric([[0],[1],[2]],[[.4],[1.4],[1.8]],{'name':'quadratic_kappa','labels':[0,1,2]})


@pytest.mark.parametrize('case', ['reordered','missing','duplicate','extra','nan','wrong_target','out_of_range','large_loss','missing_file'])
def test_score_files_rejects_structure_and_preserves_real_bad_scores(raw_prepared,fixed_cfg,case):
    root=raw_prepared; spec=admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)
    _write_json(root / 'protocol.json',spec)
    truth=pd.read_csv(root / 'truth.csv'); pred=truth[['id','label']].copy()
    pred['label']=np.where(truth.label == 1,.9,.1)
    if case=='reordered':pred=pred.iloc[::-1]
    elif case=='missing':pred=pred.iloc[:-1]
    elif case=='duplicate':pred=pd.concat([pred,pred.iloc[:1]])
    elif case=='extra':pred=pd.concat([pred,pd.DataFrame({'id':['extra'],'label':[.5]})])
    elif case=='nan':pred.loc[0,'label']=np.nan
    elif case=='wrong_target':pred=pred.rename(columns={'label':'other'})
    elif case=='out_of_range':pred.loc[0,'label']=1.2
    elif case=='large_loss':pred['label']=np.where(truth.label == 1,.001,.999)
    path=root / 'pred.csv'; pred.to_csv(path,index=False)
    if case=='missing_file':path.unlink()
    result=score_files(root,path,root / 'public/sample_submission.csv')
    if case in ('reordered','large_loss'):
        assert result['valid']
        assert result['metric'] == pytest.approx(-math.log(.001 if case=='large_loss' else .9))
    else:assert not result['valid'] and result['metric'] is None


@pytest.mark.parametrize('case', ['group_overlap','source_overlap','target_column','missing_test','missing_target','duplicate_v','bad_direction'])
def test_prepare_rejects_leakage_and_official_template_mismatch(raw_prepared,fixed_cfg,case):
    root=raw_prepared; spec=_json(root / 'protocol.json')
    members=pd.read_csv(root / 'members.csv',keep_default_na=False)
    if case=='group_overlap':
        spec['split'].update(strategy='group',unit='patient'); members['group']='same_patient'; members.to_csv(root / 'members.csv',index=False)
    elif case=='source_overlap':
        members.loc[1,'source_id']=members.loc[0,'source_id']; members.to_csv(root / 'members.csv',index=False)
    elif case=='target_column':
        path=root / 'public/validation.csv'; frame=pd.read_csv(path); frame['label']=1; frame.to_csv(path,index=False)
    elif case in ('missing_test','missing_target'):
        path=root / 'public/sample_submission.csv'; frame=pd.read_csv(path)
        (frame.iloc[:-1] if case=='missing_test' else frame.drop(columns='label')).to_csv(path,index=False)
    elif case=='duplicate_v':
        path=root / 'truth.csv'; frame=pd.read_csv(path); pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
    else:spec['metric']['maximize']=True
    _write_json(root / 'protocol.json',spec)
    with pytest.raises((EvaluationError,InvalidPrediction)):admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)


@pytest.mark.parametrize('case',['good','overlap','v_target'])
def test_native_assets_opaque_v_and_target_separation(raw_prepared,fixed_cfg,case):
    root=raw_prepared
    (fixed_cfg.data_dir / 'cat.0.jpg').write_bytes(b'feature-v')
    (fixed_cfg.data_dir / 'dog.1.jpg').write_bytes(b'feature-t')
    (fixed_cfg.data_dir / 'clean.png').write_bytes(b'supervision-t')
    assets=[{'source':'cat.0.jpg','public_path':'images/v000.jpg','sample_id':'v0','role':'feature'},
            {'source':'cat.0.jpg' if case=='overlap' else 'dog.1.jpg','public_path':'images/t001.jpg','sample_id':'v1','role':'feature'},
            {'source':'clean.png','public_path':'targets/t001.png','sample_id':'v0' if case=='v_target' else 'v1','role':'target'}]
    pd.DataFrame(assets).to_csv(root / 'assets.csv',index=False)
    if case!='good':
        with pytest.raises(EvaluationError):admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)
    else:
        admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)
        assert not (root / 'public/images/v000.jpg').is_symlink()
        assert (root / 'public/images/v000.jpg').read_bytes()==b'feature-v'
        assert not any('cat.' in str(p.relative_to(root / 'public')) for p in (root/'public').rglob('*'))
        assert (root / 'public/targets/t001.png').read_bytes()==b'supervision-t'


def test_failed_prediction_debug_is_first_valid_proposal(fixed_cfg,model_responses):
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    model_responses.code=CANDIDATE+"\n(out / 'validation.csv').unlink()\n"
    first=agent.step(None,interpreter.run,execute_immediately=False)
    agent.execute_deferred_node(first,interpreter.run)
    assert first.is_buggy and not first.evaluation_result['valid']
    assert not agent.phase1_runtime.transition_store.records()[0].valid
    model_responses.code=f'''import os
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation_helpers import binary_probabilities, fit_binary_temperature
cache=np.load({str(Path(first.evaluation_output_dir)/'calibration_cache.npz')!r},allow_pickle=False)
assert str(cache['protocol_id'])==os.environ['MLEV_EVALUATION_PROTOCOL_ID']
assert str(cache['node_id'])=={first.id!r}
temperature=fit_binary_temperature(cache['calibration_logits'],cache['calibration_labels'],[1,2,3],1e-6)
out=Path(os.environ['MLEV_EVALUATION_OUTPUT_DIR'])
pd.DataFrame({{'id':cache['validation_ids'],'label':binary_probabilities(cache['validation_logits'],temperature,1e-6)}}).to_csv(out/'validation.csv',index=False)
pd.DataFrame({{'id':cache['test_ids'],'label':binary_probabilities(cache['test_logits'],temperature,1e-6)}}).to_csv('submission/submission.csv',index=False)
'''
    repaired=agent.step(first,interpreter.run)
    assert repaired.stage=='debug' and repaired.is_valid, repaired.analysis
    rows=agent.phase1_runtime.transition_store.records(); proposals=effective_proposals(rows)
    assert len(proposals)==1 and proposals[0].valid and proposals[0].operator=='draft'
    assert proposals[0].total_wall_time_sec==pytest.approx(sum(r.total_wall_time_sec for r in rows))
    assert first.evaluation_output_dir != repaired.evaluation_output_dir
    assert 'LogisticRegression' not in repaired.code


def test_three_parallel_candidates_and_stale_attempt_rejection(fixed_cfg,model_responses):
    from concurrent.futures import ThreadPoolExecutor
    fixed_cfg.agent.initial_drafts=3; fixed_cfg.agent.search.num_drafts=3
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    nodes=[agent.step(None,interpreter.run,execute_immediately=False) for _ in range(3)]
    with ThreadPoolExecutor(3) as pool:
        results=list(pool.map(lambda node:agent.execute_deferred_node(node,interpreter.run),nodes))
    assert all(n.is_valid for n in results)
    assert len({n.evaluation_output_dir for n in results})==3
    assert len(agent.phase1_runtime.transition_store.records())==3
    node=nodes[0]; old=node.evaluation_output_dir
    agent.fixed_evaluation.begin_attempt(node,10)
    assert node.evaluation_output_dir != old
    result=agent.fixed_evaluation.score(node)
    assert not result['valid']


def test_shared_evaluator_fault_stops_before_transition(fixed_cfg,model_responses):
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    node=agent.step(None,interpreter.run,execute_immediately=False)
    (Path(fixed_cfg.evaluation.protocol_dir)/'truth.csv').unlink()
    with pytest.raises(EvaluationError):agent.execute_deferred_node(node,interpreter.run)
    assert not agent.phase1_runtime.transition_store.records()
    assert len(agent.journal.nodes)==1
    assert (Path(node.evaluation_output_dir)/'validation.csv').exists()
    assert agent.journal.evaluation_state['failure']
    with pytest.raises(EvaluationError):agent.step(None,interpreter.run)


@pytest.mark.parametrize('failure_kind', ['runtime', 'syntax'])
def test_setup_structured_repair_and_compiled_prompt(fixed_cfg,setup_server,failure_kind):
    broken = [f'def prepare(data_dir, output_dir, seed):\n    raise ValueError("distinct setup failure {i}")'
              if failure_kind == 'runtime' else f'def prepare(data_dir, output_dir, seed):\n    broken syntax ! # distinct setup failure {i}'
              for i in (1, 2)]
    setup_server.responses=[{'supported':True,'reasoning':'Fix observed program error','code':code}
                            for code in [*broken, '```python\n'+ADAPTER+'\n```']]
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal())
    assert len(setup_server.requests)==3 and len(agent.journal.nodes)==1
    assert 'You are the DEBUG agent' not in setup_server.requests[0]['messages'][0]['content']
    for i, request in enumerate(setup_server.requests[1:], 1):
        prompt=request['messages'][0]['content']
        assert '# Introduction' in prompt and '# Instructions' in prompt
        assert 'You are the DEBUG agent' in prompt and 'Previous preparation' in prompt
        assert f'distinct setup failure {i}' in prompt
        assert broken[i-1] in prompt
        assert ('ValueError' if failure_kind == 'runtime' else 'SyntaxError') in prompt
        assert 'T-only' in prompt or 'T only' in prompt
        assert 'source_id' in prompt and 'official_sample_submission' in prompt
    assert _json(fixed_cfg.log_dir/'evaluation_setup/success.json')['attempts']==3
    assert agent.fixed_evaluation.spec['metric']['name']=='log_loss'
    assert not agent.phase1_runtime.transition_store.records()
    assert fixed_cfg.agent.code.model=='local-test'


@pytest.mark.parametrize('during_debug', [False, True])
def test_setup_api_total_deadline_and_no_public_fallback(fixed_cfg,setup_server,during_debug):
    import time
    # Allow the real worker imports on shared filesystems before blocking HTTP.
    fixed_cfg.evaluation.setup_timeout_seconds=20
    setup_server.delay=30
    if during_debug:
        setup_server.delays=[0,30]
        setup_server.responses=[{'supported':True,'reasoning':'Syntax needs repair','code':'broken syntax !'}]
    started=time.monotonic()
    with pytest.raises(EvaluationError):AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal())
    assert time.monotonic()-started < fixed_cfg.evaluation.setup_timeout_seconds + 1.5
    assert setup_server.requests, 'Deadline must interrupt a live API call, not only module imports'
    if during_debug:
        assert len(setup_server.requests)==2
        assert 'You are the DEBUG agent' in setup_server.requests[1]['messages'][0]['content']
    assert not (Path(fixed_cfg.evaluation.protocol_dir)/'protocol.json').exists()
    assert not list((fixed_cfg.workspace_dir/'input').iterdir())
    assert (fixed_cfg.log_dir/'evaluation_setup/failure.json').exists()


def test_setup_exhaustion_is_bounded(fixed_cfg,setup_server):
    fixed_cfg.evaluation.setup_max_attempts=2
    response={'supported':True,'reasoning':'Parse failure','code':'broken syntax !'}
    setup_server.responses=[response,response]
    with pytest.raises(EvaluationError):AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal())
    assert len(setup_server.requests)==2
    assert not (Path(fixed_cfg.evaluation.protocol_dir)/'protocol.json').exists()


@pytest.mark.parametrize('kind',['text','multi_auc','class_auc','multi_rmsle','pixels'])
def test_full_native_target_format_scoring(raw_prepared,fixed_cfg,kind):
    root=raw_prepared; spec=_json(root/'protocol.json')
    v=pd.read_csv(root/'public/validation.csv'); test=pd.read_csv(root/'public/test.csv')
    spec.pop('value_range',None)
    spec['metric']={'name':'accuracy' if kind=='text' else ('roc_auc' if kind in ('multi_auc','class_auc') else ('rmsle' if kind=='multi_rmsle' else 'rmse')),
                    'maximize':kind in ('text','multi_auc','class_auc')}
    columns=['a','b','c','d'] if kind=='class_auc' else (['after'] if kind=='text' else (['a','b'] if kind.startswith('multi') else ['value']))
    spec.update(prediction_columns=columns,truth_columns=columns,prediction_kind='text' if kind=='text' else 'numeric')
    if kind in ('multi_auc','class_auc'):spec['metric']['average']='macro'
    truth=pd.DataFrame({'id':v.sample_id,'__sample_id':v.sample_id})
    if kind=='text':
        truth['after']=(['NA','','Москва','two words']*5)
        spec['source_target_columns']=['after']
        members=pd.read_csv(root/'members.csv'); members['group']=members.split
        members.to_csv(root/'members.csv',index=False); spec['split'].update(strategy='group',unit='sentence')
        for name in ('train','validation','test'):
            path=root/f'public/{name}.csv'; frame=pd.read_csv(path); frame['before']='source token'; frame.to_csv(path,index=False)
    elif kind=='multi_auc':
        truth['a']=np.arange(len(v))%2; truth['b']=(np.arange(len(v))//2)%2
    elif kind=='class_auc':
        spec['probability_sum']=True
        for i,col in enumerate(columns):truth[col]=(np.arange(len(v))%4==i).astype(int)
    elif kind=='multi_rmsle':
        truth['a']=np.arange(len(v)); truth['b']=np.arange(len(v))*2
    else:
        rows=[]
        for i,sample in enumerate(v.sample_id):
            for pixel in range(2 if i==0 else 1):rows.append({'id':f'{sample}_1_{pixel+1}','__sample_id':sample,'value':0.0})
        truth=pd.DataFrame(rows)
    truth.to_csv(root/'truth.csv',index=False)
    template=pd.DataFrame({'id':test.id})
    for col in columns:template[col]='' if kind=='text' else (.25 if kind=='class_auc' else 0)
    template.to_csv(root/'public/sample_submission.csv',index=False)
    template.to_csv(fixed_cfg.data_dir/'sample_submission.csv',index=False)
    _write_json(root/'protocol.json',spec)
    spec=admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed); _write_json(root/'protocol.json',spec)
    prediction=truth.drop(columns='__sample_id')
    expected=1 if kind in ('text','multi_auc','class_auc') else 0
    if kind=='pixels':
        prediction.loc[0,'value']=1; expected=math.sqrt(1/len(prediction))
    prediction.iloc[::-1].to_csv(root/'prediction.csv',index=False)
    result=score_files(root,root/'prediction.csv',root/'public/sample_submission.csv')
    assert result['valid'] and result['metric']==pytest.approx(expected)
    assert result['n_scored']==len(truth)
    prediction.iloc[:-1].to_csv(root/'prediction.csv',index=False)
    assert not score_files(root,root/'prediction.csv',root/'public/sample_submission.csv')['valid']


def test_protocol_target_shape_mismatch_fails_during_setup(raw_prepared,fixed_cfg):
    root=raw_prepared; spec=_json(root/'protocol.json')
    spec['metric']={'name':'roc_auc','average':'macro','maximize':True}; spec['prediction_columns']=['a','b']
    _write_json(root/'protocol.json',spec)
    with pytest.raises(EvaluationError,match='dimensions'):admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)


@pytest.mark.parametrize('duplicate_ids',[False,True])
def test_zip_official_template_and_exact_empty_metadata(raw_prepared,fixed_cfg,duplicate_ids):
    import zipfile
    root=raw_prepared; spec=_json(root/'protocol.json')
    spec['metadata_columns']=['Date']
    spec['official_sample_submission']={'path':'templates.zip','archive_member':'official/sample.csv'}
    truth=pd.read_csv(root/'truth.csv',dtype=str,keep_default_na=False)
    truth['Date']=['','NA']*(len(truth)//2)
    if duplicate_ids:
        truth['id']=[f'v{i//2}' for i in range(len(truth))]
    truth.to_csv(root/'truth.csv',index=False)
    template=pd.read_csv(root/'public/sample_submission.csv',dtype=str,keep_default_na=False)
    template['Date']=['','NA','2012-06-01']
    if duplicate_ids:
        template.loc[1,'id']=template.loc[0,'id']
    template.to_csv(root/'public/sample_submission.csv',index=False)
    with zipfile.ZipFile(fixed_cfg.data_dir/'templates.zip','w') as archive:
        archive.writestr('official/sample.csv',template.to_csv(index=False))
    _write_json(root/'protocol.json',spec)
    admitted=admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)
    _write_json(root/'protocol.json',admitted)
    pred=truth[['id','label','Date']].copy(); pred['label']=np.where(truth.label=='1',.9,.1)
    path=root/'validation.csv'; pred.iloc[::-1].to_csv(path,index=False)
    assert score_files(root,path,root/'public/sample_submission.csv')['valid']
    pred.loc[0,'Date']='changed'; pred.to_csv(path,index=False)
    rejected=score_files(root,path,root/'public/sample_submission.csv')
    assert not rejected['valid'] and 'metadata' in rejected['error']
    template.loc[0,'Date']='changed'; template.to_csv(root/'public/sample_submission.csv',index=False)
    with pytest.raises(EvaluationError,match='official submission metadata'):
        admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)
    template.loc[0,'Date']=''; template.iloc[:-1].to_csv(root/'public/sample_submission.csv',index=False)
    with pytest.raises(EvaluationError,match='official test IDs'):
        admit_prepared(root,fixed_cfg.data_dir,fixed_cfg.exp_id,fixed_cfg.agent.seed)


@pytest.mark.parametrize('extreme',[0,1])
def test_legal_auc_extremes_reach_valid_transition(fixed_cfg,setup_server,model_responses,extreme):
    data=pd.read_csv(fixed_cfg.data_dir/'train.csv'); data.label=(data.x>0).astype(int); data.to_csv(fixed_cfg.data_dir/'train.csv',index=False)
    setup_server.responses=[{'supported':True,'reasoning':'Binary AUC','code':ADAPTER.replace("'name':'log_loss','maximize':False,'labels':[0,1],'clip_eps':1e-15", "'name':'roc_auc','maximize':True,'average':'micro'")}]
    if extreme==0:model_responses.code=CANDIDATE.replace("'label':binary_probabilities(","'label':1-binary_probabilities(")
    agent=AgentSearch('Evaluation: AUC.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    node=agent.step(None,interpreter.run,execute_immediately=False); agent.execute_deferred_node(node,interpreter.run)
    assert node.is_valid and not node.is_buggy and node.metric.value==extreme
    assert agent.phase1_runtime.transition_store.records()[0].valid


def test_real_large_loss_reaches_gp(fixed_cfg,model_responses):
    model_responses.code=CANDIDATE.replace('binary_probabilities(v_logits,temperature,1e-6)', "np.where(model.predict(v[['x']])==1,.001,.999)")
    model_responses.code=model_responses.code.replace('binary_probabilities(test_logits,temperature,1e-6)', "np.where(model.predict(test[['x']])==1,.001,.999)")
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    node=agent.step(None,interpreter.run,execute_immediately=False); agent.execute_deferred_node(node,interpreter.run)
    assert node.is_valid and node.metric.value>3
    assert agent.phase1_runtime.transition_store.records()[0].valid


def test_protocol_reuse_changed_data_legacy_and_mode_switch(fixed_cfg,setup_server,model_responses):
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); agent.capture_resume_state()
    journal=serialize.loads_json(serialize.dumps_json(agent.journal),Journal)
    legacy=copy.deepcopy(journal); legacy.evaluation_state=None
    with pytest.raises(EvaluationError,match='legacy'):AgentSearch('ignored',fixed_cfg,legacy)
    off=copy.deepcopy(fixed_cfg); off.evaluation.enabled=False
    with pytest.raises(EvaluationError,match='legacy mode'):AgentSearch('ignored',off,journal)
    second=copy.deepcopy(fixed_cfg); second.workspace_dir=Path(fixed_cfg.workspace_dir).parent/'paired-workspace'
    second.log_dir=Path(fixed_cfg.log_dir).parent/'paired-logs'
    paired=AgentSearch('same task',second,Journal())
    assert len(setup_server.requests)==1 and paired.fixed_evaluation.protocol_id==agent.fixed_evaluation.protocol_id
    assert paired.fixed_evaluation.budget_charge >= agent.fixed_evaluation.spec['setup_cost_seconds']
    with (fixed_cfg.data_dir/'train.csv').open('a') as out:out.write('\n')
    with pytest.raises(EvaluationError):AgentSearch('ignored',fixed_cfg,journal)
    assert len(setup_server.requests)==1


def test_final_selection_uses_scored_snapshot_not_unscored_fusion(fixed_cfg,model_responses):
    from engine.fixed_evaluation import finalize_submission
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    node=agent.step(None,interpreter.run,execute_immediately=False); agent.execute_deferred_node(node,interpreter.run)
    source=Path(node.evaluation_result['submission_file']); expected=source.read_bytes()
    (fixed_cfg.workspace_dir/'submission'/f'submission_{node.id}.csv').write_text('unscored overwritten data')
    selected=finalize_submission(fixed_cfg,agent.journal)
    assert selected.id==node.id
    assert (fixed_cfg.workspace_dir/'best_submission/submission.csv').read_bytes()==expected
    assert (fixed_cfg.workspace_dir/'ensembles_csv/top1ens-total_run_time0.0h.csv').read_bytes()==expected
    result=_json(fixed_cfg.log_dir/'final_submission.json')
    assert result['selection']=='best_fixed_evaluated_node' and result['node_id']==node.id


@pytest.mark.parametrize('fault',[False,True])
def test_full_runner_real_cpu_and_fatal_evaluator_propagation(fixed_cfg,model_responses,monkeypatch,fault):
    import run as runner
    import time
    fixed_cfg.agent.steps=2
    fixed_cfg.run_result_file=str(Path(fixed_cfg.log_dir).parent/'run-info.json')
    Path(fixed_cfg.run_result_file).parent.mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(runner,'load_cfg',lambda:fixed_cfg)
    monkeypatch.setattr(runner,'load_task_desc',lambda cfg:'Evaluation: binary log loss.')
    # Fixture has already performed workspace creation; runner still calls its hook.
    monkeypatch.setattr(runner,'prep_agent_workspace',lambda cfg:None)
    if fault:
        original=FixedEvaluation.score
        def fail(self,node):
            (self.directory/'truth.csv').unlink(missing_ok=True)
            return original(self,node)
        monkeypatch.setattr(FixedEvaluation,'score',fail)
        started=time.monotonic()
        with pytest.raises(EvaluationError):runner.run()
        assert time.monotonic()-started<25
        saved=serialize.load_json(fixed_cfg.log_dir/'journal.json',Journal)
        assert len(saved.nodes)==1 and saved.evaluation_state['failure']
    else:
        runner.run()
        assert _json(fixed_cfg.run_result_file)['fixed_evaluation'] is True
        assert (fixed_cfg.log_dir/'final_submission.json').is_file()
        saved=serialize.load_json(fixed_cfg.log_dir/'journal.json',Journal)
        assert len(saved.nodes)==3 and all(n.is_valid for n in saved.nodes[1:])


@pytest.mark.parametrize('mode,main_exit,expected,postprocess',[
    ('empty',2,2,None),('invalid',2,2,None),('fixed',0,0,'fixed'),
    ('legacy',0,0,None),('legacy',124,124,'legacy'),
    ('legacy',2,2,'legacy'),('legacy',130,130,None),
])
def test_shell_exact_run_info_and_early_failure(tmp_path,mode,main_exit,expected,postprocess):
    import os, shutil, subprocess, sys
    root=tmp_path/'runner'; root.mkdir(); binaries=tmp_path/'bin'; binaries.mkdir()
    shutil.copyfile(Path(__file__).parents[1]/'run_single_task.sh',root/'run_single_task.sh')
    (root/'launch_server.sh').write_text('#!/bin/bash\nexit 0\n')
    scripts={
        'curl':'#!/bin/bash\nexit 0\n',
        'timeout':'#!/bin/bash\nwhile [[ "$1" != python ]]; do shift; done\nexec "$@"\n',
        'python':f'''#!/bin/bash
if [[ "$1" == run.py ]]; then
  for arg in "$@"; do case "$arg" in run_result_file=*) marker="${{arg#run_result_file=}}";; esac; done
  case "$STUB_MODE" in
    fixed) printf '%s' '{{"fixed_evaluation":true,"log_dir":"exact/logs","workspace_dir":"exact/workspace"}}' > "$marker";;
    legacy) printf '%s' '{{"fixed_evaluation":false,"log_dir":"exact/logs","workspace_dir":"exact/workspace"}}' > "$marker";;
    invalid) echo invalid > "$marker";;
  esac
  exit "$STUB_EXIT"
elif [[ "$1" == -m ]]; then
  echo fixed > "$STUB_POSTPROCESS"; exit 0
elif [[ "$1" == utils/submission_fusion_utils.py ]]; then
  printf '%s\\n' "$@" > "$STUB_POSTPROCESS.args"
  echo legacy > "$STUB_POSTPROCESS"; exit 0
else
  exec {sys.executable!r} "$@"
fi
'''}
    for name,content in scripts.items():
        p=binaries/name;p.write_text(content);p.chmod(0o755)
    record=tmp_path/'postprocess'
    result=subprocess.run(['bash',str(root/'run_single_task.sh'),'fixture',str(tmp_path/'data'),'111'],
        env={**os.environ,'PATH':str(binaries)+os.pathsep+os.environ['PATH'],'STUB_MODE':mode,'STUB_EXIT':str(main_exit),'STUB_POSTPROCESS':str(record)},
        capture_output=True,text=True,timeout=15)
    assert result.returncode==expected,result.stdout+result.stderr
    assert (record.read_text().strip() if record.exists() else None)==postprocess
    if postprocess=='legacy':
        args=Path(str(record)+'.args').read_text().splitlines()
        assert '--run_info' in args and '--exp_name' not in args


def test_library_entry_executes_fixed_protocol(fixed_cfg,model_responses,monkeypatch):
    import importlib.util,sys
    module_spec=importlib.util.spec_from_file_location('mlevolve_test_library',Path(__file__).parents[1]/'__init__.py')
    module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
    monkeypatch.setattr(module,'_load_cfg',lambda **kwargs:copy.deepcopy(fixed_cfg))
    monkeypatch.setattr(module,'prep_cfg',lambda cfg:cfg)
    experiment=module.Experiment(str(fixed_cfg.data_dir),'Binary classification','Binary log loss',evaluation={'enabled':True})
    solution=experiment.run(1)
    assert 0 < solution.valid_metric < 1
    assert (fixed_cfg.log_dir/'final_submission.json').is_file()


def test_evolution_and_fusion_share_actual_scoring_entry(fixed_cfg,model_responses):
    from engine.parent_selectors.base import OperatorPreview
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); interpreter=Interpreter(fixed_cfg.workspace_dir,cfg=fixed_cfg,timeout=30)
    first=agent.step(None,interpreter.run,execute_immediately=False); agent.execute_deferred_node(first,interpreter.run)
    second=agent.step(first,interpreter.run)
    runtime=agent.phase1_runtime
    for operator in ('evolution','fusion'):
        preview=OperatorPreview(operator,reference_node_ids=(first.id,) if operator=='fusion' else ())
        context=runtime.record_direct_parent(agent,second,preview=preview,reason='format integration fixture')
        _,node=agent._run_single_step(second,context,interpreter.run)
        assert node.stage==operator and node.is_valid,node.analysis
        assert node.evaluation_protocol_id==first.evaluation_protocol_id
        assert runtime.transition_store.records()[-1].valid
        second=node


def test_stepwise_diff_and_review_use_compilable_protocol_prompt(fixed_cfg,model_responses,monkeypatch):
    from agents.coder.stepwise_coder import create_default_step_agents,StepwiseContext,MetaAgent
    from agents.coder.diff_coder.diff_generate import diff_generate_and_apply
    from agents.prompts import get_impl_guideline_from_agent
    from llm import compile_prompt_to_md
    agent=AgentSearch('Evaluation: binary log loss.',fixed_cfg,Journal()); agent.update_data_preview()
    base={'Introduction':'Build a model','Task description':agent.task_desc,'Instructions':get_impl_guideline_from_agent(agent)}
    for step in create_default_step_agents():
        text=step._build_prompt(agent.task_desc,agent.data_preview,[],base,agent,StepwiseContext())
        assert 'Final Validation Score' not in text
        assert 'fit_binary_temperature' in text and 'MLEV_EVALUATION_OUTPUT_DIR' in text
        assert 'split dataset into train/validation/test' not in text
    text=MetaAgent()._build_merge_prompt(agent.task_desc,agent.data_preview,[],base,agent,StepwiseContext())
    assert 'Final Validation Score' not in text and 'fit_binary_temperature' in text
    captures=[]
    def generate(**kwargs):
        captures.append(kwargs['prompt'])
        return 'Plan\n<<<<<<< SEARCH\nx=1\n=======\nx=2\n>>>>>>> REPLACE'
    monkeypatch.setattr('agents.coder.diff_coder.diff_generate.generate',generate)
    _,code=diff_generate_and_apply(agent,{'reason':'Improve','module':[]},'x=1',agent.data_preview,'','Improve')
    assert 'x=2' in code
    text=compile_prompt_to_md(captures[0])
    assert 'fit_binary_temperature' in text and agent.fixed_evaluation.protocol_id in text
