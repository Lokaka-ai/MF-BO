"""Read-only, offline comparison of native selection traces and admitted BPB results."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # A writer can have an incomplete final append.
    return rows


def seconds(value):
    return datetime.fromisoformat(value).timestamp() if value else 0


def atomic(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def csv_rows(path, rows, fields):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def report(campaign):
    manifest = read_json(campaign / 'campaign.json')
    summaries, trial_rows, candidates, all_decisions = [], [], [], []
    for setting, arm in manifest.get('settings', {}).items():
        directory = Path(arm['directory'])
        status = read_json(directory / 'status.json')
        recovery = read_json(directory / 'recovery.json')
        if recovery.get('status_file'):
            resumed = read_json(Path(recovery['status_file']))
            if resumed:
                original_start = status.get('started_at')
                status.update(resumed)
                status['started_at'] = original_start
        runs = sorted((directory / 'runs').glob('*/logs'))
        logdir = runs[-1] if runs else directory / 'not-started'
        phase = logdir / 'phase1'
        transitions = read_lines(phase / 'transitions.jsonl')
        decisions = read_lines(phase / 'selection_decisions.jsonl')
        snapshots = read_lines(phase / 'selection_snapshots.jsonl')
        continuation = read_lines(phase / 'continuation_events.jsonl')
        journal = read_json(logdir/'journal.json')
        node_by_id = {n['id']: n for n in journal.get('nodes', [])}
        start = seconds(status.get('started_at'))
        best = None
        counts = Counter(s['decision_id'] for s in snapshots)
        chosen_branches = Counter()
        for snap in snapshots:
            candidates.append(dict(setting=setting, branch_id=node_by_id.get(snap['parent_node_id'], {}).get('branch_id'), **snap))
            if snap.get('selected'):
                branch = node_by_id.get(snap['parent_node_id'], {}).get('branch_id')
                chosen_branches[str(branch)] += 1
        for row in sorted(transitions, key=lambda r: r['timestamp']):
            value = row.get('child_metric_raw') if row.get('valid') else None
            if value is not None:
                best = min(best, value) if best is not None else value
            trial = logdir.parent / 'workspace/trials' / str(row.get('child_node_id'))
            official = read_json(trial / 'result.json')
            node = node_by_id.get(row.get('child_node_id'), {})
            trial_rows.append(dict(setting=setting, elapsed_seconds=seconds(row['timestamp'])-start,
                best_bpb=best, val_bpb=value, valid=row.get('valid'), node_id=row.get('child_node_id'),
                parent_node_id=row.get('parent_node_id'), operator=row['operator'],
                selector=row['selector_name'], parent_bpb=row.get('parent_metric_raw'),
                parent_improvement=row.get('improvement'), generation_seconds=row.get('generation_wall_time_sec'),
                execution_seconds=row.get('execution_wall_time_sec'),
                training_seconds=official.get('training_seconds'), tokens=official.get('total_tokens'),
                params=official.get('num_params'), peak_vram_mb=official.get('peak_vram_mb'),
                shared_draft_slot=node.get('shared_draft_slot'), shared_draft_pool_id=node.get('shared_draft_pool_id'),
                trial_directory=str(trial), decision_id=row.get('decision_id')))
        for row in decisions:
            all_decisions.append(dict(setting=setting, candidate_count=counts[row['decision_id']], **row))
        valid = [r for r in transitions if r.get('valid')]
        normal = [d for d in decisions if d['selected_operator'] in ('improve','evolution','fusion')]
        entry = dict(setting=setting, node=status.get('hostname'), state=status.get('state','starting'),
            started_at=status.get('started_at'), deadline_at=status.get('deadline_at'),
            elapsed_seconds=max(0, seconds(status.get('finished_at'))-start) if status.get('finished_at') else max(0,time.time()-start),
            completed=len(transitions), valid=len(valid), best_bpb=best,
            infrastructure_restarts=len(list(directory.glob('status.json.before-resume-*'))),
            normal_decisions=len(normal), multiple_candidate_decisions=sum(counts[d['decision_id']] > 1 for d in normal),
            selected_parents=dict(Counter(d['selected_parent_id'] for d in normal)),
            operators=dict(Counter(r['operator'] for r in transitions)),
            selectors=dict(Counter(d['selector_name'] for d in normal)),
            fallback_decisions=sum(bool(d.get('fallback_used')) for d in normal),
            continuation_outcomes=dict(Counter(d['outcome'] for d in continuation)),
            selected_branches=dict(chosen_branches),
            global_memory_records=len(read_json(logdir.parent/'workspace/global_memory/records.json')),
            valid_branches=len({node_by_id.get(r.get('child_node_id'),{}).get('branch_id') for r in valid if node_by_id.get(r.get('child_node_id'),{}).get('branch_id') is not None}),
            shared_draft_slots=[a['slot'] for a in read_json(logdir/'shared_draft_usage.json').get('assignments',[])],
            logs=str(logdir), configuration_audit=str(directory/'configuration_audit.json'))
        summaries.append(entry)
    payload = {'updated_at':datetime.now(timezone.utc).isoformat(),
               'shared_draft_pool_id':manifest.get('shared_draft_pool_id'),
               'infrastructure_interrupted':any(s['infrastructure_restarts'] for s in summaries), 'settings':summaries,
               'trials':trial_rows, 'decisions':all_decisions, 'candidates':candidates}
    atomic(campaign/'comparison.json', json.dumps(payload,indent=2,allow_nan=False)+'\n')
    csv_rows(campaign/'trials.csv', trial_rows, ['setting','elapsed_seconds','best_bpb','val_bpb','valid','node_id','parent_node_id','operator','selector','parent_bpb','parent_improvement','generation_seconds','execution_seconds','training_seconds','tokens','params','peak_vram_mb','shared_draft_slot','shared_draft_pool_id','trial_directory','decision_id'])
    csv_rows(campaign/'selection_candidates.csv', candidates, ['setting','decision_id','timestamp','selector_name','selection_reason','parent_node_id','operator','selected','parent_metric','global_best_metric','acquisition_score','posterior_mean_quality','posterior_std_latent','predictive_std','base_logei','agent_potential_level','rating_prior_bonus','rating_prior_applied','branch_id','fallback_used','fallback_reason','depth','visit_count'])
    lines=['# Native nanoGPT comparison', '', 'Updated: '+payload['updated_at'], '',
        '| Setting | State | Completed / valid | Best BPB ↓ | Normal choices / multi-candidate |',
        '| --- | --- | --- | --- | --- |']
    for s in summaries:
        score=f"{s['best_bpb']:.6f}" if s['best_bpb'] is not None else 'pending'
        lines.append(f"| {s['setting']} | {s['state']} | {s['completed']} / {s['valid']} | {score} | {s['normal_decisions']} / {s['multiple_candidate_decisions']} |")
    draft_note = ('The five reviewed drafts are shared in fixed order. Training, execution feedback and subsequent edits remain independent. Cached generation/review latency is replayed inside each eight-hour budget.'
                  if manifest.get('shared_draft_pool') else 'Initial drafts are generated independently.')
    lines += ['', 'The metric is the admitted official BPB, lower is better. Curves include API, compilation, training and evaluation wall time. The configured search workers share each arm\'s GPU. One run per setting does not establish statistical superiority. '+draft_note, '',
              '[Interactive offline report](comparison.html) · [Trials](trials.csv) · [Candidate scores](selection_candidates.csv)', '']
    atomic(campaign/'comparison.md', '\n'.join(lines))
    page='''<!doctype html><meta charset="utf-8"><title>nanoGPT native MLEvolve comparison</title>
<style>body{font:15px system-ui;margin:32px;color:#203040;max-width:1400px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dde3ea;text-align:left}svg{width:100%;background:#fafbfe;border:1px solid #dde3ea}select{padding:8px;max-width:100%}.muted{color:#64748b}pre{white-space:pre-wrap;font-size:12px}circle:hover{r:6}</style>
<h1>nanoGPT · full native MLEvolve</h1><p id="updated"></p><p id="protocol"></p><p class="muted">Official admitted BPB ↓ versus wall-clock hours. Each setting: 1 A100, 3 native search workers, 8 h. Hover points; choose a decision to inspect every eligible parent and score. Refresh the page for new data.</p><div id="summary"></div><h2>Best BPB over time</h2><svg id="chart" viewBox="0 0 1100 340"></svg><div id="legend"></div><h2>Parent selection evidence</h2><select id="decision"></select><div id="selected"></div><div id="options"></div><h2>Search behavior</h2><pre id="behavior"></pre><p class="muted">A single run per setting is an observed comparison, not a statistically conclusive ranking. Concurrent load and stochastic code generation can influence results. Raw per-arm phase1 traces remain authoritative.</p>
<script>
const data=__DATA__;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>typeof x==='number'?Number(x.toFixed(6)):x??'pending';
function table(rows,fields){return '<table><tr>'+fields.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+fields.map(k=>'<td>'+esc(fmt(r[k]))+'</td>').join('')+'</tr>').join('')+'</table>'}
document.getElementById('updated').textContent='Updated '+data.updated_at;
document.getElementById('protocol').textContent=(data.shared_draft_pool_id?'Paired five-draft pool: '+data.shared_draft_pool_id:'Independent initial drafts')+(data.infrastructure_interrupted?' | Allocation interruption occurred; inspect restarts before interpreting rankings.':'');
document.getElementById('summary').innerHTML=table(data.settings,['setting','state','completed','valid','best_bpb','normal_decisions','multiple_candidate_decisions','fallback_decisions','valid_branches','global_memory_records','shared_draft_slots','infrastructure_restarts']);
const colors=['#2563eb','#ea580c','#0d9488','#9333ea'], points=data.trials.filter(p=>p.best_bpb!==null),svg=document.getElementById('chart');
if(points.length){let lo=Math.min(...points.map(p=>p.best_bpb)),hi=Math.max(...points.map(p=>p.best_bpb));if(hi-lo<.01){hi+=.005;lo-=.005}let x=t=>70+t/28800*990,y=v=>295-(v-lo)/(hi-lo)*265;
let s='';for(let h=0;h<=8;h++)s+='<text x="'+x(h*3600)+'" y="325" font-size="12">'+h+' h</text>';
for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;s+='<line x1="65" x2="1065" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#e2e8f0"/><text x="3" y="'+y(v)+'" font-size="12">'+v.toFixed(4)+'</text>'}
data.settings.forEach((arm,i)=>{const p=points.filter(p=>p.setting===arm.setting);if(!p.length)return;let d='M '+x(p[0].elapsed_seconds)+' '+y(p[0].best_bpb);for(let j=1;j<p.length;j++)d+=' H '+x(p[j].elapsed_seconds)+' V '+y(p[j].best_bpb);s+='<path d="'+d+'" fill="none" stroke="'+colors[i]+'" stroke-width="2.5"/>';p.forEach(r=>s+='<circle cx="'+x(r.elapsed_seconds)+'" cy="'+y(r.best_bpb)+'" r="3" fill="'+colors[i]+'"><title>'+esc(r.setting+' / '+r.operator+' / BPB '+r.val_bpb+' / node '+r.node_id)+'</title></circle>')});svg.innerHTML=s;}
document.getElementById('legend').innerHTML=data.settings.map((s,i)=>'<span style="margin-right:24px;color:'+colors[i]+'">● '+esc(s.setting)+'</span>').join('');
const select=document.getElementById('decision');select.innerHTML=data.decisions.map((d,i)=>'<option value="'+i+'">'+esc(d.setting+' · '+d.timestamp+' · '+d.selected_operator+' · '+d.selection_reason+' · candidates='+d.candidate_count)+'</option>').join('');
function show(){const d=data.decisions[select.value];if(!d)return;document.getElementById('selected').innerHTML='<pre>'+esc(JSON.stringify(d,null,2))+'</pre>';document.getElementById('options').innerHTML=table(data.candidates.filter(c=>c.setting===d.setting&&c.decision_id===d.decision_id),['selected','parent_node_id','operator','parent_metric','depth','visit_count','acquisition_score','posterior_mean_quality','posterior_std_latent','predictive_std','base_logei','agent_potential_level','rating_prior_bonus','rating_prior_applied','fallback_used']);}select.onchange=show;if(data.decisions.length){select.value=data.decisions.length-1;show()}
document.getElementById('behavior').textContent=JSON.stringify(data.settings.map(s=>({setting:s.setting,operators:s.operators,selectors:s.selectors,continuation:s.continuation_outcomes,parents:s.selected_parents,logs:s.logs})),null,2);
</script>'''
    atomic(campaign/'comparison.html', page.replace('__DATA__',json.dumps(payload,allow_nan=False).replace('<','\\u003c')))
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('--watch',action='store_true',help='Refresh every 30 seconds until all arms finish (at most 9 h)')
    args=parser.parse_args()
    deadline=time.monotonic()+9*3600
    while True:
        summaries=report(args.campaign.resolve())
        print(json.dumps([{k:s[k] for k in ('setting','state','completed','valid','best_bpb','normal_decisions')} for s in summaries]),flush=True)
        if not args.watch or time.monotonic()>deadline or (summaries and all(s['state'] in ('finished','failed') for s in summaries)):
            break
        time.sleep(30)


if __name__=='__main__':
    main()
