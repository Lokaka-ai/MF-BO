"""Choose the largest original completed-node rating over the native legal pool."""
import copy
import time
import uuid
from .base import CandidateScore,NORMAL_OPERATORS,SelectionDecision,SelectorResult,utc_timestamp
from .study7_gp import Study7Error
from .study7_selector import Study7Selector,log_fatal,winner

class RatingOnlySelector(Study7Selector):
    def __init__(self,**kwargs):
        super().__init__(**kwargs,greedy=True)
        self.name='study7_rating_only_greedy'

    def contract(self):
        return {**super().contract(),'ranking':'completed_parent_potential_level_minus_one',
            'missing_rating':'neutral_zero_same_as_AQ04',
            'tie_rule':'birth_commit_index_then_operator_then_parent_id',
            'surrogate':None,'additional_selection_model_calls':0}

    @log_fatal
    def features(self,agent,parent,operator):
        data=super().features(agent,parent,operator)
        value=getattr(parent,'agent_potential_level',None)
        legal=type(value) is int and value in (0,1,2)
        if parent.is_buggy is not False or parent.is_valid is False:legal=False
        data.update(potential_level=value if legal else None,rating_score=value-1 if legal else 0,
            rating_assessed_at=getattr(parent,'agent_potential_assessed_at',None),
            missing_rating_policy='neutral_zero_same_as_AQ04')
        return data

    @log_fatal
    def select(self,agent,options):
        started=time.perf_counter()
        if not options or any(o.operator not in NORMAL_OPERATORS for o in options):
            raise Study7Error('Rating-only selection requires a native ordinary legal pool')
        self.observe(self.records_provider())
        values=[o.state_features['study7']['rating_score'] for o in options]
        chosen=winner(options,values)
        scores=tuple(CandidateScore(parent_node_id=o.parent_node_id,operator=o.operator,
            reference_node_ids=o.reference_node_ids,parent_utility=o.parent_utility,
            mean_gain=None,std_gain=None,predicted_child_ucb=None,predicted_cost=None,
            p_valid=None,acquisition_score=float(values[i])) for i,o in enumerate(options))
        selected=options[chosen]
        decision=SelectionDecision(self.name,selected.parent_node_id,selected.operator,
            float(values[chosen]),None,False,uuid.uuid4().hex,utc_timestamp(),
            decision_wall_time_sec=time.perf_counter()-started,selection_reason='highest_agent_rating_stable_tie')
        self.last_prediction_details={}
        candidates=[]
        for i,o in enumerate(options):
            details={'method_id':'AQ09','acquisition_kind':'agent_rating_only',
                'acquisition_score_units':'centered_agent_rating','is_exact_gp_posterior':False,
                'uncertainty_used':None,'gp_fit_performed':False,'agent_selection_call_performed':False,
                'potential_level':o.state_features['study7']['potential_level'],'rating_score':values[i]}
            self.last_prediction_details[(o.parent_node_id,o.operator)]=details
            candidates.append({'parent_id':o.parent_node_id,'operator':o.operator,
                'reference_ids':list(o.reference_node_ids),'state':copy.deepcopy(o.state_features['study7']),
                'scores':scores[i].to_dict(),**details})
        if self.logger:
            self.logger._append('study7_decisions.jsonl',{'method_id':'AQ09','decision_id':decision.decision_id,
                'timestamp':decision.timestamp,'active_time':max(0.,time.time()-agent.start_time),
                'model_version':None,'chosen_index':chosen,'candidate_count':len(options),
                'candidate_order':[[o.state_features['study7']['birth_commit_index'],NORMAL_OPERATORS.index(o.operator),o.parent_node_id] for o in options],
                'tie_rule':self.contract()['tie_rule'],'contract':self.contract(),
                'protocol_id':getattr(agent.autoresearch,'protocol_id',None),'candidates':candidates})
        return SelectorResult(decision,selected,scores,tuple(options))
