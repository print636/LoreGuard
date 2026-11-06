"""Server policy, durable source bindings and unchanged author reading."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import NARRATIVE_TIME_KNOWLEDGE_POLICY_V1, SEMANTIC_STATE_POLICY_V1, Settings
from app.db import AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunRow, AnalysisRunInputContextRow, IssueRow, SessionLocal
from app.domain import EvidenceSpan, ParsedDirective
from app.main import app, _public_diagnostics_payload
from app.projections import project_graph, project_timeline
from app.provider import OpenAICompatibleProvider
from app.semantic_state import SEMANTIC_STATE_INSTRUCTIONS, SEMANTIC_STATE_REFS_KEY, SemanticRelation, bind_semantic_state, digest
from app.semantic_state_storage import semantic_state_storage_payload
from app.service import _narrative_settings_for_run, execute_analysis, analysis_mode
from tests.test_account_provider import provider_test_settings, register, save_provider
from tests.test_character_observations import db, seed


@pytest.mark.parametrize('frozen', [None, 'other', SEMANTIC_STATE_POLICY_V1])
def test_semantic_policy_is_run_local_and_independent_of_character_review(frozen):
    base = Settings(_env_file=None, _semantic_state_policy=SEMANTIC_STATE_POLICY_V1)
    assert base._semantic_state_policy is None
    base._semantic_state_policy = SEMANTIC_STATE_POLICY_V1
    run = AnalysisRunRow(batch_coverage={'semantic_state_policy':frozen, 'narrative_time_knowledge_policy':NARRATIVE_TIME_KNOWLEDGE_POLICY_V1})
    local = _narrative_settings_for_run(base, run)
    assert local._semantic_state_policy == (frozen if frozen == SEMANTIC_STATE_POLICY_V1 else None)
    assert local._narrative_time_knowledge_policy == NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
    assert '_semantic_state_policy' not in local.model_dump()


def fact_wire():
    return {'kind':'fact', 'subject':'北门', 'predicate':'状态', 'value':'打开', 'source_line_start':1, 'source_line_end':1,
            'modality':'asserted','source_scope':'narrator','certainty':'certain',
            'semantic_relation':{'quote':'北门敞开', 'roles':{'subject':{'quote':'北门','relation':'same'},
                'predicate':{'quote':'敞开','relation':'equivalent'}, 'value':{'quote':'敞开','relation':'equivalent'}},
                'relation':'supported','actuality':'actual','basis':'explicit'}}


@pytest.mark.parametrize('enabled', [True, False])
def test_unresolved_state_comparison_keeps_coverage_partial_without_rewriting_extraction(enabled):
    counters = {'enabled':enabled,'configured':True,'succeeded_chunks':1,'failed_chunks':0,
                'skipped_chunks':0,'invalid_records':0,'empty_response_chunks':0}
    result = SimpleNamespace(diagnostics={'model':dict(counters),'semantic_state_comparison':{'unresolved_pairs':1}})
    assert analysis_mode(result)[1] is True
    assert result.diagnostics['model'] == counters
    result.diagnostics['semantic_state_comparison']['unresolved_pairs'] = 0
    assert analysis_mode(result)[1] is False


@pytest.mark.parametrize('damage', [None, 'source', 'context', 'record', 'policy', 'raw', 'offset', 'missing'])
def test_private_binding_uses_this_runs_frozen_source_and_exact_saved_record(db, damage):
    _, run, source = seed(db)
    source.content = '北门敞开。'
    source.content_sha256 = hashlib.sha256(source.content.encode()).hexdigest()
    run.batch_coverage = {'semantic_state_policy':SEMANTIC_STATE_POLICY_V1}
    wire = fact_wire()
    row = ParsedDirective(kind='fact', attrs={'subject':'北门','predicate':'状态','value':'打开'}, evidence=EvidenceSpan(
        document_id=source.document_id, document_name=source.document_name, line_start=1,line_end=1,text=source.content))
    row = bind_semantic_state(SimpleNamespace(kind='fact', semantic_relation=SemanticRelation.model_validate(wire['semantic_relation'])), row)
    record = AnalysisRecordRow(id='saved-record', run_id=run.id, kind=row.kind, attrs=dict(row.attrs), evidence=row.evidence.model_dump())
    if damage == 'source':
        source.content = '南门紧闭。'
        source.content_sha256 = hashlib.sha256(source.content.encode()).hexdigest()
    if damage == 'record': record.attrs = {**record.attrs, 'value':'关闭'}
    if damage == 'context':
        db.add(AnalysisRunInputContextRow(input_id=source.id, document_role='reference', story_scope='other-story'))
        db.flush()
    if damage == 'policy': run.batch_coverage = {}
    if damage in {'raw','offset'}:
        proof = deepcopy(row.semantic_state)
        if damage == 'raw': proof['model_text'] = 'must not persist'
        else: proof['refs']['value']['end'] = 1000
        proof['payload_sha256'] = digest({key:value for key,value in proof.items() if key != 'payload_sha256'})
        row = row.model_copy(update={'semantic_state':proof})
    if damage == 'missing':
        row = row.model_copy(update={'semantic_state':None,'attrs':{**row.attrs,'state_kind':'custody'}})
        record.attrs = row.attrs
    if damage:
        with pytest.raises(ValueError, match='integrity'):
            semantic_state_storage_payload(db, run=run, records=[record], directives=[row])
    else:
        payload = semantic_state_storage_payload(db, run=run, records=[record], directives=[row])
        assert payload['receipts'][0]['record_id'] == record.id
        assert payload['receipts'][0]['source_input_id'] == source.id
        assert '北门' not in json.dumps(payload,ensure_ascii=False)
        assert _public_diagnostics_payload({SEMANTIC_STATE_REFS_KEY:payload,'model':{'used':True}}) == {'model':{'used':True}}
        assert semantic_state_storage_payload(db, run=run, records=[record], directives=[row.model_copy(update={'semantic_state':None})]) is None


def test_normal_account_persists_binding_and_keeps_old_retry_and_full_reader():
    from app.model_extractor import BATCH_SYSTEM_PROMPT_NARRATIVE_V1, SYSTEM_PROMPT_NARRATIVE_V1
    sent = []
    def respond(request):
        body = json.loads(request.content)
        system,user = (message['content'] for message in body['messages'])
        sent.append(system)
        current = system.endswith(SEMANTIC_STATE_INSTRUCTIONS)
        base = system[:-len(SEMANTIC_STATE_INSTRUCTIONS)] if current else system
        assert base in {BATCH_SYSTEM_PROMPT_NARRATIVE_V1, SYSTEM_PROMPT_NARRATIVE_V1}
        rows = [fact_wire()] if current else []
        value = {'records':rows}
        if base == BATCH_SYSTEM_PROMPT_NARRATIVE_V1:
            value = {'documents':[{'doc_ref':doc['doc_ref'],'records':rows} for doc in json.loads(user.split('\n',1)[1])['documents']]}
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(value)}}],'usage':{'prompt_tokens':12,'completion_tokens':8}})
    original = OpenAICompatibleProvider.__init__
    def provider_init(self, config=None, **kwargs):
        kwargs['transport'] = httpx.MockTransport(respond)
        original(self, config, **kwargs)
    with provider_test_settings(), TestClient(app) as client:
        _,csrf = register(client)
        client.headers['X-CSRF-Token'] = csrf
        assert save_provider(client,csrf,expected_revision=0,api_key='synthetic-only',model='state-test').status_code == 200
        project = client.post('/api/v1/projects',json={'name':'State source'}).json()
        documents = f"/api/v1/projects/{project['id']}/documents/text"
        doc = client.post(documents,json={'name':'draft.md','content':'北门敞开。','document_role':'chapter'}).json()
        runs = f"/api/v1/projects/{project['id']}/analysis-runs"
        with patch('app.main.dispatch_analysis'):
            run_id = client.post(runs,json={}).json()['id']
            for override in ({'_semantic_state_policy':'other'},{'semantic_state_policy':'other'},{'batch_coverage':{'semantic_state_policy':'other'}}):
                assert client.post(runs,json=override).status_code == 422
        with patch.object(OpenAICompatibleProvider,'__init__',provider_init): execute_analysis(run_id,raise_on_failure=True)
        with SessionLocal() as session:
            run = session.get(AnalysisRunRow,run_id)
            assert run.status == 'completed'
            assert run.batch_coverage['semantic_state_policy'] == SEMANTIC_STATE_POLICY_V1
            assert not run.batch_coverage.get('character_review_enabled')
            proof = session.get(AnalysisDiagnosticRow,run_id).payload[SEMANTIC_STATE_REFS_KEY]
            receipt, = proof['receipts']
            stored = session.get(AnalysisRecordRow,receipt['record_id'])
            assert stored.run_id == run_id and stored.attrs['value'] == '打开'
            assert receipt['record_sha256'] == digest({'kind':stored.kind,'attrs':stored.attrs,'evidence':stored.evidence})
            issue = IssueRow(run_id=run_id,report_class='formal',category='fact_conflict',severity='high',confidence=.9,
                title='原文阅读夹具',explanation='仅验证原文入口，不代表语义判定。',suggestion='核对',evidence=[stored.evidence],extra={})
            session.add(issue)
            session.commit()
            issue_id = issue.id
        assert len(sent) == 1
        for suffix in ('records','diagnostics','export.md'):
            response = client.get(f'/api/v1/analysis-runs/{run_id}/{suffix}')
            assert response.status_code == 200
            assert SEMANTIC_STATE_REFS_KEY not in response.text and '"semantic_state"' not in response.text and 'claim_sha256' not in response.text
        preview = client.get(f'/api/v1/analysis-runs/{run_id}/evidence-preview',params={'kind':'issue','item_id':issue_id,'evidence_index':0})
        assert preview.status_code == 200
        assert [line['text'] for line in preview.json()['lines'] if line['is_evidence']] == ['北门敞开。']
        with patch('app.main.dispatch_analysis'):
            old_id = client.post(runs,json={}).json()['id']
            with SessionLocal() as session:
                old = session.get(AnalysisRunRow,old_id)
                old.batch_coverage = {k:v for k,v in old.batch_coverage.items() if k != 'semantic_state_policy'}
                old.status = 'failed'
                session.commit()
            retry = client.post(f'/api/v1/analysis-runs/{old_id}/retry').json()['id']
            with SessionLocal() as session: assert 'semantic_state_policy' not in session.get(AnalysisRunRow,retry).batch_coverage
            with patch.object(OpenAICompatibleProvider,'__init__',provider_init): execute_analysis(retry,raise_on_failure=True)
            assert not sent[-1].endswith(SEMANTIC_STATE_INSTRUCTIONS)
            assert client.post(documents,json={'name':'draft.md','content':'北门紧闭。','replace_document_id':doc['id']}).status_code == 201
            revised = client.post(f'/api/v1/analysis-runs/{run_id}/rechecks').json()['id']
            with SessionLocal() as session: assert session.get(AnalysisRunRow,revised).batch_coverage['semantic_state_policy'] == SEMANTIC_STATE_POLICY_V1
        with TestClient(app) as other:
            register(other)
            assert other.get(f'/api/v1/analysis-runs/{run_id}/records').status_code == 404


@pytest.mark.parametrize('state', ['observed_possession','custody','transfer','retained_custody','unclear'])
def test_graph_and_timeline_preserve_state_meaning(state):
    row = SimpleNamespace(id='item',kind='item',attrs={'item':'铜钥匙','owner':'林澈','time':'12:00',
        'state_kind':state,'transfer_from':'乔予','through_time':'13:00'},evidence={'document_id':'d','document_name':'draft.md','line_start':1,'line_end':1,'text':'原文。'})
    graph = project_graph('run',[row],[])
    timeline = project_timeline('run',[row],[])
    labels = json.dumps([edge.label for edge in graph.edges],ensure_ascii=False)
    titles = json.dumps([entry.title for entry in timeline.unscheduled],ensure_ascii=False)
    assert {'observed_possession':'观察到持有','custody':'保管','transfer':'交接后持有','retained_custody':'明确未交接','unclear':'待核实'}[state] in labels
    assert '所有权' not in labels + titles


def test_damaged_binding_fails_run_without_persisting_its_formal_result():
    from app.domain import ConsistencyIssue
    from app.pipeline import PipelineResult
    from app.main import write_limiter
    from tests.test_character_observation_pipeline import _new_review
    from tests.test_character_consistency_stage import _context, _create_document
    write_limiter.events.clear()
    with TestClient(app) as client:
        project = client.post('/api/v1/projects',json={'name':'Damaged state source'}).json()
        document = _create_document(client,project['id'],name='draft.md',role='chapter',content='北门敞开。',narrative_context=_context(publication='draft'))
        run_id = _new_review(client,project['id'],document['id'])
        evidence = EvidenceSpan(document_id=document['id'],document_name='draft.md',line_start=1,line_end=1,text='北门敞开。')
        directive = ParsedDirective(kind='fact',attrs={'subject':'北门','predicate':'状态','value':'打开'},evidence=evidence,semantic_state={'raw':'must-not-persist'})
        issue = ConsistencyIssue(category='fact_conflict',severity='high',confidence=.9,title='Damaged proof result',explanation='Must not persist.',suggestion='核对',evidence=[evidence])
        result = PipelineResult(directives=[directive],issues=[issue])
        runtime = SimpleNamespace(settings=Settings(_env_file=None,enable_model_extraction=False,enable_character_consistency=False),identity='server_default',available=True,unavailable_reason=None)
        with patch('app.service.runtime_for_analysis_run',return_value=runtime), patch('app.service.AnalysisPipeline',new=lambda:SimpleNamespace(run=lambda *a,**kw:result)):
            execute_analysis(run_id)
        with SessionLocal() as session:
            assert session.get(AnalysisRunRow,run_id).status == 'failed'
            assert not session.scalars(select(IssueRow).where(IssueRow.run_id==run_id)).all()
            assert not session.scalars(select(AnalysisRecordRow).where(AnalysisRecordRow.run_id==run_id)).all()
