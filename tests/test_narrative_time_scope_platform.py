"""Normal-run freezing and durable, private temporal source bindings."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import NARRATIVE_TIME_SCOPE_POLICY_V1, SEMANTIC_STATE_POLICY_V1, Settings
from app.db import AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunRow, AnalysisRunInputContextRow, IssueRow, SessionLocal
from app.domain import EvidenceSpan, ParsedDirective
from app.main import app, _public_diagnostics_payload
from app.narrative_relations import _digest
from app.narrative_time_scope import NARRATIVE_TIME_SCOPE_INSTRUCTIONS, TIME_SCOPE_REFS_KEY, NarrativeTimeScopeExtraction, bind_time_scope
from app.narrative_time_scope_storage import time_scope_storage_payload
from app.pipeline import DocumentInput
from app.projections import project_graph, project_timeline
from app.provider import OpenAICompatibleProvider
from app.semantic_state import SEMANTIC_STATE_INSTRUCTIONS
from app.service import _narrative_settings_for_run, execute_analysis
from tests.test_account_provider import provider_test_settings, register, save_provider
from tests.test_character_observations import db, seed


BODY = '这三处值守发生在同一天。\n12:00，许宁在北楼值守。\n12:00，许宁在南门值守。\n12:30，许宁在东岸值守。'


def scope_wire():
    return dict(kind='narrative_time_scope',source_line_start=1,source_line_end=1,
        modality='asserted',source_scope='narrator',certainty='certain',basis='explicit',
        relation='same_day',relation_quote=BODY.splitlines()[0],omitted_count=0,
        members=[dict(doc_ref='d1',line_start=i,line_end=i,event_quote=text,time=time,
                      clock=time,clock_relation='same') for i,text,time in
                 ((2,BODY.splitlines()[1],'12:00'),(3,BODY.splitlines()[2],'12:00'),(4,BODY.splitlines()[3],'12:30'))])


@pytest.mark.parametrize('frozen',[None,'unknown',NARRATIVE_TIME_SCOPE_POLICY_V1])
def test_worker_restores_only_exact_server_frozen_time_scope_policy(frozen):
    base = Settings(_env_file=None,_narrative_time_scope_policy=NARRATIVE_TIME_SCOPE_POLICY_V1)
    assert base._narrative_time_scope_policy is None
    base._narrative_time_scope_policy = NARRATIVE_TIME_SCOPE_POLICY_V1
    local = _narrative_settings_for_run(base,AnalysisRunRow(batch_coverage={
        'narrative_time_scope_policy':frozen,'semantic_state_policy':SEMANTIC_STATE_POLICY_V1}))
    assert local._narrative_time_scope_policy == (frozen if frozen == NARRATIVE_TIME_SCOPE_POLICY_V1 else None)
    assert local._semantic_state_policy == SEMANTIC_STATE_POLICY_V1
    assert '_narrative_time_scope_policy' not in local.model_dump()


def scope_rows(run,source,*,wire=None):
    span = lambda line: EvidenceSpan(document_id=source.document_id,document_name=source.document_name,
        line_start=line,line_end=line,text=source.content.splitlines()[line-1])
    scope = ParsedDirective(kind='narrative_time_scope',attrs={'document_role':'chapter','story_scope':'global'},evidence=span(1))
    scope = bind_time_scope(NarrativeTimeScopeExtraction.model_validate(wire or scope_wire()),scope,
        {'d1':(DocumentInput(source.document_id,source.document_name,source.content,'chapter','global'),SimpleNamespace(global_line_start=1,global_line_end=4))})
    events = [ParsedDirective(kind='event',attrs={'time':'12:00','location':place,'participants':'许宁','story_scope':'global'},evidence=span(i))
              for i,place in ((2,'北楼'),(3,'南门'))]
    rows = events+[scope]
    records = [AnalysisRecordRow(id=f'record-{i}',run_id=run.id,kind=row.kind,attrs=dict(row.attrs),evidence=row.evidence.model_dump()) for i,row in enumerate(rows)]
    return rows,records


@pytest.mark.parametrize('damage',[None,'source','context','scope_record','member_record','private_text','policy','missing'])
def test_time_scope_storage_binds_frozen_sources_and_only_actual_records(db,damage):
    _,run,source = seed(db)
    source.content = BODY
    source.content_sha256 = hashlib.sha256(BODY.encode()).hexdigest()
    run.batch_coverage = {'narrative_time_scope_policy':NARRATIVE_TIME_SCOPE_POLICY_V1}
    rows,records = scope_rows(run,source)
    if damage == 'source':
        source.content = BODY.replace('南门','西门')
        source.content_sha256 = hashlib.sha256(source.content.encode()).hexdigest()
    if damage == 'context':
        db.add(AnalysisRunInputContextRow(input_id=source.id,document_role='reference',story_scope='other'))
        db.flush()
    if damage == 'scope_record': records[-1].attrs = {**records[-1].attrs,'relation':'ordered'}
    if damage == 'member_record': records[0].attrs = {**records[0].attrs,'location':'东岸'}
    if damage == 'policy': run.batch_coverage = {}
    if damage == 'missing': rows[-1] = rows[-1].model_copy(update={'narrative_time_scope':None})
    if damage == 'private_text':
        proof = deepcopy(rows[-1].narrative_time_scope)
        proof['model_text'] = 'must not persist'
        proof['payload_sha256'] = _digest({k:v for k,v in proof.items() if k!='payload_sha256'})
        rows[-1] = rows[-1].model_copy(update={'narrative_time_scope':proof})
    if damage:
        with pytest.raises(ValueError,match='integrity'):
            time_scope_storage_payload(db,run=run,records=records,directives=rows)
    else:
        payload = time_scope_storage_payload(db,run=run,records=records,directives=rows)
        receipt, = payload['receipts']
        assert [m['matched_record_ids'] for m in receipt['members']] == [['record-0'],['record-1'],[]]
        assert '许宁' not in json.dumps(payload,ensure_ascii=False)
        assert _public_diagnostics_payload({TIME_SCOPE_REFS_KEY:payload,'model':{'used':True}}) == {'model':{'used':True}}
        assert time_scope_storage_payload(db,run=run,records=records[:2],directives=rows[:2]) is None


def test_time_scope_storage_retains_unique_shared_time_prefix_and_actual_event_matches(db):
    _,run,source = seed(db)
    source.content = BODY
    source.content_sha256 = hashlib.sha256(BODY.encode()).hexdigest()
    run.batch_coverage = {'narrative_time_scope_policy':NARRATIVE_TIME_SCOPE_POLICY_V1}
    wire = scope_wire()
    for member in wire['members']:
        member['event_quote'] = member['event_quote'].split('，',1)[1]
    rows,records = scope_rows(run,source,wire=wire)
    payload = time_scope_storage_payload(db,run=run,records=records,directives=rows)
    receipt, = payload['receipts']
    assert [member['matched_record_ids'] for member in receipt['members']] == [['record-0'],['record-1'],[]]
    for member in receipt['binding']['members']:
        assert member['time_ref']['end'] <= member['event_ref']['start']
        assert member['time_ref']['text_sha256'] == _digest(source.content.splitlines()[member['line_start']-1].split('，',1)[0])
    assert receipt['source_input_id'] == source.id
    assert receipt['record_sha256'] == _digest({'kind':records[-1].kind,'attrs':records[-1].attrs,'evidence':records[-1].evidence})


def test_normal_account_scope_reaches_storage_report_reader_and_legacy_retry():
    from app.model_extractor import BATCH_SYSTEM_PROMPT_NARRATIVE_V1,BATCH_SYSTEM_PROMPT_NARRATIVE_TIME_SCOPE_V1,SYSTEM_PROMPT_NARRATIVE_V1
    sent=[]
    def respond(request):
        body=json.loads(request.content)
        system,user=(m['content'] for m in body['messages'])
        sent.append(system)
        current=system.endswith(NARRATIVE_TIME_SCOPE_INSTRUCTIONS)
        base=system[:-len(NARRATIVE_TIME_SCOPE_INSTRUCTIONS)] if current else system
        batch=BATCH_SYSTEM_PROMPT_NARRATIVE_TIME_SCOPE_V1 if current else BATCH_SYSTEM_PROMPT_NARRATIVE_V1
        assert base in {batch+SEMANTIC_STATE_INSTRUCTIONS,SYSTEM_PROMPT_NARRATIVE_V1+SEMANTIC_STATE_INSTRUCTIONS}
        events=[dict(kind='event',time='12:00',location=place,participants=['许宁'],source_line_start=i,source_line_end=i,
                     modality='asserted',source_scope='narrator',certainty='certain') for i,place in ((2,'北楼'),(3,'南门'))]
        rows=events+[scope_wire()] if current else []
        value={'records':rows}
        if base == batch+SEMANTIC_STATE_INSTRUCTIONS:
            value={'documents':[{'doc_ref':doc['doc_ref'],'records':rows} for doc in json.loads(user.split('\n',1)[1])['documents']]}
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(value)}}],'usage':{'prompt_tokens':12,'completion_tokens':8}})
    original=OpenAICompatibleProvider.__init__
    def provider_init(self,config=None,**kwargs):
        kwargs['transport']=httpx.MockTransport(respond)
        original(self,config,**kwargs)
    with provider_test_settings(),TestClient(app) as client:
        _,csrf=register(client)
        client.headers['X-CSRF-Token']=csrf
        assert save_provider(client,csrf,expected_revision=0,api_key='synthetic-only',model='scope-test').status_code==200
        project=client.post('/api/v1/projects',json={'name':'Time scope source'}).json()
        documents=f"/api/v1/projects/{project['id']}/documents/text"
        doc=client.post(documents,json={'name':'draft.md','content':BODY,'document_role':'chapter'}).json()
        runs=f"/api/v1/projects/{project['id']}/analysis-runs"
        with patch('app.main.dispatch_analysis'):
            run_id=client.post(runs,json={}).json()['id']
            for override in ({'_narrative_time_scope_policy':'other'},{'narrative_time_scope_policy':'other'},{'batch_coverage':{'narrative_time_scope_policy':'other'}}):
                assert client.post(runs,json=override).status_code==422
        with patch.object(OpenAICompatibleProvider,'__init__',provider_init): execute_analysis(run_id,raise_on_failure=True)
        with SessionLocal() as session:
            run=session.get(AnalysisRunRow,run_id)
            assert run.status=='completed'
            assert run.batch_coverage['narrative_time_scope_policy']==NARRATIVE_TIME_SCOPE_POLICY_V1
            assert not run.batch_coverage.get('character_review_enabled')
            receipt,=session.get(AnalysisDiagnosticRow,run_id).payload[TIME_SCOPE_REFS_KEY]['receipts']
            stored=session.get(AnalysisRecordRow,receipt['record_id'])
            assert stored.kind=='narrative_time_scope'
            assert receipt['record_sha256']==_digest({'kind':stored.kind,'attrs':stored.attrs,'evidence':stored.evidence})
            assert [len(m['matched_record_ids']) for m in receipt['members']]==[1,1,0]
            assert all(session.get(AnalysisRecordRow,rid).run_id==run_id for member in receipt['members'] for rid in member['matched_record_ids'])
        assert len(sent)==1
        issues=client.get(f'/api/v1/analysis-runs/{run_id}/issues').json()
        issue,=[row for row in issues if row['category']=='location_collision']
        assert any(e['text']==BODY.splitlines()[0] for e in issue['evidence'])
        for index,evidence in enumerate(issue['evidence']):
            preview=client.get(f'/api/v1/analysis-runs/{run_id}/evidence-preview',params={'kind':'issue','item_id':issue['id'],'evidence_index':index})
            assert preview.status_code==200
            assert '\n'.join(line['text'] for line in preview.json()['lines'] if line['is_evidence'])==evidence['text']
        for suffix in ('records','diagnostics','export.md'):
            response=client.get(f'/api/v1/analysis-runs/{run_id}/{suffix}')
            assert response.status_code==200
            assert TIME_SCOPE_REFS_KEY not in response.text and 'matched_record_ids' not in response.text and 'event_ref' not in response.text
        with patch('app.main.dispatch_analysis'):
            old_id=client.post(runs,json={}).json()['id']
            with SessionLocal() as session:
                old=session.get(AnalysisRunRow,old_id)
                old.batch_coverage={k:v for k,v in old.batch_coverage.items() if k!='narrative_time_scope_policy'}
                old.status='failed'
                session.commit()
            retry=client.post(f'/api/v1/analysis-runs/{old_id}/retry').json()['id']
            with SessionLocal() as session: assert 'narrative_time_scope_policy' not in session.get(AnalysisRunRow,retry).batch_coverage
            with patch.object(OpenAICompatibleProvider,'__init__',provider_init): execute_analysis(retry,raise_on_failure=True)
            assert not sent[-1].endswith(NARRATIVE_TIME_SCOPE_INSTRUCTIONS)
            assert client.post(documents,json={'name':'draft.md','content':BODY.replace('南门','北门'),'replace_document_id':doc['id']}).status_code==201
            revised=client.post(f'/api/v1/analysis-runs/{run_id}/rechecks').json()['id']
            with SessionLocal() as session: assert session.get(AnalysisRunRow,revised).batch_coverage['narrative_time_scope_policy']==NARRATIVE_TIME_SCOPE_POLICY_V1
        with TestClient(app) as other:
            register(other)
            assert other.get(f'/api/v1/analysis-runs/{run_id}/records').status_code==404


def test_scope_projection_does_not_make_calendar_time_or_graph_edges():
    row=SimpleNamespace(id='scope',kind='narrative_time_scope',attrs={'relation':'same_day','member_count':'3'},
        evidence={'document_id':'d','document_name':'draft.md','line_start':1,'line_end':1,'text':BODY.splitlines()[0]})
    timeline=project_timeline('run',[row],[])
    assert not timeline.groups and len(timeline.unscheduled)==1
    assert timeline.unscheduled[0].timestamp is None and '时间范围' in timeline.unscheduled[0].title
    assert not project_graph('run',[row],[]).edges


def test_bad_scope_proof_cannot_leave_a_formal_report():
    from app.domain import ConsistencyIssue
    from app.pipeline import PipelineResult
    from app.main import write_limiter
    from tests.test_character_observation_pipeline import _new_review
    from tests.test_character_consistency_stage import _context,_create_document
    write_limiter.events.clear()
    with TestClient(app) as client:
        project=client.post('/api/v1/projects',json={'name':'Damaged time scope'}).json()
        document=_create_document(client,project['id'],name='draft.md',role='chapter',content=BODY,narrative_context=_context(publication='draft'))
        run_id=_new_review(client,project['id'],document['id'])
        evidence=EvidenceSpan(document_id=document['id'],document_name='draft.md',line_start=1,line_end=1,text=BODY.splitlines()[0])
        directive=ParsedDirective(kind='narrative_time_scope',attrs={},evidence=evidence,narrative_time_scope={'raw':'must-not-persist'})
        issue=ConsistencyIssue(category='location_collision',severity='high',confidence=.9,title='Damaged source',explanation='Must not persist.',suggestion='核对',evidence=[evidence])
        result=PipelineResult(directives=[directive],issues=[issue])
        runtime=SimpleNamespace(settings=Settings(_env_file=None,enable_model_extraction=False,enable_character_consistency=False),identity='server_default',available=True,unavailable_reason=None)
        with patch('app.service.runtime_for_analysis_run',return_value=runtime),patch('app.service.AnalysisPipeline',new=lambda:SimpleNamespace(run=lambda *a,**kw:result)):
            execute_analysis(run_id)
        with SessionLocal() as session:
            assert session.get(AnalysisRunRow,run_id).status=='failed'
            assert not session.scalars(select(IssueRow).where(IssueRow.run_id==run_id)).all()
            assert not session.scalars(select(AnalysisRecordRow).where(AnalysisRecordRow.run_id==run_id)).all()
