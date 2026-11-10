"""Version impact is a read-only source comparison, not another analysis."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import (AnalysisRunInputRow, AnalysisRunInputContextRow, AnalysisRunInputNarrativeContextRow,
                    AnalysisRunRow, CharacterTraitCandidateRow, DocumentRow, IssueRow, SessionLocal)
from app.main import app, write_limiter
from app.service import capture_run_inputs
from tests.test_review_batch import _document, _project
from tests.test_workspace_isolation import required_auth_settings, register


@pytest.fixture(autouse=True)
def clean_limit():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def setup_run(client, *, draft=False):
    project = _project(client)
    background = _document(client, project['id'], name='setting.md', role='canon', publication='published', content='北门敞开。')
    chapter = _document(client, project['id'], name='chapter.md', role='chapter', publication='draft', content='北门关闭。')
    with patch('app.main.dispatch_analysis'):
        response = client.post(f"/api/v1/projects/{project['id']}/analysis-runs", json=(
            dict(mode='draft_review', target_document_ids=[chapter['id']]) if draft else {}))
    assert response.status_code == 202, response.text
    run = response.json()['id']
    with SessionLocal() as db:
        db.get(AnalysisRunRow, run).status = 'completed'
        for report_class in ('formal', 'review_clue'):
            db.add(IssueRow(run_id=run, report_class=report_class, category='fact_conflict', severity='high',
                confidence=.9, title='核对北门状态', explanation='依据不同', suggestion='请核对', extra={}, evidence=[
                    dict(document_id=background['id'], document_name='setting.md', line_start=1,line_end=1,text='北门敞开。'),
                    dict(document_id=chapter['id'], document_name='chapter.md',line_start=1,line_end=1,text='北门关闭。')]))
        db.commit()
    return project, background, chapter, run


def impact(client, run, **query):
    response = client.get(f'/api/v1/analysis-runs/{run}/material-impact', params=query)
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    return response.json()


@pytest.mark.parametrize('content', ['北门关闭。', '北门敞开。'])
def test_version_impact_cites_only_frozen_evidence_and_does_not_mutate_report(content):
    with TestClient(app) as client:
        project, background, _, run = setup_run(client)
        old_report = client.get(f'/api/v1/analysis-runs/{run}/issues').json()
        assert impact(client, run)['recheck']['reason'] == 'no_document_changes'
        revised = _document(client, project['id'], name='setting.md', role='canon', publication='published',
            content=content, replace_document_id=background['id'])
        with patch('app.main.dispatch_analysis') as dispatch:
            view = impact(client, run, limit=1, issue_limit=1)
        dispatch.assert_not_called()
        assert view['sources']['total'] == 2 and view['sources']['has_more']
        assert view['affected_issues']['total'] == 2 and view['affected_issues']['has_more']
        source = view['sources']['items'][0]
        assert source['frozen']['document_id'] == background['id'] and source['current']['document_id'] == revised['id']
        assert source['content_changed'] is (content != '北门敞开。')
        assert 'version_changed' in source['changes']
        assert view['recheck'] == dict(inputs_changed=True, can_prepare=True, reason=None,
                                      input_signature=view['recheck']['input_signature'])
        assert len(view['recheck']['input_signature']) == 64
        second = impact(client, run, offset=1, issue_offset=1, limit=1, issue_limit=1)
        assert not second['sources']['has_more'] and not second['affected_issues']['has_more']
        assert {view['affected_issues']['items'][0]['report_class'], second['affected_issues']['items'][0]['report_class']} == {'formal','review_clue'}
        assert view['affected_issues']['items'][0]['evidence_indices'] == [0]
        assert client.get(f'/api/v1/analysis-runs/{run}/issues').json() == old_report
        assert not view['semantic_impact_complete'] and not view['confirmed_traits_auto_invalidated']
        with patch('app.main.dispatch_analysis'):
            assert client.post(f'/api/v1/analysis-runs/{run}/rechecks').status_code == 202


def test_draft_recheck_ignores_unselected_new_draft_and_includes_new_authority():
    with TestClient(app) as client:
        project, _, _, run = setup_run(client, draft=True)
        other = _document(client, project['id'], name='new-draft.md',role='chapter',publication='draft')
        view = impact(client, run)
        assert view['recheck']['reason'] == 'no_document_changes'
        assert view['summary']['not_included_sources'] == 1 and not view['summary']['affected_issues']
        assert next(i for i in view['sources']['items'] if i['current']['document_id'] == other['id'])['kind'] == 'not_included'
        assert client.post(f'/api/v1/analysis-runs/{run}/rechecks').json()['detail']['code'] == 'no_document_changes'
        _document(client, project['id'], name='new-canon.md',role='canon',publication='published')
        view = impact(client, run)
        assert view['recheck']['can_prepare'] and view['summary']['added_sources'] == 1
        assert not view['summary']['affected_issues']


@pytest.mark.parametrize('role,publication', [('reference','published'), ('canon','retired')])
def test_same_version_identity_change_matches_actual_recheck_selection(role, publication):
    with TestClient(app) as client:
        project, background, _, run = setup_run(client, draft=True)
        response = client.post(f"/api/v1/projects/{project['id']}/documents/{background['id']}/narrative-context/revisions",
            json=dict(document_role=role,resolution_state='confirmed',publication_status=publication,
                scope={'timeline_key':'main'},expected_revision=1))
        assert response.status_code == 201, response.text
        view = impact(client, run)
        source = view['sources']['items'][0]
        assert source['frozen']['document_version'] == source['current']['document_version'] == 1
        assert not source['content_changed'] and 'identity_changed' in source['changes']
        assert 'selection_changed' in source['changes'] and source['selected_for_recheck'] is False
        assert view['summary']['affected_issues'] == 2 and view['recheck']['can_prepare']
        with patch('app.main.dispatch_analysis'):
            actual = client.post(f'/api/v1/analysis-runs/{run}/rechecks')
        assert actual.status_code == 202, actual.text
        assert actual.json()['review_batch']['background_document_ids'] == []


@pytest.mark.parametrize('problem', ['snapshot_missing', 'snapshot_corrupt', 'input_missing', 'target_missing'])
def test_missing_and_corrupt_material_never_claims_complete_comparison(problem):
    with TestClient(app) as client:
        _, _, chapter, run = setup_run(client, draft=True)
        with SessionLocal() as db:
            rows=db.scalars(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id==run)).all()
            if problem=='snapshot_missing':
                removed=rows
            elif problem=='input_missing': removed=rows[:1]
            elif problem=='snapshot_corrupt': rows[0].content_sha256='0'*64
            else: db.get(DocumentRow,chapter['id']).active=False
            for row in removed if problem in {'snapshot_missing','input_missing'} else []:
                for cls in (AnalysisRunInputContextRow,AnalysisRunInputNarrativeContextRow):
                    context=db.get(cls,row.id)
                    if context is not None: db.delete(context)
                db.flush();db.delete(row)
            db.commit()
        view=impact(client,run)
        assert not view['recheck']['can_prepare']
        expected='logical_target_missing' if problem=='target_missing' else 'snapshot_unavailable'
        assert view['recheck']['reason']==expected
        assert client.post(f'/api/v1/analysis-runs/{run}/rechecks').json()['detail']['code']==expected
        if problem!='target_missing':
            assert view['comparison_status']=='unavailable' and not view['affected_issues']['items']


def test_equal_name_is_not_a_frozen_evidence_reference_and_pagination_is_bounded():
    with TestClient(app) as client:
        project, background, _, run=setup_run(client)
        with SessionLocal() as db:
            row=db.scalar(select(IssueRow).where(IssueRow.run_id==run,IssueRow.report_class=='formal'))
            row.evidence=[{**row.evidence[0], 'document_id':'not-a-frozen-source'}]
            db.commit()
        _document(client,project['id'],name='setting.md',role='canon',publication='published',content='已修订',replace_document_id=background['id'])
        view=impact(client,run)
        assert view['summary']['affected_issues']==1 and view['summary']['unresolved_evidence']==1
        assert client.get(f'/api/v1/analysis-runs/{run}/material-impact?limit=101').status_code==422
        assert impact(client,run,offset=200,issue_offset=200)['sources']['items']==[]


def test_workspace_ownership_is_checked_before_any_impact_projection():
    with required_auth_settings(), TestClient(app) as owner, TestClient(app) as other:
        _,csrf=register(owner,'impact-owner');register(other,'impact-other')
        project=owner.post('/api/v1/projects',headers={'X-CSRF-Token':csrf},json={'name':'private'}).json()
        with SessionLocal() as db:
            document=DocumentRow(project_id=project['id'],name='secret.md',content='私有正文')
            run=AnalysisRunRow(project_id=project['id'],status='completed')
            db.add_all([document,run]);db.flush();capture_run_inputs(db,run,[document]);db.commit();run_id=run.id
        assert owner.get(f'/api/v1/analysis-runs/{run_id}/material-impact').status_code==200
        assert other.get(f'/api/v1/analysis-runs/{run_id}/material-impact').status_code==404
        assert other.get('/api/v1/analysis-runs/unknown/material-impact').status_code==404


def test_confirmed_trait_source_change_does_not_revoke_independent_confirmation():
    from tests.test_character_consistency_stage import _confirmed_melon_project
    with TestClient(app) as client:
        # Reuse the established confirmed-profile fixture, including its
        # explicit release scope, rather than testing extraction here.
        project=_confirmed_melon_project(client)
        content='林澈喜欢蜜瓜。'
        with SessionLocal() as db:
            profile={'id':db.scalar(select(DocumentRow.id).where(DocumentRow.project_id==project['id']))}
            candidate=db.scalar(select(CharacterTraitCandidateRow.id).where(CharacterTraitCandidateRow.project_id==project['id']))
            db.get(DocumentRow,profile['id']).active=False;db.commit()
        _document(client,project['id'],name='chapter.md',role='chapter',publication='draft',content='林澈走进房间。')
        with patch('app.main.dispatch_analysis'):
            run=client.post(f"/api/v1/projects/{project['id']}/analysis-runs").json()['id']
        with SessionLocal() as db:
            db.get(AnalysisRunRow,run).status='completed'
            db.add(IssueRow(run_id=run,report_class='formal',category='character_drift',severity='high',confidence=.9,
                title='已有历史判断',explanation='已有解释',suggestion='核对',extra={},evidence=[dict(
                    document_id=profile['id'],document_name='profile.md',line_start=1,line_end=1,text=content)]))
            db.commit()
        _document(client,project['id'],name='profile.md',role='character_profile',publication='published',content='角色设定已更新。')
        view=impact(client,run)
        source=next(s for s in view['sources']['items'] if s['kind']=='confirmed_source')
        assert source['content_changed'] and source['identity_comparison_available']
        assert view['summary']['affected_issues']==1 and view['confirmed_traits_changed'] is False
        with SessionLocal() as db:
            row=db.get(CharacterTraitCandidateRow,candidate)
            assert row.review_state=='confirmed' and row.value=='林澈喜欢蜜瓜'
