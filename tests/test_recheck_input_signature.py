"""Prepare/submit optimistic identity is checked inside the snapshot transaction."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app import main
from app.db import AnalysisRunInputRow, AnalysisRunRow, DocumentRow, SessionLocal
from tests.test_material_change_impact import clean_limit, setup_run, impact
from tests.test_review_batch import _document


def prepared(client):
    project, background, old, run = setup_run(client, draft=True)
    saved = _document(client, project['id'], name=old['name'], role='chapter', publication='draft',
                      content='已保存的第二版。', replace_document_id=old['id'])
    view = impact(client, run)
    assert view['recheck']['can_prepare']
    signature = view['recheck']['input_signature']
    assert len(signature) == 64
    return project, background, old, saved, run, signature


def run_count(project_id):
    with SessionLocal() as db:
        return db.scalar(select(func.count()).select_from(AnalysisRunRow).where(AnalysisRunRow.project_id == project_id))


@pytest.mark.parametrize('change', ['version', 'identity', 'background'])
def test_changed_selection_rejected_before_provider_run_or_dispatch(change):
    with TestClient(main.app) as client:
        project, _, _, saved, run, signature = prepared(client)
        if change == 'version':
            _document(client, project['id'], name=saved['name'], role='chapter', publication='draft',
                      content='另一位编辑保存的第三版。', replace_document_id=saved['id'])
        elif change == 'identity':
            response = client.post(f"/api/v1/projects/{project['id']}/documents/{saved['id']}/narrative-context/revisions",
                json=dict(expected_revision=1, resolution_state='confirmed', publication_status='draft',
                          scope={'timeline_key': '另一时间线'}))
            assert response.status_code == 201, response.text
        else:
            _document(client, project['id'], name='新增设定.md', role='canon', publication='published')
        before = run_count(project['id'])
        with (patch('app.main.account_provider_runtime', side_effect=AssertionError('must reject before provider setup')),
              patch('app.main.dispatch_analysis') as dispatch):
            response = client.post(f'/api/v1/analysis-runs/{run}/rechecks', json={'expected_input_signature': signature})
        assert response.status_code == 409, response.text
        assert response.json()['detail']['code'] == 'recheck_inputs_changed'
        assert run_count(project['id']) == before and not dispatch.called


def test_valid_signature_freezes_exact_selected_ids_and_ignores_unselected_draft():
    with TestClient(main.app) as client:
        project, background, _, saved, run, signature = prepared(client)
        _document(client, project['id'], name='未纳入的另一章.md', role='chapter', publication='draft')
        assert impact(client, run)['recheck']['input_signature'] == signature
        with patch('app.main.dispatch_analysis') as dispatch:
            response = client.post(f'/api/v1/analysis-runs/{run}/rechecks', json={'expected_input_signature': signature})
        assert response.status_code == 202, response.text
        assert dispatch.call_count == 1
        with SessionLocal() as db:
            created = db.get(AnalysisRunRow, response.json()['id'])
            assert created.batch_coverage['recheck_expected_input_signature'] == signature
            frozen = db.scalars(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == created.id)).all()
            assert {row.document_id for row in frozen} == {background['id'], saved['id']}
            assert next(row for row in frozen if row.document_id == saved['id']).document_version == 2


def test_idempotency_replays_original_expectation_even_after_new_revision_but_rejects_changed_intent():
    with TestClient(main.app) as client:
        project, _, _, saved, run, signature = prepared(client)
        url = f'/api/v1/analysis-runs/{run}/rechecks'
        headers = {'Idempotency-Key': 'fixed-reviewed-input'}
        with patch('app.main.dispatch_analysis') as dispatch:
            first = client.post(url, headers=headers, json={'expected_input_signature': signature})
        assert first.status_code == 202 and dispatch.call_count == 1
        _document(client, project['id'], name=saved['name'], role='chapter', publication='draft', replace_document_id=saved['id'])
        next_signature = impact(client, run)['recheck']['input_signature']
        assert next_signature != signature
        with (patch('app.main.account_provider_runtime', side_effect=AssertionError('replay must not start a new run')),
              patch('app.main.dispatch_analysis') as dispatch):
            replay = client.post(url, headers=headers, json={'expected_input_signature': signature})
            changed = client.post(url, headers=headers, json={'expected_input_signature': next_signature})
            omitted = client.post(url, headers=headers)
        assert replay.status_code == 202 and replay.json()['id'] == first.json()['id'] and replay.json()['deduplicated']
        for response in (changed, omitted):
            assert response.status_code == 409 and response.json()['detail']['code'] == 'idempotency_key_conflict'
        assert not dispatch.called


def test_signature_is_bound_to_the_exact_baseline_and_missing_sources_offer_none():
    with TestClient(main.app) as client:
        project, _, _, _, run, signature = prepared(client)
        with SessionLocal() as db:
            original = db.get(AnalysisRunRow, run)
            other = AnalysisRunRow(project_id=project['id'], status='completed', batch_mode=original.batch_mode,
                                   batch_coverage=original.batch_coverage)
            db.add(other); db.flush()
            original_id, other_id = original.id, other.id
            # The same selected bodies with a different baseline must have a
            # different preparation identity; no alternate report is inferred.
            _, documents, roles, _ = main._recheck_material_selection(db, original, {})
            source = main.current_project_source_signature(db, project['id'], documents, batch_roles=roles)
            assert main._recheck_input_signature(other, documents, source) != signature
            db.commit()
        assert original_id != other_id
        broken = impact(client, other_id)
        assert broken['recheck']['input_signature'] is None and not broken['recheck']['can_prepare']


@pytest.mark.parametrize('value', ['not-a-digest', 'a' * 63, 'A' * 64, 123])
def test_expected_signature_has_a_bounded_exact_format(value):
    with TestClient(main.app) as client, patch('app.main.dispatch_analysis') as dispatch:
        response = client.post('/api/v1/analysis-runs/no-run/rechecks', json={'expected_input_signature': value})
        assert response.status_code == 422 and not dispatch.called


def test_sqlite_selection_check_and_capture_share_the_upload_write_boundary():
    captured, release, upload_trying = Event(), Event(), Event()
    original_capture = main.capture_run_inputs
    original_begin = main._begin_project_write_transaction

    def paused_capture(*args, **kwargs):
        captured.set()
        assert release.wait(5), 'timed out holding the recheck transaction'
        return original_capture(*args, **kwargs)

    def observed_begin(db):
        if captured.is_set():
            upload_trying.set()
        return original_begin(db)

    with TestClient(main.app) as client:
        project, _, _, saved, run, signature = prepared(client)
        with (patch('app.main.capture_run_inputs', side_effect=paused_capture),
              patch('app.main._begin_project_write_transaction', side_effect=observed_begin),
              patch('app.main.dispatch_analysis'), ThreadPoolExecutor(max_workers=2) as pool):
            review = pool.submit(client.post, f'/api/v1/analysis-runs/{run}/rechecks',
                                 json={'expected_input_signature': signature})
            try:
                assert captured.wait(5)
                upload = pool.submit(_document, client, project['id'], name=saved['name'], role='chapter',
                                     publication='draft', content='并发保存的第三版。', replace_document_id=saved['id'])
                assert upload_trying.wait(5) and not upload.done()
            finally:
                release.set()
            response, newer = review.result(timeout=5), upload.result(timeout=5)
        assert response.status_code == 202, response.text
        with SessionLocal() as db:
            ids = set(db.scalars(select(AnalysisRunInputRow.document_id).where(AnalysisRunInputRow.run_id == response.json()['id'])))
            assert saved['id'] in ids and newer['id'] not in ids
            assert db.get(DocumentRow, newer['id']).active
