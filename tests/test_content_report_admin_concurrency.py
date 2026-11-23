from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

from sqlalchemy import event
from sqlmodel import Session

from cowork import content_report_admin
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.content_report import ContentReport


def test_only_one_concurrent_disposition_is_accepted():
    engine = get_engine(get_app_settings().database.uri)
    report_id = uuid4()
    with Session(engine) as db:
        db.add(ContentReport(id=report_id, org_id=str(uuid4()), user_id=str(uuid4()),
                             target_key='test', evidence='test'))
        db.commit()

    rendezvous = Barrier(2, timeout=5)

    def before_write(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('UPDATE CONTENT_REPORTS'):
            rendezvous.wait()

    def resolve(reviewer):
        try:
            content_report_admin.main(['resolve', str(report_id), '--reviewer', reviewer,
                                       '--resolution', f'Reviewed by {reviewer}'])
        except SystemExit as error:
            assert error.code == 2
            return None
        return reviewer

    event.listen(engine, 'before_cursor_execute', before_write)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(resolve, ['Alice', 'Bob']))
    finally:
        event.remove(engine, 'before_cursor_execute', before_write)

    winner, = [name for name in results if name is not None]
    with Session(engine) as db:
        row = db.get(ContentReport, report_id)
        assert (row.status, row.reviewed_by, row.resolution) == ('resolved', winner, f'Reviewed by {winner}')
