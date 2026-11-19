"""Operator-only moderation queue. Run in the server environment; no HTTP surface.

python -m cowork.content_report_admin list
python -m cowork.content_report_admin show REPORT_UUID
python -m cowork.content_report_admin resolve REPORT_UUID --reviewer NAME --resolution TEXT

Resolve only after investigation and any removal/restriction have actually been
completed. This command records a disposition; it does not remove content.
"""
import argparse
import json
from uuid import UUID
from sqlalchemy import update
from cowork.db.scoped import SYSTEM_SCOPE, unsafe_unscoped_session
from cowork.db.units import unit_session
from cowork.models.content_report import ContentReport


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    show = commands.add_parser('show')
    show.add_argument('id', type=UUID)
    resolve = commands.add_parser('resolve')
    resolve.add_argument('id', type=UUID)
    resolve.add_argument('--reviewer', required=True)
    resolve.add_argument('--resolution', required=True)
    args = parser.parse_args(argv)
    # Deliberate operator view, only through deployment/DB access. Never
    # reachable from a client principal or a public moderation endpoint.
    with unit_session(scope=SYSTEM_SCOPE) as session:
        if args.command == 'list':
            reports = session.exec(session.select(ContentReport).where(
                ContentReport.status == 'pending').order_by(ContentReport.created_at).limit(100)).all()
            print(json.dumps([{'id': str(row.id), 'created_at': str(row.created_at)} for row in reports]))
            return
        if args.command == 'show':
            report = session.get(ContentReport, args.id)
            if report is None:
                parser.error('Report not found')
            print(report.model_dump_json())
            return
        if not args.reviewer.strip() or len(args.reviewer) > 128:
            parser.error('Reviewer must contain 1–128 characters')
        if not args.resolution.strip() or len(args.resolution) > 2000:
            parser.error('Resolution must contain 1–2000 characters')
        result = unsafe_unscoped_session(session).execute(
            update(ContentReport)
            .where(ContentReport.id == args.id, ContentReport.status == 'pending')
            .values(status='resolved', reviewed_by=args.reviewer.strip(), resolution=args.resolution.strip())
        )
        if result.rowcount == 0:
            if session.get(ContentReport, args.id) is None:
                parser.error('Report not found')
            parser.error('Report already resolved; do not overwrite its disposition')
    print(json.dumps({'id': str(args.id), 'status': 'resolved'}))

if __name__ == '__main__':
    main()
