from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlmodel import Session, select

from cowork.common.settings import get_app_settings
from cowork.models.file import File


class FileService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _root_dir(self) -> Path:
        return Path(get_app_settings().file.root_dir)

    def list_files(self, purpose: str | None = None) -> list[File]:
        stmt = select(File)
        if purpose is not None:
            stmt = stmt.where(File.purpose == purpose)
        return list(self.session.exec(stmt).all())

    def get_file(self, file_id: UUID) -> File:
        file = self.session.get(File, file_id)
        if file is None:
            raise ValueError("File not found")
        return file

    async def create_file(self, upload: UploadFile, purpose: str) -> File:
        contents = await upload.read()
        filename = upload.filename or "upload"

        file = File(
            filename=filename,
            content_type=upload.content_type or "application/octet-stream",
            size=len(contents),
            purpose=purpose,
            path="",
        )

        file_dir = self._root_dir() / str(file.id)
        file_dir.mkdir(parents=True)
        dest = file_dir / filename
        dest.write_bytes(contents)
        file.path = str(dest)
        self.session.add(file)
        self.session.commit()
        self.session.refresh(file)
        return file

    def delete_file(self, file_id: UUID) -> bool:
        file = self.session.get(File, file_id)
        if file is None:
            return False
        file_dir = Path(file.path).parent
        self.session.delete(file)
        self.session.commit()
        if file_dir.exists():
            shutil.rmtree(file_dir)
        return True

    def get_file_path(self, file_id: UUID) -> Path:
        file = self.get_file(file_id)
        path = Path(file.path)
        if not path.exists():
            raise ValueError("File content not found on disk")
        return path
