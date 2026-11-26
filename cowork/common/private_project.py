"""Protect the desktop's .anton before saving project data outside its profile."""

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from cowork.common.paths import PinnedDir, dir_mkdir, open_pinned_child


@contextmanager
def private_project_folder(
    project: PinnedDir, *, memory: bool = False
) -> Iterator[PinnedDir]:
    """Keep .anton (and, for a memory write, its memory folder) pinned through the write."""
    with ExitStack() as folders:
        # Windows PinnedDir ordinarily holds only a path. Deny rename on the
        # project and each selected child while a private write uses that path.
        if os.name == "nt":
            if project.path.parent != project.path:
                folders.enter_context(_windows_directory(project.path.parent))
            folders.enter_context(_windows_directory(project.path))
        folder = folders.enter_context(_private_child(project, ".anton"))
        if memory:
            folder = folders.enter_context(_private_child(folder, "memory"))
        yield folder


@contextmanager
def _private_child(parent: PinnedDir, name: str) -> Iterator[PinnedDir]:
    try:
        dir_mkdir(parent, name, mode=0o700)
    except FileExistsError:
        pass
    if os.name == "nt":
        path = parent.path / name
        with _windows_directory(path, private=True):
            yield PinnedDir(None, path)
        return

    child = open_pinned_child(parent, name)
    try:
        st = os.fstat(child.fd)
        if st.st_uid != os.geteuid():
            raise PermissionError("Project data directory belongs to another user")
        if stat.S_IMODE(st.st_mode) != 0o700:
            os.fchmod(child.fd, 0o700)
            if stat.S_IMODE(os.fstat(child.fd).st_mode) != 0o700:
                raise PermissionError("Project data directory cannot be made private")
        yield child
    finally:
        child.close()


@contextmanager
def _windows_directory(path: Path, *, private: bool = False) -> Iterator[None]:
    import ntsecuritycon
    import win32con
    import win32file

    access = ntsecuritycon.FILE_READ_ATTRIBUTES
    if private:
        access |= win32con.READ_CONTROL | win32con.WRITE_DAC
    handle = win32file.CreateFile(
        str(path),
        access,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        win32con.FILE_FLAG_BACKUP_SEMANTICS | win32file.FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    try:
        attrs = win32file.GetFileInformationByHandle(handle)[0]
        if not attrs & stat.FILE_ATTRIBUTE_DIRECTORY:
            raise NotADirectoryError("Expected a project directory")
        if (
            attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT
            and path.lstat().st_reparse_tag & 0x20000000
        ):
            raise NotADirectoryError("Expected a project directory without redirection")
        if private:
            _protect_windows_directory(handle)
        yield
    finally:
        handle.Close()


def _protect_windows_directory(handle) -> None:
    """Remove inherited access by handle, preserving explicit administrator grants."""
    import ntsecuritycon
    import win32api
    import win32con
    import win32security as security

    descriptor = security.GetSecurityInfo(
        handle,
        security.SE_FILE_OBJECT,
        security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION,
    )
    token = security.OpenProcessToken(
        win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
    )
    try:
        user = security.GetTokenInformation(token, security.TokenUser)[0]
        default_owner = security.GetTokenInformation(token, security.TokenOwner)
    finally:
        token.Close()
    if descriptor.GetSecurityDescriptorOwner() not in (user, default_owner):
        raise PermissionError("Project data directory belongs to another user")

    dacl = descriptor.GetSecurityDescriptorDacl()
    protected = (
        descriptor.GetSecurityDescriptorControl()[0] & security.SE_DACL_PROTECTED
    )
    if dacl is None:
        dacl = security.ACL()
    inherit = win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE
    full_control = ntsecuritycon.FILE_ALL_ACCESS
    granted = []
    changed = not protected
    for i in reversed(range(dacl.GetAceCount())):
        ace = dacl.GetAce(i)
        kind, flags = ace[0]
        if flags & security.INHERITED_ACE:
            dacl.DeleteAce(i)
            changed = True
        elif (
            kind == security.ACCESS_ALLOWED_ACE_TYPE
            and flags == inherit
            and ace[1] == full_control
        ):
            granted.append(ace[2])
    for sid in (
        user,
        security.CreateWellKnownSid(security.WinLocalSystemSid),
        security.CreateWellKnownSid(security.WinBuiltinAdministratorsSid),
    ):
        if sid not in granted:
            dacl.AddAccessAllowedAceEx(
                security.ACL_REVISION_DS, inherit, full_control, sid
            )
            changed = True
    if not changed:
        return
    security.SetSecurityInfo(
        handle,
        security.SE_FILE_OBJECT,
        security.DACL_SECURITY_INFORMATION
        | security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        dacl,
        None,
    )
    actual = security.GetSecurityInfo(
        handle, security.SE_FILE_OBJECT, security.DACL_SECURITY_INFORMATION
    )
    if (
        not (actual.GetSecurityDescriptorControl()[0] & security.SE_DACL_PROTECTED)
        or actual.GetSecurityDescriptorDacl() is None
    ):
        raise PermissionError("Project data directory cannot be made private")
