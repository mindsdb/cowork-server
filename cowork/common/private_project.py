"""Protect the desktop's .anton before saving project data outside its profile.

A filesystem that keeps no modes, such as WSL's ``/mnt/c``, exFAT or an SMB
share, or a Windows drive with no access lists, such as FAT or exFAT, leaves
the folder as it is, with a warning: no mode protects a folder there, and
refusing would stop the user without stopping any reader. A folder another
account owns is still refused.
"""

import errno
import logging
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from cowork.common.paths import PinnedDir, dir_mkdir, open_pinned_child

logger = logging.getLogger(__name__)

#: GetVolumeInformation's flag for a volume that stores access lists. NTFS and
#: ReFS set it; FAT, FAT32 and exFAT don't.
_FILE_PERSISTENT_ACLS = 0x00000008

_WARNED_MODES_NOT_KEPT: set[str] = set()


def _warn_modes_not_kept(path: Path, reason: str) -> None:
    """Warn once per folder that its filesystem keeps no private permissions."""
    key = os.path.normcase(os.path.abspath(path))
    if key in _WARNED_MODES_NOT_KEPT:
        return
    _WARNED_MODES_NOT_KEPT.add(key)
    logger.warning(
        "The filesystem keeps no private permissions for a project's %s folder (%s); it is used as it is",
        path.name,
        reason,
    )


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
            try:
                os.fchmod(child.fd, 0o700)
            except OSError as exc:
                _warn_modes_not_kept(
                    parent.path / name, errno.errorcode.get(exc.errno, "unknown")
                )
            else:
                if stat.S_IMODE(os.fstat(child.fd).st_mode) != 0o700:
                    _warn_modes_not_kept(parent.path / name, "mode_not_kept")
        yield child
    finally:
        child.close()


@contextmanager
def _windows_directory(path: Path, *, private: bool = False) -> Iterator[None]:
    import ntsecuritycon
    import win32con
    import win32file

    if private and not _volume_keeps_acls(path):
        _warn_modes_not_kept(path, "volume_keeps_no_acls")
        private = False
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


def _volume_keeps_acls(path: Path) -> bool:
    """Whether the volume holding ``path`` stores access lists; True when Windows can't say."""
    import pywintypes
    import win32api
    import win32file

    try:
        root = win32file.GetVolumePathName(str(path))
        flags = win32api.GetVolumeInformation(root)[3]
    except pywintypes.error:
        return True
    return bool(flags & _FILE_PERSISTENT_ACLS)


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
