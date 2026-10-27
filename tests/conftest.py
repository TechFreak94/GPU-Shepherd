"""Shared test setup."""
import shutil

import pytest

_real_disk_usage = shutil.disk_usage


@pytest.fixture(autouse=True)
def plenty_of_disk_space(monkeypatch):
    """Report 1 TB free everywhere, so space checks don't depend on the test machine's disk.
    A test that needs a full disk patches disk_usage again."""
    def disk_usage(path):
        _real_disk_usage(path)   # still raises for a missing path
        return shutil._ntuple_diskusage(2 * 10**12, 10**12, 10**12)
    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
