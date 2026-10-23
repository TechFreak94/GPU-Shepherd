"""Host CPU and memory figures read from a fake /proc."""
import gpushepherd.host as host

STAT = "cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0 0 0\n"
MEMINFO = """MemTotal:       131072000 kB
MemFree:         1000000 kB
MemAvailable:   98304000 kB
SwapTotal:       8388608 kB
SwapFree:        8388608 kB
"""
CPUINFO = """processor\t: 0
model name\t: Test CPU   9000
physical id\t: 0
core id\t: 0

processor\t: 1
model name\t: Test CPU   9000
physical id\t: 0
core id\t: 1

processor\t: 2
model name\t: Test CPU   9000
physical id\t: 0
core id\t: 0
"""


def fake_proc(tmp_path, monkeypatch, busy, idle):
    (tmp_path / "stat").write_text(STAT.format(busy=busy, idle=idle))
    (tmp_path / "meminfo").write_text(MEMINFO)
    (tmp_path / "cpuinfo").write_text(CPUINFO)
    (tmp_path / "loadavg").write_text("1.50 0.75 0.25 2/300 1234\n")
    monkeypatch.setattr(host, "PROC", str(tmp_path))


def test_host_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_cpu_info", None)
    monkeypatch.setattr(host, "_prev", None)
    fake_proc(tmp_path, monkeypatch, busy=100, idle=900)
    h = host.query()
    assert h.cpu_model == "Test CPU 9000" and h.cpu_cores == 2 and h.cpu_threads == 3 and h.load == (1.5, 0.75, 0.25)
    assert h.mem_total == 131072000 * 1024 and h.mem_used == (131072000 - 98304000) * 1024 and h.mem_pct == 25
    assert h.swap_total == 8388608 * 1024 and h.swap_used == 0 and not h.errors

    fake_proc(tmp_path, monkeypatch, busy=400, idle=1600)     # +300 busy of +1000 since the last call
    assert host.query().cpu_pct == 30.0


def test_missing_proc_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "_cpu_info", None)
    monkeypatch.setattr(host, "_prev", None)
    monkeypatch.setattr(host, "PROC", str(tmp_path / "nope"))
    h = host.query()
    assert h.mem_total == 0 and len(h.errors) == 2
