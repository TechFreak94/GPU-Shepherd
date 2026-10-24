"""Waiting for running work before stopping: the busy checks and the wait itself."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gpushepherd import drain
from gpushepherd.config import Catalog, CatalogModel, DrainPolicy, Instance, PortMap
from gpushepherd.jobs import Cancelled, Job


def inst(iid="x", image="example/app:1", kind="container", port=None, **policy) -> Instance:
    ports = [PortMap(host=port, container=port)] if port else []
    return Instance(id=iid, name=iid.upper(), image=image, kind=kind, ports=ports, drain=DrainPolicy(**policy))


def test_auto_check_follows_the_image():
    assert drain.check_for(inst(kind="llama-swap", image="ghcr.io/mostlygeek/llama-swap:cuda")) == "llama-swap"
    assert drain.check_for(inst(image="yanwk/comfyui-boot:cu130-slim-v2")) == "comfyui"
    assert drain.check_for(inst(image="ghcr.io/ggml-org/llama.cpp:server-cuda")) == "llama-server"
    assert drain.check_for(inst(image="ghcr.io/open-webui/open-webui:main")) == "none"
    assert drain.check_for(inst(image="yanwk/comfyui-boot:cu130", check="none")) == "none"


def test_http_check_needs_a_path():
    with pytest.raises(ValueError):
        DrainPolicy(check="http", path="status")


# ----------------------------------------------------------------- busy checks against a fake server

class Server:
    """Answers like llama-swap, llama-server or ComfyUI, from a dict of path -> JSON."""

    def __init__(self):
        self.routes: dict[str, object] = {}
        self.events: list[dict] | None = None      # /api/events messages; None = 404 (old llama-swap)
        routes, me = self.routes, self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/api/events" and me.events is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for msg in me.events:
                        self.wfile.write(f"event:message\ndata:{json.dumps(msg)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.2)
                    return
                if self.path not in routes:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(routes[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("SHEPHERD_CONNECT", "host")
    s = Server()
    yield s
    s.httpd.shutdown()


def test_comfyui_counts_running_and_queued(server):
    i = inst(image="comfyui", port=server.port, check="comfyui")
    server.routes["/queue"] = {"queue_running": [[1, "a"]], "queue_pending": [[2, "b"], [3, "c"]]}
    act = drain.probe(i)
    assert act.busy == 3 and act.detail == "1 prompt running, 2 queued"
    server.routes["/queue"] = {"queue_running": [], "queue_pending": []}
    assert drain.probe(i).busy == 0


def test_llama_server_slots(server):
    i = inst(image="llama.cpp:server", port=server.port, check="llama-server")
    server.routes["/slots"] = [{"id": 0, "is_processing": True}, {"id": 1, "is_processing": False}]
    assert drain.probe(i).busy == 1
    server.routes["/slots"] = [{"id": 0, "state": 0}]               # older llama.cpp
    assert drain.probe(i).busy == 0


def test_llama_swap_in_flight_from_events(server):
    cat = Catalog(instance_id="llm", models=[CatalogModel(id="coder", file="/models/c.gguf")])
    i = inst("llm", kind="llama-swap", image="llama-swap", port=server.port)
    server.events = [{"type": "modelStatus", "data": "[]"},
                     {"type": "inflight", "data": json.dumps({"operation": "snapshot", "requests": [
                         {"id": "1", "model": "coder"}, {"id": "2", "model": "coder"}]})}]
    act = drain.probe(i, cat)
    assert act.busy == 2 and act.detail == "2 requests (coder)"
    server.events = [{"type": "inflight", "data": json.dumps({"operation": "snapshot", "requests": []})}]
    assert drain.probe(i, cat).busy == 0


def test_llama_swap_without_events_asks_the_loaded_models(server):
    cat = Catalog(instance_id="llm", models=[CatalogModel(id="coder", file="/models/c.gguf"),
                                             CatalogModel(id="chat", file="/models/q.gguf")])
    i = inst("llm", kind="llama-swap", image="llama-swap", port=server.port)
    server.routes["/running"] = {"running": [{"model": "coder", "state": "ready"}]}
    server.routes["/upstream/coder/slots"] = [{"is_processing": True}]
    act = drain.probe(i, cat)
    assert act.busy == 1 and "coder" in act.detail


def test_http_check_reads_a_json_field(server):
    i = inst(port=server.port, check="http", path="/status", key="work.jobs")
    server.routes["/status"] = {"work": {"jobs": [1, 2]}}
    assert drain.probe(i).busy == 2
    server.routes["/status"] = {"work": {"jobs": 0}}
    assert drain.probe(i).busy == 0


def test_unreachable_server_is_unknown(server):
    i = inst(image="comfyui", port=server.port, check="comfyui")      # no /queue route: 404
    act = drain.probe(i)
    assert act.busy is None and "comfyui check failed" in act.detail


# ----------------------------------------------------------------- the wait

class Script:
    """Busy counts to return per instance, one per check; the last one repeats."""

    def __init__(self, **counts):
        self.counts = {k: list(v) for k, v in counts.items()}
        self.asked: list[str] = []

    def __call__(self, i: Instance) -> drain.Activity:
        self.asked.append(i.id)
        seq = self.counts[i.id]
        n = seq.pop(0) if len(seq) > 1 else seq[0]
        return drain.Activity(n, f"{n} requests" if n is not None else "no answer")


def run(instances, ask, job=None):
    lines = []
    drain.wait_until_idle({i.id: i for i in instances}, [i.id for i in instances], lines.append, job,
                          ask=ask, interval=0.01)
    return lines


def test_waits_until_idle():
    llm = inst("llm", image="llama-swap", kind="llama-swap")
    ask = Script(llm=[2, 1, 0])
    lines = run([llm], ask)
    assert ask.asked == ["llm"] * 3
    assert lines[0] == "Waiting for LLM to finish: 2 requests" and "LLM finished its work" in lines[-1]


def test_idle_at_once_and_none_checks_do_not_wait():
    ask = Script(a=[0], b=[5])
    lines = run([inst("a", image="comfyui"), inst("b", image="open-webui")], ask)   # b has no check
    assert ask.asked == ["a"] and lines == []


def test_unknown_does_not_block():
    lines = run([inst("a", image="comfyui")], Script(a=[None]))
    assert "Could not tell whether A is busy" in lines[0]


def test_timeout_stops_anyway_or_cancels():
    lines = run([inst("a", image="comfyui", timeout=1)], Script(a=[1]))
    assert "still busy (1 requests) after 0:01; stopping it anyway" in lines[-1]
    with pytest.raises(Cancelled, match="still busy"):
        run([inst("a", image="comfyui", timeout=1, on_timeout="cancel")], Script(a=[1]))


def test_zero_timeout_means_no_wait():
    ask = Script(a=[3])
    run([inst("a", image="comfyui", timeout=0)], ask)
    assert ask.asked == []


def test_dashboard_buttons():
    job = Job(id=1, title="t")
    a = inst("a", image="comfyui")
    t = threading.Thread(target=run, args=([a], Script(a=[1]), job))
    t.start()
    for _ in range(200):
        if job.waiting:
            break
        time.sleep(0.01)
    assert job.waiting["a"]["detail"] == "1 requests" and job.waiting["a"]["then"] == "stop"
    job.skip.set()                                     # "Stop now"
    t.join(2)
    assert not t.is_alive() and job.waiting == {}

    job = Job(id=2, title="t")
    job.call_off()                                     # "Cancel change"
    with pytest.raises(Cancelled):
        run([a], Script(a=[1]), job)
