from gpushepherd.config import GpuPolicy, Instance
from gpushepherd.scheduler import gpu_text, plan


def inst(iid, mode="none", gpus=(), min_gpus=1, max_gpus=0, priority=50):
    pol = GpuPolicy(mode=mode, gpus=list(gpus), min_gpus=min_gpus, max_gpus=max_gpus, priority=priority)
    return Instance(id=iid, name=iid.upper(), image="img:1", gpu=pol)


def stack():
    """A typical two-GPU setup: ComfyUI pinned to GPU 0, an elastic LLM server, a CPU-only agent."""
    return {
        "comfyui": inst("comfyui", "fixed", [0], priority=80),
        "llm": inst("llm", "elastic", min_gpus=1, priority=50),
        "hermes": inst("hermes"),
    }


TWO = [0, 1]


def test_llm_alone_gets_both_gpus():
    p = plan(TWO, stack(), {}, "start", "llm")
    assert p.allocations == {"llm": [0, 1]}
    assert p.to_start == ["llm"] and not p.side_effects


def test_starting_comfyui_moves_llm_to_gpu1():
    p = plan(TWO, stack(), {"llm": [0, 1]}, "start", "comfyui")
    assert p.allocations == {"comfyui": [0], "llm": [1]}
    assert p.to_start == ["comfyui"]
    assert p.to_restart == ["llm"]
    assert p.side_effects == ["llm"]
    assert "Restart LLM to move it from GPUs 0,1 to GPU 1" in p.describe({"llm": "LLM"})


def test_stopping_comfyui_gives_llm_everything():
    p = plan(TWO, stack(), {"comfyui": [0], "llm": [1]}, "stop", "comfyui")
    assert p.allocations == {"llm": [0, 1]}
    assert p.to_stop == ["comfyui"] and p.to_restart == ["llm"] and not p.preempted


def test_starting_llm_while_comfyui_runs_avoids_gpu0():
    p = plan(TWO, stack(), {"comfyui": [0]}, "start", "llm")
    assert p.allocations == {"comfyui": [0], "llm": [1]}
    assert not p.side_effects


def test_cpu_only_instance_never_disturbs_anyone():
    p = plan(TWO, stack(), {"comfyui": [0], "llm": [1]}, "start", "hermes")
    assert p.allocations["hermes"] == [] and not p.side_effects and p.to_start == ["hermes"]


def test_single_gpu_take_turns():
    one = [0]
    p = plan(one, stack(), {"comfyui": [0]}, "start", "llm")
    assert p.allocations == {"llm": [0]} and p.preempted == ["comfyui"]
    p = plan(one, stack(), {"llm": [0]}, "start", "comfyui")
    assert p.allocations == {"comfyui": [0]} and p.preempted == ["llm"]


def test_restart_same_gpus_is_explicit_restart():
    p = plan(TWO, stack(), {"comfyui": [0], "llm": [1]}, "restart", "llm")
    assert p.to_restart == ["llm"] and not p.side_effects


def test_already_running_start_is_noop():
    p = plan(TWO, stack(), {"comfyui": [0], "llm": [1]}, "start", "comfyui")
    assert p.is_noop


def test_fixed_gpu_missing_on_machine_is_an_error():
    s = stack()
    s["big"] = inst("big", "fixed", [3])
    p = plan(TWO, s, {}, "start", "big")
    assert "does not have" in p.error


def test_elastic_minimum_too_large_is_an_error():
    s = {"huge": inst("huge", "elastic", min_gpus=4)}
    p = plan(TWO, s, {}, "start", "huge")
    assert "needs at least 4" in p.error


def test_priority_decides_who_is_stopped():
    s = {
        "a": inst("a", "fixed", [0], priority=90),
        "b": inst("b", "fixed", [1], priority=10),
        "c": inst("c", "elastic", min_gpus=2, priority=50),
    }
    # starting c needs both GPUs: the target wins, both a and b are preempted
    p = plan(TWO, s, {"a": [0], "b": [1]}, "start", "c")
    assert p.allocations == {"c": [0, 1]} and p.preempted == ["a", "b"]


def test_elastic_max_and_leftovers_by_priority():
    s = {
        "hi": inst("hi", "elastic", min_gpus=1, max_gpus=2, priority=80),
        "lo": inst("lo", "elastic", min_gpus=1, priority=20),
    }
    p = plan([0, 1, 2, 3], s, {}, "rebalance", extra_wanted={"hi", "lo"})
    assert p.allocations == {"hi": [0, 1], "lo": [2, 3]}


def test_elastic_preference_order():
    s = {"x": inst("x", "elastic", gpus=[1], min_gpus=1, max_gpus=1)}
    p = plan(TWO, s, {}, "start", "x")
    assert p.allocations == {"x": [1]}


def test_two_fixed_on_same_gpu_lower_priority_loses():
    s = {"a": inst("a", "fixed", [0], priority=60), "b": inst("b", "fixed", [0], priority=40)}
    p = plan(TWO, s, {"b": [0]}, "start", "a")
    assert p.allocations == {"a": [0]} and p.preempted == ["b"]


def test_autostart_rebalance_respects_priorities():
    p = plan(TWO, stack(), {}, "rebalance", extra_wanted={"comfyui", "llm", "hermes"})
    assert p.allocations == {"comfyui": [0], "llm": [1], "hermes": []}
    assert p.to_start[0] == "comfyui"  # highest priority first


def test_unknown_target():
    assert plan(TWO, stack(), {}, "start", "nope").error


def test_running_instances_keep_their_gpus():
    s = {
        "hi": inst("hi", "elastic", min_gpus=1, max_gpus=1, priority=80),
        "lo": inst("lo", "elastic", min_gpus=1, max_gpus=1, priority=20),
    }
    # lo already runs on GPU 0; starting hi must not move lo
    p = plan(TWO, s, {"lo": [0]}, "start", "hi")
    assert p.allocations == {"hi": [1], "lo": [0]} and not p.side_effects


def test_fixed_is_kept_when_elastic_target_can_fit_elsewhere():
    s = {"pin": inst("pin", "fixed", [1], priority=10), "el": inst("el", "elastic", min_gpus=1)}
    p = plan(TWO, s, {"pin": [1]}, "start", "el")
    assert p.allocations == {"pin": [1], "el": [0]} and not p.preempted


def test_second_comfyui_takes_gpu1_from_llm():
    """The example's ComfyUI (GPU 1): starting it stops the LLM, which outranks it afterwards."""
    s = stack() | {"comfyui-2": inst("comfyui-2", "fixed", [1], priority=40)}
    p = plan(TWO, s, {"comfyui": [0], "llm": [1]}, "start", "comfyui-2")
    assert p.allocations == {"comfyui": [0], "comfyui-2": [1]}
    assert p.preempted == ["llm"] and p.side_effects == ["llm"]
    p = plan(TWO, s, {"comfyui": [0], "comfyui-2": [1]}, "start", "llm")
    assert p.allocations == {"comfyui": [0], "llm": [1]}
    assert p.preempted == ["comfyui-2"]


def test_starting_an_app_starts_what_it_requires_first():
    s = stack() | {"db": inst("db"), "chat": inst("chat").model_copy(update={"requires": ["db"]})}
    p = plan(TWO, s, {}, "start", "chat")
    assert p.to_start == ["db", "chat"] and p.allocations == {"db": [], "chat": []} and not p.side_effects
    p = plan(TWO, s, {"db": []}, "start", "chat")          # already running: left alone
    assert p.to_start == ["chat"] and not p.to_restart
    assert plan(TWO, s, {"db": [], "chat": []}, "stop", "chat").to_stop == ["chat", "db"]   # app first, then its db
    shared = s | {"chat2": inst("chat2").model_copy(update={"requires": ["db"]})}
    p = plan(TWO, shared, {"db": [], "chat": [], "chat2": []}, "stop", "chat")
    assert p.to_stop == ["chat"] and not p.side_effects          # chat2 still needs it
    auto = plan(TWO, s, {}, "rebalance", None, {"chat"})
    assert auto.to_start.index("db") < auto.to_start.index("chat")


def test_missing_requirement_is_an_error():
    s = stack() | {"chat": inst("chat").model_copy(update={"requires": ["gone"]})}
    assert "needs instance 'gone'" in plan(TWO, s, {}, "start", "chat").error


def test_requirement_cycle_does_not_hang():
    s = {"a": inst("a").model_copy(update={"requires": ["b"]}), "b": inst("b").model_copy(update={"requires": ["a"]})}
    assert sorted(plan(TWO, s, {}, "start", "a").to_start) == ["a", "b"]


EIGHT = list(range(8))


def test_eight_gpus_shared_by_pinned_capped_and_open_ended_instances():
    s = {
        "comfy-a": inst("comfy-a", "fixed", [0], priority=80),
        "comfy-b": inst("comfy-b", "fixed", [7], priority=80),
        "coder": inst("coder", "elastic", min_gpus=2, max_gpus=3, priority=60),
        "llm": inst("llm", "elastic", min_gpus=1, priority=50),
    }
    p = plan(EIGHT, s, {}, "rebalance", extra_wanted=set(s))
    assert p.allocations == {"comfy-a": [0], "comfy-b": [7], "coder": [1, 2, 3], "llm": [4, 5, 6]}
    # a big server that needs most of the machine pushes the open-ended one down to its minimum
    s["big"] = inst("big", "elastic", min_gpus=4, max_gpus=4, priority=90)
    p = plan(EIGHT, s, p.allocations, "start", "big")
    assert len(p.allocations["big"]) == 4 and p.allocations["coder"] == [1, 2]
    assert "llm" in p.preempted          # 2 pinned + 4 + 2 = all eight
    assert sorted(g for gs in p.allocations.values() for g in gs) == EIGHT


def test_elastic_instances_get_neighbouring_gpus():
    s = {"pin": inst("pin", "fixed", [2], priority=80), "x": inst("x", "elastic", min_gpus=3, max_gpus=3)}
    p = plan([0, 1, 2, 3, 4, 5], s, {}, "rebalance", extra_wanted=set(s))
    assert p.allocations["x"] == [0, 1, 3]
    p = plan([0, 1, 2, 3, 4, 5], s, {"x": [3, 4]}, "rebalance", extra_wanted=set(s))
    assert p.allocations["x"] == [3, 4, 5]                     # keeps its GPUs, grows next to them


def test_gpu_text_shortens_runs():
    assert gpu_text([]) == "no GPU" and gpu_text([3]) == "GPU 3"
    assert gpu_text([0, 1]) == "GPUs 0,1"
    assert gpu_text(EIGHT) == "GPUs 0-7"
    assert gpu_text([6, 0, 1, 2, 4]) == "GPUs 0-2,4,6"
