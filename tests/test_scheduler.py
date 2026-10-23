from gpushepherd.config import GpuPolicy, Instance
from gpushepherd.scheduler import plan


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
