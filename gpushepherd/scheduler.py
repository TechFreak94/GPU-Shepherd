"""GPU planner: decides which instance gets which GPUs.

Pure functions, no Docker - easy to test. Rules:

1. The instance you are starting (the *target*) always gets what it needs first.
2. Everyone else is served by priority (higher first); fixed instances before elastic ones
   at the same priority.
3. Fixed instances need exactly their GPUs. Elastic instances need `min_gpus` and only say
   *how many* GPUs they want, so they move aside instead of pushing a fixed instance out.
4. Leftover GPUs go to elastic instances (highest priority first) up to their `max_gpus`,
   and no more than they can use (a model server whose models need fewer GPUs).
   Running instances keep the GPUs they already have where possible (no needless restarts).
5. A running instance that can no longer get its minimum is stopped ("preempted"), unless it
   may fall back to the CPU (`cpu_fallback`): then it keeps running with no GPU, and moves back
   onto one when a GPU is free again. A running instance whose GPU set changes is restarted.
7. Instances with a video memory budget (`vram_gb`, one-GPU elastic only) share a GPU with each
   other, as many as fit in its memory. Instances without one never share.
8. At boot (`first`), model servers can be put ahead of everything but the target, so chat works
   after a reboot on a machine whose GPUs can't hold every app.
6. Starting an instance also starts the instances it `requires` (an app's database), first.
   Stopping it stops them too, after it, unless something else still running requires them.

The UI shows the plan and asks for confirmation whenever anything other than the target
would be stopped or restarted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import AMD_BASE, VENDORS, Instance, gpu_vendor

ACTIONS = ("start", "stop", "restart", "rebalance")


@dataclass
class Plan:
    action: str
    target: str | None
    allocations: dict[str, list[int]] = field(default_factory=dict)   # instances running afterwards
    previous: dict[str, list[int]] = field(default_factory=dict)      # running before
    to_start: list[str] = field(default_factory=list)
    to_stop: list[str] = field(default_factory=list)
    to_restart: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    on_cpu: list[str] = field(default_factory=list)                   # fell back to the CPU
    error: str = ""

    @property
    def side_effects(self) -> list[str]:
        """Instances other than the target that this plan stops or restarts."""
        others = [i for i in self.preempted + self.to_restart if i != self.target]
        return list(dict.fromkeys(others))

    @property
    def is_noop(self) -> bool:
        return not (self.to_start or self.to_stop or self.to_restart)

    def describe(self, names: dict[str, str] | None = None) -> list[str]:
        names = names or {}

        def n(i: str) -> str:
            return names.get(i, i)

        lines = []
        for i in self.to_stop:
            if i in self.preempted:
                lines.append(f"Stop {n(i)} - it no longer gets the GPUs it needs")
            else:
                lines.append(f"Stop {n(i)}")
        for i in self.to_restart:
            new, old = self.allocations[i], self.previous.get(i, [])
            if new == old:
                lines.append(f"Restart {n(i)}")
            elif i in self.on_cpu:
                lines.append(f"Restart {n(i)} to move it from {gpu_text(old)} to the CPU - no GPU is free "
                             "for it, so it runs slower until one is")
            else:
                lines.append(f"Restart {n(i)} to move it from {gpu_text(old) if old else 'the CPU'} to {gpu_text(new)}")
        for i in self.to_start:
            if i in self.on_cpu:
                lines.append(f"Start {n(i)} on the CPU - no GPU is free for it, so it runs slower until one is")
            else:
                lines.append(f"Start {n(i)}" + (f" on {gpu_text(self.allocations[i])}" if self.allocations[i] else ""))
        return lines


def gpu_text(gpus: list[int]) -> str:
    """"GPU 0", "GPUs 0-3,6", "AMD GPUs 0,1" (AMD numbers count from config.AMD_BASE)."""
    if not gpus:
        return "no GPU"
    nvidia = [g for g in gpus if g < AMD_BASE]
    amd = [g - AMD_BASE for g in gpus if g >= AMD_BASE]
    return " and ".join([("GPU " if len(nvidia) == 1 else "GPUs ") + _runs(nvidia)] * bool(nvidia)
                        + [("AMD GPU " if len(amd) == 1 else "AMD GPUs ") + _runs(amd)] * bool(amd))


def gpu_list(gpus: list[int]) -> str:
    """GPU numbers with runs of three or more shortened, as on a big machine: "0-3,6"; AMD ones
    after them as "amd0,amd1"."""
    nvidia = [g for g in gpus if g < AMD_BASE]
    amd = [g - AMD_BASE for g in gpus if g >= AMD_BASE]
    return ",".join(x for x in (_runs(nvidia), ",".join(f"amd{g}" for g in sorted(amd))) if x)


def _runs(gpus: list[int]) -> str:
    parts, run = [], []
    for g in sorted(gpus) + [None]:
        if run and (g is None or g != run[-1] + 1):
            parts += [f"{run[0]}-{run[-1]}"] if len(run) > 2 else [str(x) for x in run]
            run = []
        if g is not None:
            run.append(g)
    return ",".join(parts)


def plan(gpus: list[int], instances: dict[str, Instance], running: dict[str, list[int]],
         action: str, target: str | None = None, extra_wanted: set[str] | None = None,
         usable: dict[str, int] | None = None, gpu_mem: dict[int, float] | None = None,
         first: set[str] | None = None) -> Plan:
    """Work out what has to happen.

    gpus         GPU indices present on the machine
    instances    all configured instances
    running      instances running right now -> GPUs they currently have
    action       start | stop | restart | rebalance
    target       the instance the action is about (None for rebalance)
    extra_wanted additional instances to bring up (used for autostart on boot)
    usable       instance -> the most GPUs it can put to use (a model server whose models are
                 small); an elastic instance grows no further than that, leaving the rest free
    gpu_mem      GPU index -> its video memory in GB, for instances that share a GPU (vram_gb);
                 without it they each need a GPU of their own
    first        instances served right after the target, ahead of higher priorities (autostart
                 puts model servers here)
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action}")
    previous = {i: sorted(g) for i, g in running.items() if i in instances}
    result = Plan(action=action, target=target, previous=previous)
    if target is not None and target not in instances:
        result.error = f"unknown instance '{target}'"
        return result

    wanted = set(previous)
    if extra_wanted:
        wanted |= {i for i in extra_wanted if i in instances}
    if action in ("start", "restart") and target:
        wanted.add(target)
    elif action == "stop" and target:
        wanted.discard(target)
        wanted -= with_requirements(instances, instances[target].requires)   # put back below if still needed
    missing = {r: i for i in sorted(wanted) for r in instances[i].requires if r not in instances}
    if missing:
        r, i = next(iter(missing.items()))
        result.error = f"{instances[i].name} needs instance '{r}', which does not exist"
        return result
    wanted = with_requirements(instances, wanted)

    # each kind of GPU (NVIDIA, AMD) is planned on its own, for the apps that run on it; apps
    # without a GPU go with the NVIDIA ones (their kind is nvidia by default)
    allocations, dropped, on_cpu = {}, [], []
    for vendor in VENDORS:
        members = {i for i in wanted if instances[i].gpu.vendor == vendor}
        if not members:
            continue
        pool = sorted(g for g in gpus if gpu_vendor(g) == vendor)
        a, d, c, error = _allocate(pool, set(pool), instances, members, target, previous, usable or {},
                                   gpu_mem or {}, first or set())
        if error:
            result.error = error
            return result
        allocations |= a
        dropped += d
        on_cpu += c

    result.allocations = allocations
    result.preempted = sorted(i for i in dropped if i in previous)
    result.on_cpu = sorted(on_cpu)
    # apps before what they require (LibreChat before its database)
    result.to_stop = start_order(instances, [i for i in previous if i not in allocations], None)[::-1]
    result.to_start = [i for i in start_order(instances, allocations, target) if i not in previous]
    result.to_restart = [i for i in _priority_order(instances, allocations, target)
                         if i in previous and (allocations[i] != previous[i]
                                               or (action == "restart" and i == target))]
    return result


def with_requirements(instances: dict[str, Instance], ids) -> set[str]:
    """The instances plus everything they require, and what that requires."""
    out, todo = set(), list(ids)
    while todo:
        i = todo.pop()
        if i in out or i not in instances:
            continue
        out.add(i)
        todo += instances[i].requires
    return out


def start_order(instances: dict[str, Instance], ids, target: str | None) -> list[str]:
    """The order to start instances in: by priority, but anything an instance requires first."""
    ids, out = set(ids), []

    def add(i: str, path: tuple[str, ...] = ()) -> None:
        if i in out or i not in ids or i in path:
            return
        for r in instances[i].requires:
            add(r, path + (i,))
        out.append(i)
    for i in _priority_order(instances, ids, target):
        add(i)
    return out


def _priority_order(instances: dict[str, Instance], ids, target: str | None,
                    first: set[str] | None = None) -> list[str]:
    first = first or set()

    def key(i: str):
        pol = instances[i].gpu
        mode_rank = {"fixed": 0, "elastic": 1, "none": 2}[pol.mode]
        return (0 if i == target else 1 if i in first else 2, -pol.priority, mode_rank, i)
    return sorted(ids, key=key)


def _allocate(gpus: list[int], gpu_set: set[int], instances: dict[str, Instance], wanted: set[str],
              target: str | None, current: dict[str, list[int]], usable: dict[str, int],
              gpu_mem: dict[int, float], first: set[str]
              ) -> tuple[dict[str, list[int]], list[str], list[str], str]:
    """Two phases: first decide HOW MANY GPUs every instance gets (and who does not fit),
    then decide WHICH ones, keeping running instances on the GPUs they already have.
    Instances that share a GPU are grouped into shared slots, which take one GPU each."""
    order = _priority_order(instances, wanted, target, first)
    free = set(gpus)                 # GPUs not taken by a fixed instance yet
    reserved = 0                     # GPUs promised to elastic instances and shared slots (picked in phase 2)
    counts: dict[str, int] = {}
    fixed: dict[str, list[int]] = {}
    dropped: list[str] = []
    on_cpu: list[str] = []
    # shared slots: members and the video memory they have promised. Which GPU a slot gets is only
    # known in phase 2, so it must fit the smallest GPU (all the same size on most machines).
    slots: list[dict] = []
    slot_cap = min((gpu_mem.get(g, 0) for g in gpus), default=0)

    def no_gpu(i: str) -> str:
        """Somewhere for an elastic instance that gets no GPU: the CPU, or stopped."""
        if instances[i].gpu.cpu_fallback:
            counts[i] = 0
            on_cpu.append(i)
            return ""
        if i == target:
            pol = instances[i].gpu
            return (f"{instances[i].name} needs at least {pol.min_gpus} GPU(s); "
                    f"only {max(len(free) - reserved, 0)} of this machine's {len(gpus)}"
                    f"{' AMD GPUs' if pol.vendor == 'amd' else ''} are available to it")
        dropped.append(i)
        return ""

    # phase 1a: minimums in priority order (the target first)
    for i in order:
        pol = instances[i].gpu
        if pol.mode == "none":
            counts[i] = 0
        elif pol.mode == "fixed":
            missing = [g for g in pol.gpus if g not in gpu_set]
            if missing:
                if i == target:
                    return {}, [], [], (f"{instances[i].name} is set to use GPU {','.join(map(str, missing))}, "
                                        "which this machine does not have")
                dropped.append(i)
                continue
            # its GPUs must be free, and taking them must still leave room for elastic promises
            if set(pol.gpus) <= free and len(free) - len(pol.gpus) >= reserved:
                fixed[i] = sorted(pol.gpus)
                free -= set(pol.gpus)
            else:
                dropped.append(i)
        elif pol.shares and pol.vram_gb <= slot_cap:
            # join a shared slot it fits in (one it shares a GPU with now first), else open one on
            # a free GPU. Without GPU memory sizes, or with a budget bigger than a GPU, it takes a
            # GPU of its own like any other instance (the branch below).
            mine = set(current.get(i, []))
            fits = [s for s in slots if s["used"] + pol.vram_gb <= slot_cap]
            fits.sort(key=lambda s: not any(set(current.get(m, [])) & mine for m in s["members"]))
            if fits:
                fits[0]["members"].append(i)
                fits[0]["used"] += pol.vram_gb
                counts[i] = 1
            elif len(free) - reserved >= 1:
                slots.append({"members": [i], "used": pol.vram_gb})
                reserved += 1
                counts[i] = 1
            elif err := no_gpu(i):
                return {}, [], [], err
        else:
            if len(free) - reserved >= pol.min_gpus:
                counts[i] = pol.min_gpus
                reserved += pol.min_gpus
            elif err := no_gpu(i):
                return {}, [], [], err

    # phase 1b: elastic instances grow into what is left, highest priority first
    sharing = {m for s in slots for m in s["members"]}
    spare = len(free) - reserved
    for i in order:
        pol = instances[i].gpu
        if pol.mode != "elastic" or i not in counts or i in on_cpu or i in sharing:
            continue
        cap = min(pol.max_gpus or len(gpus), max(usable.get(i, len(gpus)), counts[i]))
        extra = min(spare, cap - counts[i])
        counts[i] += extra
        spare -= extra

    # phase 2: pick concrete GPUs for elastic instances and shared slots from what fixed ones left.
    # A unit is one elastic instance, or one shared slot (all its members get the same GPU).
    alloc: dict[str, list[int]] = {i: [] for i, c in counts.items() if instances[i].gpu.mode == "none"}
    alloc.update(fixed)
    alloc.update({i: [] for i in on_cpu})
    pool = sorted(free)
    rank_of = {i: n for n, i in enumerate(order)}
    units = [([i], counts[i]) for i in order
             if i in counts and instances[i].gpu.mode == "elastic" and i not in on_cpu and i not in sharing]
    units += [(s["members"], 1) for s in slots]
    units.sort(key=lambda u: min(rank_of[m] for m in u[0]))
    got: list[list[int]] = []
    # 2a: everyone keeps the GPUs they already have (up to their new count); a slot keeps the GPU
    # most of its members are on
    for members, n in units:
        held = [g for m in members for g in current.get(m, []) if g in pool]
        keep = sorted(sorted(set(held), key=lambda g: (-held.count(g), g))[:n])
        got.append(keep)
        pool = [g for g in pool if g not in keep]
    # 2b: fill the rest in priority order: preferred GPUs first, then next to ones it has, then lowest
    for (members, n), have in zip(units, got):
        pref = instances[members[0]].gpu.gpus
        while len(have) < n and pool:
            def rank(g, pref=pref, have=have):
                near = min((abs(g - h) for h in have), default=0)
                return (pref.index(g) if g in pref else len(pref), near, g)
            g = min(pool, key=rank)
            have[:] = sorted(have + [g])
            pool.remove(g)
        for m in members:
            alloc[m] = list(have)
    return alloc, dropped, on_cpu, ""
