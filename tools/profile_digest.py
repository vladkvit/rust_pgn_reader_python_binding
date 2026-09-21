#!/usr/bin/env python3
"""Turn a samply profile.json into a compact, symbol-aware text digest.

Symbolicates native frames through the local samply symbol server
(POST {symbol_base}/symbolicate/v5, Mozilla symbolication API v5), then
aggregates self/total sample weights per function and folded hot-path
stacks per thread group. Everything it writes (digest, collapsed stacks,
symbol cache) goes into the --outdir directory (default: profile_out/);
delete that folder to clean up. Resolved symbols are cached in
profile_symbols.json so re-runs are instant and need no server.

Thread groups: "main" = python.exe threads (boots Python, prints
results), "arrow-loader" = the thread that loads the data (default tid
41720), "idle-tpp" = Windows thread-pool threads that are >80% idle
waits, "workers" = everything else (the parser thread pool).

The symbol server answers "unknown" for addresses in gaps of the PDB
function records (~1/3 of the Rust pyd's addresses) and nothing at all
for modules without debug info (pyarrow/numpy ship without PDBs).
Unsymbolicated frames are bucketed per module as "<mod:unsym>" in the
self/total tables and kept as "mod+0x<rva>" in the hot paths.

Usage:
    python tools/profile_digest.py
    python tools/profile_digest.py --symbol-base http://127.0.0.1:3000/<token>

First run requires the samply server from `samply record` (or
`samply load profile.json`) to be reachable.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.error
import urllib.request

ZERO_ID = "0" * 32          # breakpadId of modules without symbols (pyarrow, numpy, ...)
CHUNK = 1500                # frames per symbolication request
RUST_CRATE = "rust_pgn_reader_python_binding"


def eprint(*args):
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------- profile --

def frame_lib_rva(thread):
    """Per frame index -> (lib_index | None, rva | None).

    Frame -> module mapping: frameTable.func -> funcTable.resource
    -> resourceTable.lib (index into the top-level libs array).
    frameTable.address is already module-relative.
    """
    ft = thread["frameTable"]
    funct = thread["funcTable"]
    rt = thread["resourceTable"]
    out = []
    for i in range(ft["length"]):
        a = ft["address"][i]
        if a is None or a < 0:
            out.append((None, None))
            continue
        r = funct["resource"][ft["func"][i]]
        li = rt["lib"][r] if r is not None and r >= 0 else None
        out.append((li, a))
    return out


def stack_resolver(thread):
    """Return a fn stack_idx -> tuple of frame indices, root..leaf."""
    st = thread["stackTable"]
    memo = {}

    def resolve(idx):
        if idx in memo:
            return memo[idx]
        chain = []
        j = idx
        while j is not None and j not in memo:
            chain.append(j)
            j = st["prefix"][j]
        base = memo.get(j, ())
        for j in reversed(chain):
            base = base + (st["frame"][j],)
            memo[j] = base
        return memo[idx]

    return resolve


# ----------------------------------------------------------- symbolication --

def symbolicate(base, libs, needed, cache):
    """Fill cache {debugName/breakpadId: {rva_str: func_name | None}}."""
    for li in sorted(needed):
        lib = libs[li]
        key = "%s/%s" % (lib["debugName"], lib["breakpadId"])
        entry = cache.setdefault(key, {})
        todo = [a for a in needed[li] if str(a) not in entry]
        if not todo:
            continue
        if lib["breakpadId"] == ZERO_ID:
            for a in todo:
                entry[str(a)] = None
            continue
        eprint("symbolicating %5d addrs  %s" % (len(todo), lib["debugName"]))
        for off in range(0, len(todo), CHUNK):
            part = todo[off:off + CHUNK]
            job = {"jobs": [{"memoryMap": [[lib["debugName"], lib["breakpadId"]]],
                             "stacks": [[[0, a]] for a in part]}]}
            req = urllib.request.Request(
                base + "/symbolicate/v5", data=json.dumps(job).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    out = json.load(resp)
                stacks = out["results"][0]["stacks"]
            except urllib.error.HTTPError as e:
                eprint("  HTTP %s for %s; marking chunk unresolved" % (e.code, lib["debugName"]))
                stacks = [[] for _ in part]
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                eprint("cannot reach symbol server: %s" % e)
                eprint("restart with: samply load profile.json "
                       "(then pass --symbol-base with the new token)")
                sys.exit(2)
            for a, s in zip(part, stacks):
                entry[str(a)] = s[0].get("function") if s else None


# ----------------------------------------------------------------- labels --

def is_os_lib(lib):
    return "WINDOWS" in lib.get("path", "").upper()


def rust_short(func):
    prefix = RUST_CRATE + "::"
    return func[len(prefix):] if func and func.startswith(prefix) else func


def short_mod(mod):
    if mod.startswith(RUST_CRATE):
        return "rust_pgn"
    return mod.rsplit(".", 1)[0]


def table_label(lib, func, rva, raw_name):
    """Label for self/total tables. Unsymbolicated frames are bucketed
    per module so they aggregate instead of fragmenting by address."""
    if lib is None:
        return raw_name or "[unknown]"
    mod = lib["name"]
    if func and not func.startswith("fun_"):
        if mod.startswith(RUST_CRATE):
            return rust_short(func)
        return "%s!%s" % (mod, func)
    return "<%s:unsym>" % short_mod(mod)


def path_label(lib, func, rva):
    """Short label for folded hot paths; None = drop from path.
    Unsymbolicated frames keep their address here for drill-down."""
    if lib is None:
        return None
    mod = lib["name"]
    if mod.startswith(RUST_CRATE):
        return rust_short(func) if func else "rust+0x%x" % rva
    if is_os_lib(lib):
        return "[os]"
    if mod == "python311.dll":
        return "py:%s" % func if func and not func.startswith("fun_") else None
    if func and not func.startswith("fun_"):
        return "%s!%s" % (short_mod(mod), func)
    return "%s+0x%x" % (short_mod(mod), rva)   # unresolved pyarrow/numpy etc.


# ------------------------------------------------------------------ main --

def add_counter(lines, title, counter, total, limit, min_pct=0.3):
    lines.append("")
    lines.append(title)
    for label, c in counter.most_common(limit):
        pct = 100.0 * c / total if total else 0.0
        if pct < min_pct:
            break
        lines.append("%9.0f %5.1f%%  %s" % (c, pct, label))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profile.json")
    ap.add_argument("--outdir", default="profile_out",
                    help="directory for the digest, collapsed stacks and symbol cache")
    ap.add_argument("--symbol-base",
                    default="http://127.0.0.1:3000/2wnjgjpyzlsdzg2pbqib466i3zbml8qm1jrh7fp")
    ap.add_argument("--cache", default=None, help="default: <outdir>/profile_symbols.json")
    ap.add_argument("--out", default=None, help="default: <outdir>/profile_digest.txt")
    ap.add_argument("--collapsed-prefix", default=None,
                    help="default: <outdir>/profile_collapsed")
    ap.add_argument("--arrow-tid", default="41720")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache_path = args.cache or os.path.join(args.outdir, "profile_symbols.json")
    out_path = args.out or os.path.join(args.outdir, "profile_digest.txt")
    collapsed_prefix = args.collapsed_prefix or os.path.join(args.outdir, "profile_collapsed")

    with open(args.profile, "r", encoding="utf-8") as f:
        prof = json.load(f)
    libs = prof["libs"]
    interval = float(prof.get("meta", {}).get("interval", 1.0))

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}

    per_thread = []
    needed = collections.defaultdict(set)
    for t in prof["threads"]:
        flr = frame_lib_rva(t)
        per_thread.append(flr)
        for li, a in flr:
            if li is not None and a is not None:
                needed[li].add(a)

    symbolicate(args.symbol_base, libs, needed, cache)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    def resolve_func(li, a):
        lib = libs[li]
        return cache.get("%s/%s" % (lib["debugName"], lib["breakpadId"]), {}).get(str(a))

    def group_of(name, dominant_leaf, dominant_share):
        if name == "python.exe":
            return "main"
        if (dominant_leaf == "ntdll.dll!NtWaitForWorkViaWorkerFactory"
                and dominant_share > 0.8):
            return "idle-tpp"
        if args.arrow_tid and ("<%s>" % args.arrow_tid) in name:
            return "arrow-loader"
        return "workers"

    groups = collections.OrderedDict()
    global_self = collections.Counter()
    global_total = collections.Counter()
    global_mod_self = collections.Counter()
    thread_rows = []
    total_weight = 0.0
    max_span_ms = 0.0

    for t, flr in zip(prof["threads"], per_thread):
        s = t.get("samples")
        if not s or not s.get("length"):
            continue
        sa = t.get("stringArray", [])
        ft = t["frameTable"]
        funct = t["funcTable"]
        weights = s.get("weight") or [1.0] * s["length"]
        resolve = stack_resolver(t)
        label_memo = {}

        def labels(fi):
            if fi not in label_memo:
                li, a = flr[fi]
                func = resolve_func(li, a) if li is not None else None
                if func == "unknown":
                    func = None   # server answer for PDB gaps; treat as unresolved
                raw = sa[funct["name"][ft["func"][fi]]]
                lib = libs[li] if li is not None else None
                label_memo[fi] = (table_label(lib, func, a, raw),
                                  path_label(lib, func, a))
            return label_memo[fi]

        # first pass: dominant leaf, used for thread classification
        t_self = collections.Counter()
        t_weight = 0.0
        for i in range(s["length"]):
            w = weights[i]
            w = float(w) if w else 1.0
            stidx = s["stack"][i]
            if stidx is not None:
                frames = resolve(stidx)
                if frames:
                    t_self[labels(frames[-1])[0]] += w
            t_weight += w
        top_leaf, top_leaf_w = t_self.most_common(1)[0] if t_self else ("-", 0.0)
        top_share = top_leaf_w / t_weight if t_weight else 0.0

        gname = group_of(t.get("name", ""), top_leaf, top_share)
        gd = groups.setdefault(gname, {
            "self": collections.Counter(), "total": collections.Counter(),
            "mod_self": collections.Counter(), "collapsed": collections.Counter(),
            "ctx": collections.Counter(),
            "unresolved_self": collections.Counter(),
            "weight": 0.0, "threads": 0})
        gd["threads"] += 1

        for i in range(s["length"]):
            w = weights[i]
            w = float(w) if w else 1.0
            frames = resolve(s["stack"][i]) if s["stack"][i] is not None else ()
            tl = [labels(f)[0] for f in frames]
            leaf = tl[-1] if tl else "[empty]"
            global_self[leaf] += w
            gd["self"][leaf] += w
            for x in set(tl):
                global_total[x] += w
                gd["total"][x] += w
            if frames:
                li, a = flr[frames[-1]]
                mod = libs[li]["name"] if li is not None else "[no-lib]"
                if (li is not None and libs[li]["breakpadId"] == ZERO_ID):
                    gd["unresolved_self"][mod] += w
            else:
                mod = "[empty]"
            global_mod_self[mod] += w
            gd["mod_self"][mod] += w
            dedup = []
            for f in frames:
                x = labels(f)[1]
                # cap entry length: giant monomorphized rayon names otherwise
                # make collapsed lines ~20 KB and the file unusable
                if x:
                    x = x[:100]
                    if not dedup or dedup[-1] != x:
                        dedup.append(x)
            gd["collapsed"][";".join(dedup) if dedup else "[filtered]"] += w
            ctx = ";".join(x[:80] for x in dedup[-3:])
            gd["ctx"][ctx if ctx else "[filtered]"] += w
            gd["weight"] += w
            total_weight += w
        span_ms = sum(x for x in (s.get("timeDeltas") or []) if x)
        max_span_ms = max(max_span_ms, span_ms)
        thread_rows.append((t.get("name", "?"), t_weight, span_ms, top_leaf,
                            100.0 * top_share))

    # ------------------------------------------------------------- render --
    L = []
    w = L.append
    w("profile digest: %s" % args.profile)
    w("interval %.4f ms | weighted samples %.0f (~%.1f s CPU) | approx wall span %.1f s"
      % (interval, total_weight, total_weight * interval / 1000.0, max_span_ms / 1000.0))
    w("groups: " + ", ".join("%s=%d threads" % (g, gd["threads"]) for g, gd in groups.items()))
    w("symbols: %d libs, cache %s" % (len(needed), cache_path))

    w("")
    w("== per-thread overview (by weight) ==")
    thread_rows.sort(key=lambda r: -r[1])
    for name, tw, span, leaf, pct in thread_rows[:20]:
        w("%-22s %8.0f %6.1f s  top leaf (%4.1f%%): %s"
          % (name, tw, span / 1000.0, pct, leaf[:70]))
    if len(thread_rows) > 20:
        w("... and %d more threads" % (len(thread_rows) - 20))

    add_counter(L, "== self time by module (all threads) ==", global_mod_self, total_weight, 15)
    add_counter(L, "== top self functions (all threads) ==", global_self, total_weight, args.top)
    add_counter(L, "== top total functions (on-stack, all threads) ==", global_total, total_weight, args.top)

    for g, gd in groups.items():
        w("")
        w("==" * 30)
        w("== group: %s | %d threads | %.0f samples (%.1f%% of CPU)"
          % (g, gd["threads"], gd["weight"], 100.0 * gd["weight"] / total_weight))
        add_counter(L, "-- self time by module --", gd["mod_self"], gd["weight"], 10)
        add_counter(L, "-- top self functions --", gd["self"], gd["weight"], 15)
        add_counter(L, "-- hot leaf contexts (last 3 interesting frames) --",
                    gd["ctx"], gd["weight"], 20)
        add_counter(L, "-- hot paths (filtered) --", gd["collapsed"], gd["weight"], args.top, min_pct=0.5)
        if gd["unresolved_self"]:
            add_counter(L, "-- unresolved self time (no symbols available) --",
                        gd["unresolved_self"], gd["weight"], 5)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    for g, gd in groups.items():
        with open("%s_%s.txt" % (collapsed_prefix, g), "w", encoding="utf-8") as f:
            f.write("# folded stacks: frames root->leaf, ';'-separated, trailing int = weight\n")
            for path, cnt in gd["collapsed"].most_common():
                if cnt < 5:
                    break   # noise tail
                f.write("%s %.0f\n" % (path, cnt))
    eprint("wrote %s and %s_{%s}.txt" % (out_path, collapsed_prefix, ",".join(groups)))


if __name__ == "__main__":
    main()
