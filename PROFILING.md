# Profiling

- `samply record --rate 10000 python src\bench_parse_games.py` writes profile.json(.gz) and stays running as symbol server + profiler UI on http://127.0.0.1:3000/.
- The symbol-server URL token changes per server start. GET http://127.0.0.1:3000/ shows the current token in its "Open the profile in the profiler UI" link. 404s from profile_digest.py = stale `--symbol-base`.
- samply may leave profile.json stale while updating profile.json.gz; refresh with `python -c "import gzip,shutil;shutil.copyfileobj(gzip.open('profile.json.gz','rb'),open('profile.json','wb'))"`.
- Symbol cache profile_out/profile_symbols.json is keyed by pdb/breakpadId: once a build's entries exist, digest reruns need no server; old builds' entries never collide. If the server is unreachable the digest exits before writing the cache — safe to just retry.
- Digest outputs (profile_digest.txt, profile_collapsed_<group>.txt, profile_symbols.json) all go to profile_out/; delete the dir to clean up.
- Digest thread groups lie: "main" merges both python.exe pids — the .venv shim is a launcher process parked in WaitForSingleObject (~12% of samples, pure noise). "workers" includes ~15 parked pyarrow-pool threads (~283 samples each, ntdll leafs) and the arrow loader unless `--arrow-tid` matches; the loader tid changes every run — find it by the `arrow.dll` top leaf in the per-thread overview. Real workers: the 16 threads with pyd-dominant leafs.
- samply samples blocked threads too: "weighted samples (~N s CPU)" is thread-time incl. waits, not CPU time.
- samply's breakpad symbolication leaves ~1/3 of the Rust pyd's addresses "unknown" even for `[profile.profiling]` builds with `debug = true` (128 MB PDB present) — its PDB dump misses function records the PDB itself has. `tools/profile_digest.py` patches this after server symbolication via dbghelp (`SymFromAddr` displacements cluster unresolved addresses by function start; clusters are named after server-resolved anchors sharing the start, or labeled `<rust gap fn 0x...>` when no anchor covers them).
- dbghelp enrichment resolves against the pyd currently at the profiled module path (`.venv\...cp311-win_amd64.pyd`); its embedded CodeView record finds the PDB under `target\profiling\` automatically. Re-record the profile after a rebuild — enrichment would otherwise attach stale addresses' names.
- profile_collapsed lines can be ~20 KB (rayon monomorphized names, capped at 100 chars/frame).
- Machine: 7800X3D, 8 physical / 16 logical cores → 16 rayon workers.
-Don't run the samply record line inside a harness (e.g. opencode). The command stays running, and the harness waits for it to close - ends up timing out. 
