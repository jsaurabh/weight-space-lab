"""Validate the generic GPU interface in a separate disposable worker workspace."""
import argparse
from pathlib import Path

from driver import Bridge, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="weight-space-qwen-compare")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bridge = Bridge(args.session, "validation")
    print(bridge.launch(), flush=True)
    try:
        metadata = bridge.rpc("metadata", {}, timeout=900)
        assert metadata["ok"], metadata
        edited = bridge.rpc("python_analysis", {"code":
            "n=next(n for n in names if n.startswith('model.language_model.layers.') and len(shapes[n])==2)\n"
            "w=weight(n); w.flat[0]+=0.125\nsave_edit(n,w)\n"
            "clear_edits()\nw=weight(n); w[:,1]+=0.125\nsave_edit(n,w)\nprint(list_edits())"})
        assert edited["ok"], edited
        native = bridge.rpc("smoke_reload", {}, timeout=900)
        assert native["ok"] and native["result"]["base_roundtrip_exact"], native
        probe = bridge.rpc("probe", {"which": "base", "texts": ["The result is 4.\n"],
                                   "modules": [], "positions": [-2, -1]}, timeout=900)
        assert probe["ok"], probe
        write(args.output / "preflight.json", {"metadata_practice_count": metadata["result"]["practice_count"],
            "native_reload_validated": True, "probability_probe_validated": True,
            "sparse_entry_and_column_edits_validated": True,
            "separate_workspace": True, "editor_budget_used": 0})
        bridge.export(Path(".artifacts/qwen-compare/preflight-export"))
        print("Hardware preflight passed", flush=True)
    finally:
        bridge.stop_worker()


if __name__ == "__main__":
    main()
