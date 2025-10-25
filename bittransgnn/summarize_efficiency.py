import re, argparse, json, math
from pathlib import Path
from typing import List, Dict, Any, Tuple

### code to parse benchmark_results_*.txt and summarize efficiency

KEEP_SCENARIOS = {"fp32_32-32-32-32", "bin_1-1-1-8_int8Linear"}

def _to_num(s: str):
    s = s.strip()
    if s.lower() == "none":
        return None
    try:
        return float(s.replace(",", ""))  # allow commas in numbers
    except Exception:
        return None

def parse_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Parse the loose key:value text into a list of dict blocks.
    A new block starts at a line beginning with 'model:'.
    """
    blocks = []
    cur = {}
    for line in text.splitlines():
        if not line.strip():
            continue

        # lines look like "key: value" with varying left padding
        m = re.match(r"^\s*([^:]+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()

        # If we hit a new 'model:', start a new block
        if key == "model":
            if cur:
                blocks.append(cur)
            cur = {}

        # Normalize keys we care about
        key_norm = key.strip()

        # Try to coerce common numeric fields
        if key_norm in {
            "avg_latency_ms", "iters", "samples_per_iter", "throughput_items_per_s",
            "gpu_peak_mem_mb", "avg_latency_ms_events", "num_batches",
            "bert_est_epoch_s_batches", "epoch_time_ms", "epoch_energy_j",
            "energy_j", "avg_watts", "peak_watts", "duration_s"
        }:
            cur[key_norm] = _to_num(val)
        else:
            # keep raw string for identifiers
            cur[key_norm] = val

    if cur:
        blocks.append(cur)
    return blocks

def compute_epoch_time_ms(entry: Dict[str, Any]) -> float:
    """
    Apply rules:
      - bert, bertgcn_d (and 'bertgcn_d (doc_doc)'): epoch_time_ms = avg_latency_ms * num_batches
      - bertgcn, bertgcn (doc_doc): epoch_time_ms = avg_latency_ms
    """
    model = entry.get("model", "")
    avg_ms = entry.get("avg_latency_ms", None)
    nb = entry.get("num_batches", None)

    if avg_ms is None:
        return None

    if model.startswith("bertgcn") and not model.startswith("bertgcn_d"):
        # 'bertgcn' and 'bertgcn (doc_doc)'
        return avg_ms
    else:
        # 'bert' or 'bertgcn_d' (and its doc_doc variant)
        if nb is None:
            return None
        return avg_ms * nb

def compute_avg_watts(epoch_energy_j, epoch_time_ms):
    if epoch_energy_j is None or epoch_time_ms is None or epoch_time_ms == 0:
        return None
    # W = J / s ; epoch_time_ms is in ms
    return (epoch_energy_j / (epoch_time_ms / 1000.0))

def filter_and_project(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the two scenarios; compute epoch_time_ms per rules;
    pick epoch_energy_j from the existing field (may be None); compute average_watts.
    Return compact rows: model, scenario, device, epoch_time_ms, epoch_energy_j, average_watts, adj_type (if present).
    """
    out = []
    for b in blocks:
        model = b.get("model", "")
        scenario = b.get("scenario", "")
        device = b.get("device", "")
        if scenario not in KEEP_SCENARIOS:
            continue

        epoch_time_ms = compute_epoch_time_ms(b)

        # Take epoch_energy_j if present; else None
        epoch_energy_j = b.get("**epoch_energy_j**", None)
        print(epoch_energy_j)
        epoch_energy_j = float(epoch_energy_j) if epoch_energy_j not in (None, "None") else None
        avg_watts = compute_avg_watts(epoch_energy_j, epoch_time_ms)

        row = {
            "model": model,
            "scenario": scenario,
            "device": device,
            "adj_type": b.get("adj_type", None),
            "epoch_time_ms": epoch_time_ms,
            "epoch_energy_j": epoch_energy_j,
            "average_watts": avg_watts,
        }
        out.append(row)
    return out

def build_combo_transformer_gnn(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create 'transformer + gnn' rows by pairing:
      - transformer: model == 'bert'
      - gnn: model in {'bertgcn', 'bertgcn (doc_doc)'}  (not the *_d variant)
    Must match on (scenario, device, adj_type).
    """
    out = []
    # index rows by key
    by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (r["model"], r["scenario"], r.get("device"), r.get("adj_type"))
        by_key[key] = r

    # collect candidates
    scenarios = set(r["scenario"] for r in rows)
    devices = set(r.get("device") for r in rows)
    adj_types = set(r.get("adj_type") for r in rows)

    for sc in scenarios:
        for dev in devices:
            for adj in adj_types:
                bert = by_key.get(("bert", sc, dev, adj))
                # Prefer to pair BERT with GCN of same adj_type if available; if not, try default 'full'/'doc_doc'
                for gname in ("bertgcn", "bertgcn (doc_doc)"):
                    g = by_key.get((gname, sc, dev, adj))
                    if not g:
                        continue
                    if not bert:
                        # try also matching BERT with no adj_type (many BERT rows have adj_type 'full' in your file)
                        bert = by_key.get(("bert", sc, dev, "full")) or by_key.get(("bert", sc, dev, None))
                    if not bert:
                        continue

                    t_ms = None
                    if bert["epoch_time_ms"] is not None and g["epoch_time_ms"] is not None:
                        t_ms = bert["epoch_time_ms"] + g["epoch_time_ms"]

                    e_j = None
                    be = bert["epoch_energy_j"]; ge = g["epoch_energy_j"]
                    if be is not None and ge is not None:
                        e_j = be + ge

                    avg_w = compute_avg_watts(e_j, t_ms)

                    out.append({
                        "model": "transformer + gnn",
                        "scenario": sc,
                        "device": dev,
                        "adj_type": adj,
                        "epoch_time_ms": t_ms,
                        "epoch_energy_j": e_j,
                        "average_watts": avg_w,
                        "components": {"transformer": bert["model"], "gnn": g["model"]},
                    })
    return out

def split_by_device(rows: List[Dict[str, Any]]):
    cpu = [r for r in rows if (r.get("device") or "").startswith("cpu")]
    gpu = [r for r in rows if (r.get("device") or "").startswith("cuda")]
    return cpu, gpu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_txt", required=True, type=Path, help="Path to benchmark_results_*_cpu.txt")
    ap.add_argument("--gpu_txt", required=True, type=Path, help="Path to benchmark_results_*_cuda:0.txt")
    ap.add_argument("--save_json", type=Path, default=None)
    ap.add_argument("--save_csv", type=Path, default=None)
    args = ap.parse_args()

    cpu_blocks = parse_blocks(args.cpu_txt.read_text())
    gpu_blocks = parse_blocks(args.gpu_txt.read_text())

    cpu_rows = filter_and_project(cpu_blocks)
    gpu_rows = filter_and_project(gpu_blocks)

    # build combined “transformer + gnn” rows (per device)
    cpu_combo = build_combo_transformer_gnn(cpu_rows)
    gpu_combo = build_combo_transformer_gnn(gpu_rows)

    cpu_final = cpu_rows + cpu_combo
    gpu_final = gpu_rows + gpu_combo

    print("\n=== CPU summary ===")
    for r in cpu_final:
        print(r)

    print("\n=== GPU summary ===")
    for r in gpu_final:
        print(r)

    if args.save_json:
        out = {"cpu": cpu_final, "gpu": gpu_final}
        args.save_json.write_text(json.dumps(out, indent=2))
        print(f"[saved] {args.save_json}")

    if args.save_csv:
        import csv
        keys = ["model", "scenario", "device", "adj_type", "epoch_time_ms", "epoch_energy_j", "average_watts"]
        with args.save_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in cpu_final + gpu_final:
                w.writerow({k: r.get(k) for k in keys})
        print(f"[saved] {args.save_csv}")

if __name__ == "__main__":
    main()
