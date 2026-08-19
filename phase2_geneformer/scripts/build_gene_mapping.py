"""
Map Baron pancreas gene symbols -> Ensembl gene IDs, required by Geneformer's
tokenizer (var['ensembl_id']). Uses the mygene.info API, chunked with retries
-- a single 20k-gene batch request is prone to transient connection drops.
"""
import json
import time
from pathlib import Path

import anndata as ad
import mygene

DATA_DIR = Path(__file__).resolve().parents[2] / "phase1_baseline" / "data" / "processed"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "symbol_to_ensembl.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
MAX_RETRIES = 4


def query_chunk_with_retry(mg, chunk):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return mg.querymany(
                chunk, scopes="symbol", fields="ensembl.gene", species="human", verbose=False
            )
        except Exception as e:
            wait = 2**attempt
            print(f"    chunk query failed ({e!r}), retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Chunk failed after {MAX_RETRIES} retries: {chunk[:5]}...")


def main():
    all_genes = set()
    for i in range(1, 5):
        adata = ad.read_h5ad(DATA_DIR / f"baron_human{i}.h5ad")
        all_genes.update(adata.var_names.tolist())
    all_genes = sorted(all_genes)
    print(f"Total unique gene symbols across all donors: {len(all_genes)}")

    mg = mygene.MyGeneInfo()
    mapping = {}
    n_ambiguous = 0

    chunks = [all_genes[i : i + CHUNK_SIZE] for i in range(0, len(all_genes), CHUNK_SIZE)]
    for idx, chunk in enumerate(chunks, 1):
        print(f"  querying chunk {idx}/{len(chunks)} ({len(chunk)} genes)...")
        results = query_chunk_with_retry(mg, chunk)
        for r in results:
            symbol = r["query"]
            if symbol in mapping or r.get("notfound"):
                continue
            ens = r.get("ensembl")
            if ens is None:
                continue
            if isinstance(ens, list):
                n_ambiguous += 1
                ens_id = ens[0]["gene"]
            else:
                ens_id = ens["gene"]
            mapping[symbol] = ens_id

    print(
        f"Mapped {len(mapping)}/{len(all_genes)} symbols to Ensembl IDs "
        f"({n_ambiguous} ambiguous, took first hit)"
    )
    with open(OUT_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
