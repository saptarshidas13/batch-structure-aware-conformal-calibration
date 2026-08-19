"""Time embedding extraction on a small cell subset to extrapolate full runtime."""
import time
from pathlib import Path

from geneformer import EmbExtractor

BASE_DIR = Path(__file__).resolve().parents[1]
TOKENIZED_DATASET = BASE_DIR / "data" / "tokenized" / "baron_pancreas.dataset"
OUT_DIR = BASE_DIR / "results" / "timing_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / "models" / "Geneformer" / "Geneformer-V1-10M"

N_TEST_CELLS = 200

if __name__ == "__main__":
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_layer=-1,
        max_ncells=N_TEST_CELLS,
        emb_label=["cell_type", "donor"],
        forward_batch_size=16,
        model_version="V1",
        nproc=4,
    )
    start = time.time()
    embs = embex.extract_embs(
        str(MODEL_DIR),
        str(TOKENIZED_DATASET),
        str(OUT_DIR),
        "timing_test",
    )
    elapsed = time.time() - start
    print(f"\nTIMING: {N_TEST_CELLS} cells took {elapsed:.1f}s ({elapsed/N_TEST_CELLS:.3f} s/cell)")
    print(f"Embeddings shape: {embs.shape}")
    print(embs.head())
