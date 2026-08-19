"""
Tokenize the Baron pancreas donor h5ad files and extract frozen Geneformer-
V1-10M cell embeddings (pretrained model, no fine-tuning).

Output: results/geneformer_embeddings.csv -- one row per cell, embedding
columns plus preserved cell_type/donor labels, aligned by cell barcode via
the custom_attr dict passed to the tokenizer.
"""
from pathlib import Path

from geneformer import EmbExtractor, TranscriptomeTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
FOR_TOKENIZER_DIR = BASE_DIR / "data" / "for_tokenizer"
TOKENIZED_DIR = BASE_DIR / "data" / "tokenized"
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
EMB_OUT_DIR = BASE_DIR / "results"
EMB_OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / "models" / "Geneformer" / "Geneformer-V1-10M"

OUTPUT_PREFIX = "baron_pancreas"


def main():
    print("=== Tokenizing ===")
    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type", "donor": "donor"},
        nproc=4,
        model_version="V1",
    )
    tk.tokenize_data(
        str(FOR_TOKENIZER_DIR),
        str(TOKENIZED_DIR),
        OUTPUT_PREFIX,
        file_format="h5ad",
    )
    print(f"Tokenized dataset -> {TOKENIZED_DIR / (OUTPUT_PREFIX + '.dataset')}")

    print("=== Extracting embeddings ===")
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_layer=-1,
        max_ncells=None,
        emb_label=["cell_type", "donor"],
        forward_batch_size=32,
        model_version="V1",
        nproc=4,
    )
    embs = embex.extract_embs(
        str(MODEL_DIR),
        str(TOKENIZED_DIR / (OUTPUT_PREFIX + ".dataset")),
        str(EMB_OUT_DIR),
        OUTPUT_PREFIX + "_embs",
    )
    print("Embeddings shape:", embs.shape)
    print(embs.head())


if __name__ == "__main__":
    main()
