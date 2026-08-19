"""Tokenize all 4 Baron donor h5ad files into a single Geneformer dataset."""
from pathlib import Path

from geneformer import TranscriptomeTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
FOR_TOKENIZER_DIR = BASE_DIR / "data" / "for_tokenizer"
TOKENIZED_DIR = BASE_DIR / "data" / "tokenized"
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = "baron_pancreas"


def main():
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


if __name__ == "__main__":
    main()
