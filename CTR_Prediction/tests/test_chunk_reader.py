import gzip
from pathlib import Path

from src.data.chunk_reader import GzipChunkReader


def test_gzip_chunk_reader(tmp_path: Path) -> None:
    path = tmp_path / "train.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("id,click,hour,C1,banner_pos\n")
        f.write("a,1,14102100,1005,0\n")
        f.write("b,0,14102101,1005,1\n")
    chunks = list(GzipChunkReader(path, chunksize=1, dtype_map={"click": "int8", "C1": "int16", "banner_pos": "int8"}))
    assert len(chunks) == 2
    assert chunks[0].shape[0] == 1

