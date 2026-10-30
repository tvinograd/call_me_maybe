"""Map every token ID of the model's vocabulary to the bytes it produces."""
import json

from pydantic import BaseModel


def build_byte_decoder() -> dict[str, int]:
    """Return the mapping from a vocabulary character to the byte it encodes.

    Reverses the byte-level BPE table: printable bytes map to themselves,
    the other 68 bytes were shifted to code points starting at 256.
    """
    printable = set(range(33, 127)) | set(range(161, 173)) \
        | set(range(174, 256))
    decoder: dict[str, int] = {}
    shift = 0
    for byte in range(256):
        if byte in printable:
            decoder[chr(byte)] = byte
        else:
            decoder[chr(256 + shift)] = byte
            shift += 1
    return decoder


class Vocabulary(BaseModel):
    """Raw bytes produced by each token ID (None for unused IDs)."""
    token_bytes: list[bytes | None]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Load vocab.json and convert every token string to bytes."""
        with open(path, encoding="utf-8") as file:
            vocab: dict[str, int] = json.load(file)
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError(f"Vocabulary file {path} is empty or invalid")

        decoder = build_byte_decoder()
        token_bytes: list[bytes | None] = [None] * (max(vocab.values()) + 1)
        for token, token_id in vocab.items():
            try:
                token_bytes[token_id] = bytes(decoder[c] for c in token)
            except KeyError:
                raise ValueError(
                    f"Vocabulary token {token!r} is not byte-level encoded"
                ) from None
        return cls(token_bytes=token_bytes)

    @property
    def size(self) -> int:
        """Return the number of token IDs covered by this vocabulary."""
        return len(self.token_bytes)

    def get(self, token_id: int) -> bytes | None:
        """Return the bytes of a token, or None if it is unknown/special."""
        if 0 <= token_id < self.size:
            return self.token_bytes[token_id]
        return None
