"""Map every token ID of the model's vocabulary to the bytes it produces."""
import json

from pydantic import BaseModel


def build_byte_decoder() -> dict[str, int]:
    """Build the byte-level BPE character-to-byte table.

    Returns:
        Mapping from each of the 256 characters to its byte.
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
    """Raw bytes of each token ID.

    Attributes:
        token_bytes: Bytes per token ID, None if missing.
    """

    token_bytes: list[bytes | None]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Load vocab.json as bytes per token ID.

        Args:
            path: Vocabulary file path.

        Returns:
            The vocabulary.

        Raises:
            ValueError: If the file or a token is invalid.
        """
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
        """Returns the number of token IDs covered by this vocabulary."""
        return len(self.token_bytes)

    def get(self, token_id: int) -> bytes | None:
        """Return a token's bytes.

        Args:
            token_id: Any token ID.

        Returns:
            The bytes, or None for special or unknown IDs.
        """
        if 0 <= token_id < self.size:
            return self.token_bytes[token_id]
        return None
