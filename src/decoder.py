"""Generate JSON values token by token, masking every forbidden token."""
import numpy as np
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model

from .grammar import ValueConstraint, find_end
from .vocab import Vocabulary


class ConstrainedDecoder(BaseModel):
    """Drive the model so that every generated value obeys a constraint.

    Attributes:
        llm: The language model from the SDK.
        texts: Text of each token ID, or None if it may never be generated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    texts: list[str | None]

    @classmethod
    def build(cls, llm: Small_LLM_Model,
              vocabulary: Vocabulary) -> "ConstrainedDecoder":
        """Prepare the text of every token once.

        Args:
            llm: The loaded language model.
            vocabulary: Bytes of every token ID.

        Returns:
            A decoder ready to generate values.
        """
        texts: list[str | None] = []
        for token_id in range(vocabulary.size):
            raw = vocabulary.get(token_id)
            text: str | None = None
            if raw:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
            if text is not None and any(ord(char) < 0x20 for char in text):
                text = None
            texts.append(text)

        return cls(llm=llm, texts=texts)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs.

        Args:
            text: Any text, special tokens included.

        Returns:
            The token IDs.
        """
        ids: list[int] = self.llm.encode(text).tolist()[0]
        return ids

    def allowed_tokens(self, constraint: ValueConstraint,
                       generated: str) -> list[int]:
        """Return every token ID the constraint accepts next.

        Args:
            constraint: Rules for the value being generated.
            generated: Value text generated so far.

        Returns:
            The allowed token IDs.
        """
        return [token_id for token_id, text in enumerate(self.texts)
                if text is not None and constraint.accepts(generated, text)]

    def choose_best(self, ids: list[int], allowed: list[int]) -> int:
        """Pick the best allowed next token.

        Args:
            ids: Every token ID so far (prompt and output).
            allowed: Token IDs the constraint accepts.

        Returns:
            The allowed token with the highest logit.
        """
        if len(allowed) == 1:
            return allowed[0]
        logits = np.array(self.llm.get_logits_from_input_ids(ids),
                          dtype=np.float32)
        masked = np.full_like(logits, -np.inf)
        masked[allowed] = logits[allowed]
        return int(np.argmax(masked))

    def generate_value(self, ids: list[int],
                       constraint: ValueConstraint) -> tuple[str, str]:
        """Generate one value and the fixed text that follows it.

        Args:
            ids: Every token ID so far; extended in place.
            constraint: Rules for the value and its suffix.

        Returns:
            The value, and all the text that was written.

        Raises:
            RuntimeError: If no token is allowed.
        """
        generated = ""
        written = ""
        while True:
            allowed = self.allowed_tokens(constraint, generated)
            if not allowed:
                raise RuntimeError(
                    f"no token allowed after {generated!r}")
            token_id = self.choose_best(ids, allowed)
            text = self.texts[token_id] or ""
            ids.append(token_id)
            written += text
            full = generated + text
            end = find_end(full, constraint.suffix[0])
            if end == -1:
                generated = full
                continue
            remaining_suffix = constraint.suffix[len(full) - end:]
            if remaining_suffix:
                ids.extend(self.encode(remaining_suffix))
                written += remaining_suffix
            return full[:end], written
