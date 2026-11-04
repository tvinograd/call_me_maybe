*This project has been created as part of the 42 curriculum by tvinogra.*

# call me maybe

## Description

Translates natural-language requests into function calls using the small
model Qwen3-0.6B. For *"What is the sum of 2 and 3?"* the program does not
answer `5`, it outputs:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

**Constrained decoding** guarantees valid JSON: at each step, every token
that would break the JSON structure or the function's schema gets a score
of `-inf`, so the model can only choose valid tokens. The model still
decides which function to call and which values to extract.

## Instructions

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), about 8 GB of
disk space (PyTorch and the model), and internet access on the first run.

```sh
make install                     # uv sync
uv run python -m src [--functions_definition FILE] [--input FILE] [--output FILE]
```

Defaults: `data/input/functions_definition.json`,
`data/input/function_calling_tests.json`,
`data/output/function_calling_results.json`.

Other rules: `make run`, `make debug` (pdb), `make clean`, `make lint`.

In case of limited storage:

```sh
export UV_CACHE_DIR=/goinfre/$USER/uv-cache
export UV_PROJECT_ENVIRONMENT=/goinfre/$USER/venvs/call_me_maybe
export HF_HOME=/goinfre/$USER/hf
```

## Example usage

```
$ uv run python -m src
Loaded 5 functions and 11 prompts.
[1/11] 'What is the sum of 2 and 3?' -> fn_add_numbers{"a": 2, "b": 3}
[3/11] 'Greet shrek' -> fn_greet{"name": "shrek"}
...
Wrote 11 results to data/output/function_calling_results.json in 152.2 second(s).

$ uv run python -m src --input missing.json
Error: Cannot read missing.json: No such file or directory
```

A request that matches no function is reported and not written to the file:

```
[2/3] SKIP "What's the weather in Paris?": No function matches this request
```

## Algorithm explanation

The program generates this text for each request:

```
{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}
```

Braces, keys and separators are fixed, so the program writes them itself.
The model only generates the **values**: the function name, then each
argument.

1. **Vocabulary** (`vocab.py`): `vocab.json` stores tokens in byte-level BPE
   form (`Ġ` is a space). Every token is converted back to the bytes it
   produces, so the grammar knows its exact text.
2. **Grammar** (`grammar.py`): each value has a kind (number, integer,
   string, boolean, or enum for the name) and a suffix (the fixed text after
   it, e.g. `,"b":`). A token is allowed if, after adding it, the value can
   still be completed, or the value is complete and the rest of the token is
   the start of the suffix. This handles tokens like `",` that close a string
   and add a comma at once. Every accepted prefix can be completed, so
   generation never gets stuck.
3. **Decoding** (`decoder.py`), for each token:
   get the logits with `get_logits_from_input_ids`, set all disallowed
   tokens to `-inf`, take the `argmax`, append it, repeat.
4. **Function call** (`caller.py`, `prompt.py`): the prompt lists the
   functions with their descriptions. The name is generated as an enum of
   the defined names plus `None`, then each parameter is generated under
   the rules for its type, and the result is parsed and validated with
   pydantic.

## Design decisions

- **The grammar handles structure, the model handles meaning.** The output
  is valid whatever the prompt says. The prompt only improves the choices.
- **Compact JSON (no spaces).** With `"b": ` the model dropped the minus in
  `-3` and wrote `" *"` instead of `"*"`. Without spaces both were correct.
- **Check every token at every step** (about 0.085 s). Grouping tokens a little faster but much more complex.
- **Greedy decoding:** the same input always gives the same output.
- **Unmatched requests:** the model can choose `None`. Such requests, and
  empty ones, are reported and left out of the output, since a placeholder
  name would break the schema.
- **Numbers are kept as written** (`2`, `1.5`, `3e5`). The subject shows both
  `2` and `2.0`, and both are JSON numbers. Large integers stay exact.
- **Excluded tokens:** fragments of UTF-8 characters and tokens with control
  characters are never generated. `\uXXXX` escapes are not supported.
- **Thinking disabled** with an empty `<think></think>` block, following the
  model's `chat_template`, written by hand because `apply_chat_template`
  needs the SDK's private tokenizer.
- **Errors:** files are validated with pydantic before the model loads.
  Every error becomes one clear message and exit code 1.
- **Names:** the SDK method is `get_path_to_vocab_file()` (the subject says
  `get_path_to_vocabulary_json()`), and the provided definitions file is
  `functions_definition.json`.

## Performance analysis

| Metric | Result |
|---|---|
| Valid JSON | 100%, by construction |
| Provided prompts | 11/11 correct |
| Extra prompts (all types, escapes, Unicode, large numbers, 0 and 5 parameters) | all correct |
| Time, 11 prompts | 100 to 150 s (Intel Core Ultra 5, CPU only) |

The SDK has no KV cache, so each model call re-reads the whole sequence
(0.25 s for 50 tokens, 0.8 s for 300). Fixed text costs no model call, and
the prompt is kept short (about 180 tokens).

## Challenges faced

- **Disk space:** PyTorch filled the home quota, so the caches were moved
  to `goinfre`.
- **Tokens cross value boundaries** (`",`, `"}`), which led to the
  character-level "value plus suffix" rule.
- **Byte-level tokens:** 1,448 tokens are fragments of UTF-8 characters.
- **Prompt wording:** "copy values exactly" made the model use `NUMBERS` as
  a regex. Removing it gave `\d+`.
- **Scientific notation:** `3e5` became `30000` until exponents were
  allowed, after: 50/50.
- **mypy** confused the `llm_sdk/` folder with the package, fixed with
  `follow_imports = "skip"`.

## Testing strategy

Test scripts are not submitted.

- **Vocabulary:** exact round trip of `Greet José 😀 and say "hi"\n`.
- **Grammar:** assertions for each kind (prefixes, endings, escapes, leading
  zeros, exponents, length limit), without the model.
- **End to end:** the provided prompts plus edge cases (all four types, 0 and
  5 parameters, quotes, backslashes, empty strings, emoji, long strings,
  huge numbers, unmatched requests, prompt injection), checked by hand.
- **Errors:** missing, empty, broken, binary and unreadable files, wrong
  structure, unknown types, duplicate names, unwritable output, Ctrl+C. Each gives one clear message, no traceback.
- `make lint` after every change.

**Known limitations:** strings longer than about 200 characters are cut off,
requests close to an existing function may be mapped to it, and notations
like `1.5e-3` may be misread.

## Resources

- [pydantic](https://docs.pydantic.dev/), [NumPy](https://numpy.org/doc/),
  [uv](https://docs.astral.sh/uv/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) and its
  `chat_template`
- [Hugging Face: chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Hugging Face: environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- [uv: environment variables](https://docs.astral.sh/uv/reference/environment/)
- [RFC 8259: JSON](https://www.rfc-editor.org/rfc/rfc8259)
- [GPT-2 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py):
  byte-level BPE table

**Use of AI:** Claude (Anthropic) was used for: explaining the concepts, creating test cases, and drafting docstrings and this README.
