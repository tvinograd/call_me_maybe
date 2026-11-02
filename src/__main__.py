"""Entry point: python -m src."""
import argparse
import sys
import time
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .caller import FunctionCaller
from .decoder import ConstrainedDecoder
from .utils import load_functions, load_prompts, save_results
from .models import FunctionCallResult
from .vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function "
                    "calls with constrained decoding.")
    parser.add_argument(
        "--functions_definition", type=Path,
        default=Path("data/input/functions_definition.json"),
        help="JSON file with the available functions")
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="JSON file with the prompts to process")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="JSON file to write the function calls to")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """Load the inputs, generate every function call, save the results.

    Args:
        args: The parsed command-line options.

    Raises:
        ValueError: If an input file is invalid or the output cannot be
            written.
        RuntimeError: If the model cannot be loaded.
    """
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")

    try:
        llm = Small_LLM_Model()
        vocabulary = Vocabulary.from_file(llm.get_path_to_vocab_file())
    except Exception as error:
        raise RuntimeError(f"Cannot load the model: {error}") from None
    caller = FunctionCaller(
        decoder=ConstrainedDecoder.build(llm, vocabulary),
        functions=functions
    )

    results: list[FunctionCallResult] = []
    start = time.perf_counter()
    for number, item in enumerate(prompts, start=1):
        try:
            result = caller.call(item.prompt)
        except Exception as error:
            print(f"[{number}/{len(prompts)}] skipped {item.prompt!r}: "
                  f"{error}", file=sys.stderr)
            continue
        results.append(result)
        print(f"[{number}/{len(prompts)}] {item.prompt!r} -> {result.name}")

    save_results(args.output, results)
    print(f"Wrote {len(results)} results to {args.output} "
          f"in {time.perf_counter() - start:.1f} second(s).")


def main() -> int:
    """Run the program and turn errors into messages.

    Returns:
        The exit code: 0 on success, 1 on error, 130 if interrupted.
    """
    args = parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
