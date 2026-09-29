import argparse
import json
from pathlib import Path

from futurecad.descriptions import load_descriptions
from futurecad.prompts import prompt_sft
from futurecad.rewards.cad_reward import CODE_PATTERN
from futurecad.splits import split_ids


def extract_code(text):
    matches = CODE_PATTERN.findall(text)
    return matches[-1].strip() if matches else text


def main():
    parser = argparse.ArgumentParser('Generate CAD programs using a local Hugging Face checkpoint and vLLM.')
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--descriptions",
        help="Optional local ID-to-text JSON; defaults to Hugging Face",
    )
    parser.add_argument(
        "--description-level", choices=["overall", "detail"], default="detail"
    )
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    ids = split_ids(args.split_file, args.split)
    descriptions = load_descriptions(
        ids, args.split, args.description_level, args.descriptions
    )
    if args.n < 1:
        raise ValueError("--n must be positive")
    from vllm import LLM, SamplingParams

    engine_options = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        engine_options["max_model_len"] = args.max_model_len
    llm = LLM(**engine_options)
    conversations = [
        [{"role": "user", "content": prompt_sft(descriptions[i])}] for i in ids
    ]
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.chat(conversations, sampling_params=params, use_tqdm=True)
    codes, responses = {}, {}
    for data_id, output in zip(ids, outputs, strict=True):
        texts = [result.text.strip() for result in output.outputs]
        responses[data_id] = texts

        codes[data_id] = [extract_code(t) for t in texts]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(codes, indent=2), encoding="utf-8")
    target.with_suffix(".responses.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )
    target.with_suffix(".generation.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
