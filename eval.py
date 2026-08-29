import os
import json
import argparse
import models
import benchmarks
import baselines

def load_model(args):
    # Initialize the base model wrapper
    if args.model == "llava":
        model = models.llava.LlavaModel(model_path=args.model_path, device=args.device)
    elif args.model == "qwen_vl":
        model = models.qwen_vl.QwenVLModel(model_path=args.model_path, device=args.device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Apply baseline wrappers / patchers (e.g., FastV)
    if args.baseline is not None:
        model = baselines.build_baseline(args.baseline, model=model)

    return model

def load_benchmark(args):
    if args.benchmark == "MMBench":
        dataset_cls = benchmarks.MMBench.MMBenchDataset
    elif args.benchmark == "MMMU":
        dataset_cls = benchmarks.MMMU.MMMUDataset
    elif args.benchmark == "MMMUPro":
        dataset_cls = benchmarks.MMMUPro.MMMUProDataset
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    # data_path is a HuggingFace dataset ID (or a local dataset directory);
    # each dataset class carries its own default, so only override when asked.
    kwargs = {}
    if args.data_path:
        kwargs["data_path"] = args.data_path
    if args.split:
        kwargs["split"] = args.split
    if args.config_name:
        kwargs["config_name"] = args.config_name

    return dataset_cls(**kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Benchmark Evaluation")

    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture to evaluate")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace ID for the model checkpoint")

    parser.add_argument("--benchmark", type=str, required=True,
                        help="Benchmark dataset to run")

    parser.add_argument("--data_path", type=str, default=None,
                        help="HuggingFace dataset ID or local dataset directory "
                             "(defaults to the benchmark's own dataset ID)")

    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (defaults to the benchmark's own split)")

    parser.add_argument("--config_name", type=str, default=None,
                        help="Dataset config/subset name (defaults to the benchmark's own config)")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Cache directory for downloaded datasets (sets HF_DATASETS_CACHE)")

    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N samples")

    parser.add_argument("--baseline", type=str, default=None, choices=["FastV", None],
                        help="Baseline or acceleration technique to apply")

    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save evaluation results")

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_dir:
        os.makedirs(args.data_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = args.data_dir

    print(f"Loading model '{args.model}'...")
    model = load_model(args)

    print(f"Loading benchmark '{args.benchmark}'...")
    dataset = load_benchmark(args)

    print("Starting evaluation...")
    evaluation = dataset.run_evaluation(model, limit=args.limit)
    print(f"Evaluation Metrics: {evaluation['summary']}")

    # Save results and configuration
    output_filename = f"{args.model}_{args.benchmark}" + (f"_{args.baseline}" if args.baseline else "")
    out_file = os.path.join(args.output_dir, f"{output_filename}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "metrics": evaluation["summary"],
            "results": evaluation["details"]
        }, f, indent=4)

    print(f"Results successfully saved to {out_file}")


if __name__ == "__main__":
    main()
