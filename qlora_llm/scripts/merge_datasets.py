import json
import os
import argparse
import random
from rich.console import Console

console = Console()

def merge_jsonl(input_files, output_file, shuffle=True, seed=42):
    """Merge multiple JSONL files into one, optionally shuffling."""
    all_data = []
    
    for file in input_files:
        if not os.path.exists(file):
            console.print(f"[bold red]❌ File not found:[/] {file}")
            continue
            
        console.print(f"📖 Reading [cyan]{file}[/]...")
        count = 0
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                    count += 1
                except:
                    continue
        console.print(f"   Collected {count} samples.")

    if shuffle:
        console.print(f"🔀 Shuffling {len(all_data)} samples (seed={seed})...")
        random.seed(seed)
        random.shuffle(all_data)

    console.print(f"💾 Writing to [bold green]{output_file}[/]...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(f"\n✅ [bold green]Successfully merged {len(all_data)} samples into {output_file}![/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSONL datasets into one.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    parser.add_argument("--output", default="data/train.jsonl", help="Output JSONL file")
    parser.add_argument("--no_shuffle", action="store_true", help="Don't shuffle the merged dataset")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    
    args = parser.parse_args()
    merge_jsonl(args.inputs, args.output, shuffle=not args.no_shuffle, seed=args.seed)
