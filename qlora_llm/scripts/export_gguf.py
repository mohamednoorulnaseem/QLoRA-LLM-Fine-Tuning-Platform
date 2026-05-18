import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console

console = Console()

def merge_and_save(base_model_path: str, adapter_path: str, output_path: str):
    """
    Merges the LoRA adapter with the base model and saves the full model weights.
    This is the required first step before converting to GGUF.
    """
    console.print(f"Loading base model: [bold cyan]{base_model_path}[/]")
    
    # Load base model in 16-bit to avoid massive memory overhead while merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    console.print(f"Loading adapter: [bold cyan]{adapter_path}[/]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    console.print("Merging weights... (This might take a few minutes)")
    model = model.merge_and_unload()
    
    console.print(f"Saving merged model to: [bold green]{output_path}[/]")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    console.print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    console.print("\n[bold green]✅ Merge complete![/]")
    console.print("\n[bold yellow]To convert to GGUF, use llama.cpp:[/]")
    console.print(f"1. git clone https://github.com/ggerganov/llama.cpp")
    console.print(f"2. pip install -r llama.cpp/requirements.txt")
    console.print(f"3. python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile {output_path}/model.gguf --outtype q8_0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model for GGUF export.")
    parser.add_argument("--base_model", required=True, help="Path or HF Hub ID of the base model")
    parser.add_argument("--adapter_path", required=True, help="Path to the fine-tuned LoRA adapter")
    parser.add_argument("--output_path", default="models/merged-model", help="Where to save the merged weights")
    
    args = parser.parse_args()
    merge_and_save(args.base_model, args.adapter_path, args.output_path)
