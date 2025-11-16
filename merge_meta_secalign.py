# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from huggingface_hub import login

# Check if HF_TOKEN is set, if not prompt for login
if not os.environ.get("HF_TOKEN"):
    print("⚠️  HuggingFace token not found in environment.")
    print("Please authenticate:")
    print("  Option 1: Run: huggingface-cli login")
    print("  Option 2: Set: export HF_TOKEN='your_token'")
    print("  Option 3: Enter token below (will be used for this session only)")
    token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if token:
        login(token=token)
    else:
        print("⚠️  No token provided. Attempting to use cached credentials...")
        try:
            login()  # Try to use cached credentials
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("\nPlease authenticate first:")
            print("  huggingface-cli login")
            exit(1)
else:
    login(token=os.environ["HF_TOKEN"])
    
# Load base model
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
adapter_name = "facebook/Meta-SecAlign-8B"
save_path = "checkpoints/Meta-SecAlign-8B-merged"

# base_model_name = "meta-llama/Llama-3.3-70B-Instruct"
# adapter_name = "facebook/Meta-SecAlign-70B"
# save_path = "checkpoints/Meta-SecAlign-8B-merged"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name, device_map="auto", torch_dtype="auto"
)

# Load adapter
model = PeftModel.from_pretrained(model, adapter_name)

# Merge adapter into base model weights
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(save_path)

# Also save tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_name)
tokenizer.save_pretrained(save_path)
