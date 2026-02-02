import json
import subprocess
import re
import csv
import time
import os

# --- Path Configuration ---
PROJECT_ROOT = "/workspace"
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/hotpot_qa_30.json")
MODEL = "Llama-3.2-1B-Instruct-Q4_0.gguf"

# Output files
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
LOG_FILE = os.path.join(RESULT_DIR, "hotpot_results.csv")
RAW_LOG_FILE = os.path.join(RESULT_DIR, "raw_terminal_output.log")
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts/snapdragon/adb/run-completion.sh")

# --- Temporary File Path for ADB ---
DEVICE_PROMPT_PATH = "/data/local/tmp/prompt.txt"
LOCAL_TEMP_PROMPT = os.path.join(PROJECT_ROOT, "temp_prompt.txt")

def run_llama(prompt):
    """
    Saves the prompt to a file, pushes it to the device via ADB, 
    and runs the inference script.
    """
    # 1. Save query to a local temp file
    with open(LOCAL_TEMP_PROMPT, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    # 2. Push the file to the Android device via ADB
    subprocess.run(["adb", "push", LOCAL_TEMP_PROMPT, DEVICE_PROMPT_PATH], 
                   capture_output=True, text=True)

    # 3. Setup environment variables for the Snapdragon toolchain
    custom_env = os.environ.copy()
    custom_env["M"] = MODEL
    custom_env["D"] = "HTP0"

    # 4. Execute inference using file input (-f) to avoid shell quoting issues
    cmd = ["sh", RUN_SCRIPT, "-f", DEVICE_PROMPT_PATH, "-n", "256"]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=custom_env, cwd=PROJECT_ROOT)
    return (result.stdout or "") + (result.stderr or "")

# Ensure the result directory exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Initialize CSV file with the 4 requested metrics
with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["num", "prefill_ms", "prefill_tps", "decode_ms", "decode_tps"])

# Clear previous raw log file
with open(RAW_LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("=== Snapdragon NPU Inference Raw Logs ===\n")

# Load Dataset
try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        questions = data.get("questions", [])
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}")
    exit()

print(f"Starting NPU Benchmark (Prefill & Decode metrics)...")

for i, question in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] Processing inference...", end="\r")
    
    output = run_llama(question)

    # --- Save Raw Terminal Output for review ---
    with open(RAW_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- QUESTION {i+1} ---\n")
        f.write(f"PROMPT: {question}\n")
        f.write("-" * 30 + "\n")
        f.write(output) 
        f.write("\n" + "=" * 50 + "\n")

    # --- Data Parsing (Advanced Regex) ---
    
    # 1. Extract Prefill (Prompt Eval) metrics
    prefill_ms = "N/A"
    prefill_tps = "N/A"
    # Matches: prompt eval time = XX.XX ms / XX tokens ( XX.XX ms per token, XX.XX tokens per second)
    p_match = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms.*?([\d.]+)\s*tokens per second", output)
    if p_match:
        prefill_ms = p_match.group(1)
        prefill_tps = p_match.group(2)

    # 2. Extract Decode (Eval) metrics
    decode_ms = "N/A"
    decode_tps = "N/A"
    # Uses negative lookbehind to avoid matching 'prompt eval'
    d_match = re.search(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms.*?([\d.]+)\s*tokens per second", output)
    if d_match:
        decode_ms = d_match.group(1)
        decode_tps = d_match.group(2)

    # Error checking
    if prefill_ms == "N/A" or decode_ms == "N/A":
        print(f"\n[!] Question {i+1} parsing failed. Check logs.")

    # Save all 4 metrics to CSV
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([i+1, prefill_ms, prefill_tps, decode_ms, decode_tps])
    
    print(f"[{i+1}/{len(questions)}] Prefill: {prefill_tps} TPS | Decode: {decode_tps} TPS")
    
    # Wait to allow NPU to cool down
    time.sleep(1)

# Cleanup
if os.path.exists(LOCAL_TEMP_PROMPT):
    os.remove(LOCAL_TEMP_PROMPT)

print(f"\nBenchmark completed successfully!")
print(f"Summary: {LOG_FILE}")
print(f"Full Logs (including model answers): {RAW_LOG_FILE}")
