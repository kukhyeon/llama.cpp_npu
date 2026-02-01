import json
import subprocess
import re
import csv
import time
import os

# 현재 스크립트 위치를 기준으로 프로젝트 루트 경로 계산
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/hotpot_qa_20.json")
MODEL = "Llama-3.2-1B-Instruct-Q4_0.gguf"
LOG_FILE = os.path.join(PROJECT_ROOT, "result/hotpot_results.csv")
RUN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts/snapdragon/adb/run-completion.sh")

def run_llama(prompt):
    custom_env = os.environ.copy()
    custom_env["M"] = MODEL
    custom_env["D"] = "HTP0"

    cmd = ["sh", RUN_SCRIPT, "-p", prompt, "-n", "256", "--repeat-penalty", "1.1"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=custom_env, cwd=PROJECT_ROOT)
    return result.stdout + result.stderr

with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["num", "question", "prefill_ms", "decode_ms"])

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
    questions = data.get("questions", [])

for i, question in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] Processing: {question[:50]}...")
    output = run_llama(question)
    prefill_match = re.search(r"prompt eval time =\s+([\d.]+)\s+ms", output)
    decode_match = re.search(r"eval time =\s+([\d.]+)\s+ms", output)
    prefill_ms = prefill_match.group(1) if prefill_match else "N/A"
    decode_ms = decode_match.group(1) if decode_match else "N/A"
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([i+1, question, prefill_ms, decode_ms])
    print(f"   => Prefill: {prefill_ms} ms | Decode: {decode_ms} ms")
    time.sleep(3)

print(f"\n모든 작업 완료! 결과 파일: {LOG_FILE}")
