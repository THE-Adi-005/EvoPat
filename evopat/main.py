import argparse
from src.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", type=str, required=True)
parser.add_argument("--query", type=str, required=True)

args = parser.parse_args()

result = run_pipeline(args.pdf, args.query)

print("\n===== FINAL OUTPUT =====\n")
print(result)