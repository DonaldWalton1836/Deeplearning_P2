import requests

# URL of the file
url = "https://www.cs.fsu.edu/~liux/courses/deepRL/assignments/word-test.v1.txt"

# Download the file
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to download file: {response.status_code}")

lines = response.text.strip().split('\n')

extracted_lines = []

# Parse lines
for line in lines:
    line = line.strip()
    if not line or line.startswith(":") or line.startswith("//"):
        continue
    tokens = line.split()
    if len(tokens) == 4:
        extracted_lines.append(" ".join(tokens))

# Save to output file
with open("capital_country_pairs.txt", "w", encoding="utf-8") as f:
    for line in extracted_lines:
        f.write(line + "\n")

print("Extraction complete. Saved to 'capital_country_pairs.txt'")
