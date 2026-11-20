
from prompt_injection_checker import PromptInjectionChecker

checker = PromptInjectionChecker()

with open("poison_test.txt", "r", encoding="utf-8") as file:
    content = file.read()

result = checker.check_text(content)

print("\nDocument Security Test")
print("-" * 40)
print("Suspicious:", result["is_suspicious"])
print("Matched patterns:")

for pattern in result["matched_patterns"]:
    print("-", pattern)

if result["is_suspicious"]:
    print("\nPASS: Malicious instructions detected.")
else:
    print("\nFAIL: Malicious instructions were not detected.")
