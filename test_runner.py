from src.prompt_runner import PromptRunner

runner = PromptRunner()
out = runner.run("Reply with exactly this word and nothing else: positive", temperature=0.0, max_tokens=5)
print("MODEL OUTPUT:", repr(out))