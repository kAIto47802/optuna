from pathlib import Path

log_file = Path("journal_storage.log").read_text()

nums = [1 << i for i in range(1, 20)]
for num in nums:
    Path(f"journal_storage{num}.log").write_text(
    "\n".join(log_file.splitlines()[:num])
    )