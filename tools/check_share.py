"""提交前辅助检查。检查暂存版本及对应工作文件，不上传任何数据。"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAME = "AI Infra Learner"
EXPECTED_EMAIL = "learner@users.noreply.github.com"
ALLOWED_ROOTS = {"chapters", "tools"}
ALLOWED_FILES = {".gitignore", "AGENTS.md", "README.md", "PROGRESS.md", "ROADMAP.md",
                 "PRIVACY.md", "VALIDATION.md", "requirements.txt", "requirements-tested.txt", "run.py"}
PATTERNS = {
    "本机用户目录": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+|[A-Z]:\\Users\\[^\\\s]+"),
    "邮箱": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "访问令牌": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})"),
    "私钥": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "手机号样式": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
}


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=check)


def inspect(name, data):
    errors = []
    if len(data) > 300_000:
        return [f"{name}：文件超出教学源码大小上限"]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{name}：非 UTF-8 文本，不在分享范围"]
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if label == "邮箱" and match.group() == EXPECTED_EMAIL:
                continue
            line = text[:match.start()].count("\n") + 1
            errors.append(f"{name}:{line}：疑似{label}，请本地审阅")
    return errors


def main():
    errors = []
    top = git("rev-parse", "--show-toplevel").stdout.decode().strip()
    if Path(top).resolve() != ROOT:
        raise SystemExit("请先将当前学习项目初始化为独立 Git 仓库，不要操作上级目录。")
    for key, expected in [("user.name", EXPECTED_NAME), ("user.email", EXPECTED_EMAIL)]:
        actual = git("config", "--local", "--get", key, check=False).stdout.decode().strip()
        if actual != expected:
            errors.append(f"请按 README 设置本仓库 {key}，避免使用全局个人身份")
    paths = git("ls-files", "--cached", "-z").stdout.decode().split("\0")
    count = 0
    for name in filter(None, paths):
        path = Path(name)
        if (name not in ALLOWED_FILES and path.parts[0] not in ALLOWED_ROOTS) or (
            path.suffix not in {".py", ".md", ".txt"} and name != ".gitignore"
        ) or any(part.startswith(".") for part in path.parts if part != ".gitignore"):
            errors.append(f"{name}：不在允许分享的源码/文档范围")
            continue
        indexed = git("show", f":{name}").stdout
        errors.extend(inspect(name + "（暂存）", indexed))
        local = ROOT / name
        if local.is_symlink():
            errors.append(f"{name}：不允许符号链接")
        elif local.is_file() and local.read_bytes() != indexed:
            errors.extend(inspect(name + "（工作区）", local.read_bytes()))
        count += 1
    if not count:
        errors.append("没有已跟踪/暂存的文件；请先显式 git add 要分享的文件")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"通过：检查 {count} 个文件及仓库提交身份。仍需人工阅读 git diff --cached。")


if __name__ == "__main__":
    main()
