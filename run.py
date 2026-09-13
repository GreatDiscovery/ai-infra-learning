"""跨平台章节入口；从任意工作目录运行都使用仓库内路径。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "00": "chapters/00-tensors/demo.py",
    "01": "chapters/01-classifier/train.py",
    "predict": "chapters/01-classifier/predict.py",
    "04": "chapters/04-performance/lab.py",
}
if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
    print("用法：python run.py {00|01|predict|04} [章节参数]")
    raise SystemExit(2)
raise SystemExit(subprocess.call(
    [sys.executable, str(ROOT / SCRIPTS[sys.argv[1]]), *sys.argv[2:]], cwd=ROOT))
