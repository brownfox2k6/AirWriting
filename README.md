# AirWriting

**Thiết kế và đánh giá cơ chế tương tác cho hệ thống viết chữ trên không bằng cử chỉ ngón tay**
*(Designing and Evaluating Interaction Mechanisms for a Finger-Based Air-Writing System)*

## Cài đặt môi trường (Python 3.12 / 3.13, CPU)

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Windows (PowerShell)
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install
```

Chạy kiểm tra cục bộ:

```bash
pre-commit run --all-files
pytest -v
```

> Mở thư mục bằng **VS Code** và cài các extension gợi ý trong `.vscode/extensions.json` (đã cấu hình sẵn `pytest` và tự format bằng `Ruff` khi lưu).

## Cấu trúc mã nguồn

```text
config/                  # Cấu hình tập trung (16 ký tự chuẩn & ngưỡng M1/M2/M3)
src/airwriting/
  core/                  # Camera, Active ROI, MediaPipe tracking, One Euro Filter, JSON trajectory
  interaction/           # State Machine (Buxton 0-1-2 + Commit) & 3 cơ chế M1, M2, M3
  ui/                    # PyQt6 views (Demo, Data Collector, Experiment Runner), feedback, earcons, XAI
  recognition/           # Baseline $P, dựng ảnh quỹ đạo màu, MobileNetV3-Small
  experiment/            # Bộ sinh lịch trình 6 hoán vị, auto-detector False Start/End, CSV & thống kê
tests/                   # Kiểm thử tự động (pytest)
```

## Quy trình làm việc theo GitHub Issues

1. Chọn issue chưa bị chặn (`Blocked by` đã đóng) trên tab **Issues** và tự gắn **Assignee**.
2. Tạo nhánh mới từ `main`: `git checkout -b feat/issue-<số>-<mô-tả-ngắn>`.
3. `pre-commit` tự kiểm tra và format (`pre-commit-hooks`, `ruff`, `mypy`) mỗi khi `git commit`.
4. Mở **Pull Request** vào `main`, ghi `Closes #<số>` trong mô tả PR — CI chỉ đọc sẽ kiểm tra lại `Ruff`, `mypy` và `pytest` trước khi cho phép merge.
