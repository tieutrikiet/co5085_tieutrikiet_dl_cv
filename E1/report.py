"""Export the "Nhận xét" section as a text report.

Numbers (Table 1, gaps, confused pairs, confident errors) are computed from the
current run; the written commentary below them is the analysis of the seed-42 run.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import CLASS_NAMES, CONFIG

# Class pairs (true -> predicted) discussed in the commentary
KEY_PAIRS = [
    ("Coat", "Pullover"),
    ("Shirt", "T-shirt/top"),
    ("T-shirt/top", "Shirt"),
    ("Shirt", "Pullover"),
    ("Shirt", "Coat"),
]

COMMENTARY = """\
### Dựa theo Learning Curves

#### Softmax
- Underfit, đường train và val gần như trùng nhau. Ở loss thì dao động quanh 0.4, còn ở accuracy thì chững lại
  ở khoảng 86% chỉ sau vài epoch đầu (sau đó tiếp tục đi ngang).
- Train thêm cũng không cải thiện.

#### MLP
- Overfit, nhìn vào đồ thị loss, val loss đạt đáy rồi đi lên trong khi train loss vẫn tiếp tục giảm. Hai đường
  tách xa nhau dần, val loss thấp nhất khoảng 0.295 sau đó tăng, trong khi train loss giảm đều từ 0.3 -> 0.16.
- Nhìn qua đồ thị accuracy, cho thấy hậu quả của overfit, val acc đứng yên trong khi train acc tiếp tục tăng
  tạo ra khoảng cách. Ở thời điểm dừng, khoảng cách train-val accuracy (đo ở chế độ eval) là khoảng 4.9 điểm.

#### CNN
- Overfit. Hai đồ thị loss/accuracy trong learning curves của CNN tương tự với MLP. Thậm chí, so với MLP thì CNN
  còn overfit mạnh hơn một chút do train loss giảm sâu hơn, val loss tăng nhiều hơn tính theo tỉ lệ và khoảng
  cách train-val lớn hơn, do không có `BatchNorm`.
- Cả MLP và CNN đều overfit sau epoch 11~12, nhưng early stopping chọn checkpoint tốt nhất theo val accuracy nên
  hạn chế được tác hại. CNN overfit tương đương hoặc mạnh hơn MLP nhưng vẫn khái quát tốt hơn vì có inductive
  bias phù hợp với ảnh.
- So với MLP, CNN có tốc độ hội tụ nhanh hơn, đạt val accuracy lớn nhất (~93%) ở epoch 21, trong khi MLP đạt lớn
  nhất chỉ có 90.26% nhưng ở epoch 22.

### Dựa theo Confusion Matrix
- Dựa theo đường chéo (recall), tức là nhãn được dự đoán đúng thì có thể thấy được CNN > MLP > Softmax.
- Trong đó, các cặp lớp dễ nhầm nhiều nhất:
  - Coat -> Pullover: 13% ở Softmax, 12% ở MLP tức là tốt hơn Softmax nhưng không đáng kể, và 6% ở CNN tức là
    với CNN thì tỉ lệ đoán nhầm cặp lớp này đã giảm một nửa.
  - Shirt -> T-shirt/top: 14% -> 11% -> 12% và T-shirt -> Shirt là 8 -> 10 -> 7% ở Softmax, MLP và CNN. Điều này
    cho thấy các cặp lớp này ở cả ba bộ phân loại đều không có quá nhiều sự khác biệt, tức là đây là những ranh
    giới khó nhất của tập dữ liệu.
  - Shirt -> Pullover: 14% -> 9% -> 6%. Shirt -> Coat: 11% -> 5% -> 6%. Cho thấy ở các cặp này, MLP và CNN có
    nhiều điểm tương đồng và phân loại mạnh hơn bộ phân loại tuyến tính Softmax.

### Các ví dụ lỗi
- Trong các hình, thể hiện các lỗi tự tin nhất của các mô hình, không phải là các lỗi ngẫu nhiên. Với cả ba hình
  đều liệt kê 16 lỗi tự tin nhất tức là mô hình tự tin vào "khả năng dự đoán" nhưng kết quả là "sai".
- Trong đó, 16 lỗi của Softmax chỉ có một nửa (8 lỗi) là tự tin 100%, còn lại là 99%; với MLP và CNN đều hoàn toàn
  tự tin vào dự đoán của mình.
- Ở cả ba mô hình, có thể thấy được đều tự tin đoán ảnh thuộc lớp "Ankle Boot" nhưng nhãn thật sự là Sandal hoặc
  Sneaker. Với nhóm giày, Sandal <-> Sneaker là hai lớp tương đối nhập nhằng và tạo ra ranh giới khó trong tập
  dữ liệu.
- Hoặc nằm ở nhóm áo: Shirt <-> T-shirt/Coat, ở độ phân giải 28 x 28, khác biệt nằm ở cổ áo, tay áo và các chi
  tiết này "trông" khá mơ hồ và chỉ chiếm vài pixel ảnh.
"""


def _results_table(results: List[dict], outputs: Dict[str, tuple]) -> str:
    header = (f"{'Cls':<9}{'Params':>10}{'Best epoch':>12}{'Train time':>12}"
              f"{'Train acc':>11}{'Val acc':>10}{'Test acc':>10}{'Test error':>12}")
    lines = [header, "-" * len(header)]
    for r in results:
        y_true, y_pred, _ = outputs[r["model"]]
        lines.append(f"{r['model']:<9}{r['params']:>10,}{r['best_epoch']:>12}{r['train_time']:>12.1f}"
                     f"{r['train_acc']:>11.2%}{r['val_acc']:>10.2%}{r['test_acc']:>10.2%}"
                     f"{int(np.sum(y_pred != y_true)):>12,}")
    return "\n".join(lines)


def _learning_curve_stats(histories: Dict[str, dict], results: List[dict]) -> str:
    lines = []
    for r in results:
        h = histories[r["model"]]
        best_val_loss_ep = int(np.argmin(h["val_loss"]))
        lines.append(
            f"- {r['model']}: {len(h['val_loss'])} epochs run, best val acc {max(h['val_acc']):.2%} "
            f"at epoch {r['best_epoch']}; min val loss {h['val_loss'][best_val_loss_ep]:.3f} "
            f"at epoch {best_val_loss_ep + 1}; train loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.3f}; "
            f"train-val acc gap (eval mode, best ckpt) {(r['train_acc'] - r['val_acc']) * 100:.1f} pts")
    return "\n".join(lines)


def _confusion_stats(cms: Dict[str, np.ndarray]) -> str:
    names = list(cms)
    lines = ["Recall per model (mean of diagonal): "
             + ", ".join(f"{n} {np.diag(cms[n]).mean():.2%}" for n in names),
             "",
             f"{'True -> Predicted':<28}" + "".join(f"{n:>10}" for n in names)]
    for true_name, pred_name in KEY_PAIRS:
        i, j = CLASS_NAMES.index(true_name), CLASS_NAMES.index(pred_name)
        lines.append(f"{true_name + ' -> ' + pred_name:<28}"
                     + "".join(f"{cms[n][i, j]:>10.0%}" for n in names))
    return "\n".join(lines)


def _confident_errors(outputs: Dict[str, tuple], n: int = 16) -> str:
    lines = []
    for name, (y_true, y_pred, probs) in outputs.items():
        wrong = np.flatnonzero(y_pred != y_true)
        conf = np.sort(probs[wrong, y_pred[wrong]])[::-1][:n]
        lines.append(f"- {name}: top-{len(conf)} confident errors, "
                     f"{int(np.sum(conf >= 0.995))} at ~100% confidence, min {conf.min():.1%}")
    return "\n".join(lines)


def export_report(path: Path, results: List[dict], histories: Dict[str, dict],
                  outputs: Dict[str, tuple], cms: Dict[str, np.ndarray]) -> Path:
    sections = [
        "## Nhận xét",
        f"(generated {datetime.now():%Y-%m-%d %H:%M}, seed={CONFIG['seed']}, "
        f"epochs={CONFIG['epochs']}, patience={CONFIG['patience']})",
        "Table 1. Bảng kết quả chạy của ba bộ phân loại",
        _results_table(results, outputs),
        "### Số liệu Learning Curves (run hiện tại)",
        _learning_curve_stats(histories, results),
        "### Số liệu Confusion Matrix (run hiện tại, % theo hàng)",
        _confusion_stats(cms),
        "### Số liệu lỗi tự tin (run hiện tại)",
        _confident_errors(outputs),
        "-" * 80,
        "Nhận xét chi tiết (phân tích cho run seed 42):",
        COMMENTARY,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path
