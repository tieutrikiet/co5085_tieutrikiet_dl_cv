"""Export phần "Evaluation" của notebook E3 thành báo cáo text.

Số liệu (bảng kết quả, McNemar, recall) được tính lại từ run hiện tại;
phần nhận xét viết tay bên dưới mỗi bảng là phân tích cho run seed 42 trong notebook.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

from common.evaluation import format_results_table
from E1.report import KEY_PAIRS

from .config import CLASS_NAMES, CONFIG

COMMENTARY_TABLE = """\
#### Theo cách biểu diễn

- row tốt nhất trong ba cách biến ảnh thành chuỗi (row | col | patch), với test_acc là 89.8%~89.9%, col kém hơn khoảng 0.7 điểm, patch thì kém hơn khoảng 1.4~1.7 điểm so với row. Thứ tự này giống nhau ở cả LSTM và GRU.
- row và col có cùng T, D và số tham số tham gia mô hình, nên chênh lệch này giữa chúng đến từ hướng quét. Đọc ảnh từ tren xuống dưới (row) có lợi hơn đọc từ trái sang phải, vì hình dạng quần áo thay đổi theo chiều dọc rõ hơn.

#### LSTM vs GRU

- Accuracy gần như bằng nhau trên mọi cách biểu diễn.
- GRU ít tham số hơn LSTM 1/4 và dừng sớm hơn LSTM.

#### So với Baseline

- LSTM và GRU có accuracy ngang với MLP nhưng dùng ít tham số hơn rất nhiều.
- Tất cả RNN đều kém hơn CNN khoảng 2.4, 2.5 điểm trở lên đến khoảng 4.5 điểm.

#### Thời gian train

- Patch có thời gian train nhanh nhất trong số tất cả, vì chỉ có T=16 bước tuần từ, so với T=28 của row/col.
- Xét về độ trễ (latency), các mô hình RNN rơi vào khoảng 0.14~0.27ms, gần tương đương với CNN 0.13ms, trong khi đó MLP nhanh nhất chỉ với 0.02ms vì chỉ là vài phép nhân ma trận.

#### Gap

- Gap là khoảng cách giữa train và val.
- Trong đó, patch có gap lớn nhất khoảng 5.9 điểm, nên patch overfit nhiều nhất.
- GRU row có gap nhỏ nhất là 2.2
- MLP có gap = 6.0, do nó sử dụng rất nhiều tham số.
"""

COMMENTARY_MCNEMAR = """\
#### best_rnn

- Model "tốt nhất" được chọn theo val accuracy, trong đó `lstm_row` có acc đạt 90.71%, `gru_row` chỉ đạt 90.53%.
- Dù trên test thì `gru_row` nhỉnh hơn `lstm_row` với điểm số 89.91% so với 89.82%, tuy nhiên ta không được chọn model theo test, làm vậy là dùng tập test để ra quyết định, trong khi con số test báo cáo sẽ bị lạc quan.

#### Kết luận:

- LSTM và GRU: có p > 0.4 ở cả ba cách biểu diễn, lớn hơn nhiều so với 0.05 (5%), tức là không có khác biệt có ý nghĩa thống kê.
- RNN tốt nhất (`lstm_row`) và MLP: có p≈0.15, chênh lệch 0.17 điểm, chưa đủ kết luận để nói rằng RNN tốt hơn.
- RNN tốt nhất (`lstm_row`) và CNN: có p<0.001, CNN đúng ở 505 ảnh mà RNN sai, trong khi ngược lại RNN chỉ đúng khoảng 263, tức là tỉ lệ CNN đúng gấp đôi RNN đúng. Điều này cho thấy CNN tốt hơn là có ý nghĩa thống kê.
"""

COMMENTARY_CONFUSION = """\
#### Nhận xét

- RNN (lstm_row) mạnh hơn MLP ở nhận dạng đúng Pullover (85 > 76). Nhưng kém hơn ở nhận dạng Coat (85 < 88) và T-Shirt (83 < 85).
- Trong khi đó, tỉ lệ coat bị nhầm thành pullover là 7%, ở MLP là 4%.
- Tổng accuracy của hai model (RNN_lstm_row và MLP) gần bằng nhau, chúng sai ở những chỗ khác nhau, điều này phản ánh lại kiểm định McNemar: hai model bất đồng ở khoảng 9% ảnh.
- Nhìn chung, thì CNN vẫn là model nhận dạng tốt hơn, có các chỉ số cao đồng đều hơn so với hai mô hình còn lại.
- Shirt vẫn là lớp khó nhất (71–73%) ở mọi model.
"""


# ---------------------------------------------------------------- tables (cũng được main in ra console)

def summary_table(results_by_name: Dict[str, dict], names: Sequence[str]) -> str:
    lines = [format_results_table([results_by_name[n] for n in names]),
             "*" * 80,
             f"\n{'model':<12}{'T':>4}{'D':>5}{'epochs':>8}{'s/ep':>7}{'lat(ms)':>9}{'gap':>6}"]
    for n in names:
        r = results_by_name[n]
        lines.append(f"{n:<12}{r['num_step']:>4}{r['input_size']:>5}{r['epochs_run']:>8}"
                     f"{r['sec_per_epoch']:>7.1f}{r['latency_ms']:>9.2f}{(r['train_acc'] - r['val_acc']) * 100:>6.1f}")
    return "\n".join(lines)


def mcnemar(outputs, a, b):
    y, p_a, _ = outputs[a]
    _, p_b, _ = outputs[b]
    n_a = int(np.sum((p_a == y) & (p_b != y)))   # a đúng, b sai
    n_b = int(np.sum((p_a != y) & (p_b == y)))   # a sai, b đúng
    pval = binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    return (p_a == p_b).mean(), n_a, n_b, pval


def mcnemar_table(outputs: Dict[str, tuple], results_by_name: Dict[str, dict], best_rnn: str,
                  pairs: Sequence[Tuple[str, str]]) -> str:
    lines = [f"best RNN (theo val acc): {best_rnn}",
             f"{'A':<12}{'B':<12}{'acc A':>8}{'acc B':>8}{'agree':>8}{'A>B':>6}{'B>A':>6}{'p':>8}"]
    for a, b in pairs:
        agree, n_a, n_b, pval = mcnemar(outputs, a, b)
        lines.append(f"{a:<12}{b:<12}{results_by_name[a]['val_acc']:>8.2%}{results_by_name[b]['val_acc']:>8.2%}"
                     f"{agree:>8.2%}{n_a:>6}{n_b:>6}{pval:>8.3f}")
    return "\n".join(lines)


def recall_table(cms: Dict[str, np.ndarray]) -> str:
    lines = [f"{'Recall (%)':<26}" + "".join(f"{n:>12}" for n in cms)]
    for k, cls_name in enumerate(CLASS_NAMES):
        lines.append(f"{cls_name:<26}" + "".join(f"{cms[n][k, k]:>12.1%}" for n in cms))
    for true_name, pred_name in KEY_PAIRS:
        i, j = CLASS_NAMES.index(true_name), CLASS_NAMES.index(pred_name)
        lines.append(f"{true_name + ' -> ' + pred_name:<26}" + "".join(f"{cms[n][i, j]:>12.0%}" for n in cms))
    return "\n".join(lines)


def _learning_curve_stats(histories: Dict[str, dict], results_by_name: Dict[str, dict], names: Sequence[str]) -> str:
    lines = []
    for n in names:
        h, r = histories[n], results_by_name[n]
        best_val_loss_ep = int(np.argmin(h["val_loss"]))
        lines.append(
            f"- {n}: {len(h['val_loss'])} epochs run, best val acc {max(h['val_acc']):.2%} "
            f"at epoch {r['best_epoch']}; min val loss {h['val_loss'][best_val_loss_ep]:.3f} "
            f"at epoch {best_val_loss_ep + 1}; train loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.3f}")
    return "\n".join(lines)


def _figures(names: Sequence[str]) -> str:
    return "\n".join(f"- figures/{n}.png" for n in names)


def export_report(path: Path, results_by_name: Dict[str, dict], histories: Dict[str, dict],
                  outputs: Dict[str, tuple], cms: Dict[str, np.ndarray], names: Sequence[str],
                  rnn_types: Sequence[str], sequences: Sequence[str], best_rnn: str,
                  pairs: Sequence[Tuple[str, str]]) -> Path:
    sections = [
        "## Evaluation",
        f"(generated {datetime.now():%Y-%m-%d %H:%M}, seed={CONFIG['seed']}, "
        f"epochs={CONFIG['epochs']}, patience={CONFIG['patience']})\n"
        "Số liệu trong các bảng được tính từ run hiện tại; phần nhận xét là phân tích cho run seed 42 trong notebook.",

        "### 1. Bảng kết quả: LSTM / GRU × row / col / patch và baseline E1",
        summary_table(results_by_name, names),
        COMMENTARY_TABLE,

        "### 2. Learning curves",
        _figures([f"E3_curves_{s}_lstm_vs_gru" for s in sequences] + [f"E3_curves_{r}_sequences" for r in rnn_types]),
        "Số liệu learning curves (run hiện tại):\n" + _learning_curve_stats(histories, results_by_name, names),

        "### 3. Kiểm định McNemar",
        mcnemar_table(outputs, results_by_name, best_rnn, pairs),
        COMMENTARY_MCNEMAR,

        "### 4. Confusion matrix, recall theo lớp và các lỗi tự tin nhất (% theo hàng)",
        recall_table(cms),
        _figures(["E3_confusion_matrices", f"E3_misclassified_{best_rnn}"]),
        COMMENTARY_CONFUSION,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path
