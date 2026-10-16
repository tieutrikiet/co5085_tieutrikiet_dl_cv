"""Export phần "Evaluation" của notebook E2 thành báo cáo text.

Số liệu (bảng kết quả, McNemar, chi phí, recall, lỗi tự tin) được tính lại từ run hiện tại;
phần nhận xét viết tay bên dưới mỗi bảng là phân tích cho run seed 42 trong notebook.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import binomtest

from common.evaluation import format_results_table
from E1.report import KEY_PAIRS

from .config import CLASS_NAMES, CONFIG

COMMENTARY_TABLE = """\
Qua đó ta có thể nhận xét thành hai nhóm:

- Nhóm 1: so sánh với cùng tokenizer nhưng khác nhau ở **scratch** và **torch**:
  - Về số parameters, số lượng tham số của mô hình scratch và torch ở các tokenizer đều giống nhau, khoảng 72k cho patch7x7, 71k cho row chia thành 28 tokens, hoặc 93k cho CNN.
  - Nhưng có sự khác biệt đôi chút về best epoch, tức là epoch có val accuracy cao nhất, cũng là checkpoint được giữ lại.
  - Với tokenizer là patch 7x7 thì mô hình MSA scratch và torch đều không có sự khác nhau về số lượng tham số, best epoch, không có sự khác nhau quá lớn hoặc đáng kể ở thời gian training, train acc/val acc/test acc.
  - Với tokenizer là row 28px thì hai mô hình có sự khác nhau nhiều hơn và thấy rõ mô hình viết tay scratch sẽ có các giá trị lớn hơn mô hình native, như thông tin best epoch sẽ cho biết tốc độ hội tụ của mô hình native tốt hơn mô hình scratch khoảng 12 epoch, điều này dẫn tới thời gian training của scratch cũng sẽ lớn hơn torch vì early stopping, nếu nhìn vào test acc cuối cùng để đánh giá, thì mô hình torch nhỉnh hơn scratch 0.21 điểm, không quá lớn.
  - Với tokenizer là CNN, tương tự với cách so sánh ở Row, thì mô hình torch có kết quả kém hơn scratch một chút.

- Nhóm 2: so sánh với cùng loại mô hình nhưng khác tokenizer:
  - Với scratch, patch 7x7 cho test acc thấp nhất khi so với Row và CNN (88.01% < 89.26% < 91.13%), tuy best epoch của Row là 36 (cao nhất) và CNN là 15 (nhất nhất) nhưng cả hai mô hình có hai tokenizer này thì có thời gian training gần sát nhau (139s với 138s), điều này còn phản ánh một điều, tuy dừng sớm nhưng CNN với số lượng tham số nhiều hơn hẳn so với Row đồng thời đi qua hai lớp conv nên thời gian training network này chắc chắn phải chậm hơn các mô hình tokenizer khác.
  - Với torch, xét về test acc sau cùng thì CNN vẫn tốt hơn Row sau đó mới đến Patch, best epoch của ba mô hình có tokenizer này cũng có sai khác không lớn, tuy nhiên, với cùng lập luận trên, ở cùng số epoch thì CNN phải tiêu tốn thời gian training nhiều hơn hẳn các mô hình tokenizer còn lại.
"""

COMMENTARY_CURVES = """\
Qua ba learning curves của Patch, Row, CNN giữa scratch và torch ở trên, thì ta có thể thấy:
- Đường "val" của các tokenizer ở cả hai phiên bản đều gần như trùng nhau, xác thực kết quả số liệu trên bảng ở trước đó.
- Ở tokenizer Patch 7x7 thì chấm tròn (thể hiện cho early stopping) trùng nhau, còn ở Row thì torch dừng sớm hơn scratch, ngược lại với CNN, scratch dừng sớm hơn torch.
- Trong ba bộ tokenizer, CNN hội tụ nhanh nhất được dựa vào best epoch = 15 so với 25 của patch và 36 của row, đạt khoảng 88% (val cao nhất của patch) ở epoch thứ 3 và sau đó tiếp tục tăng lên đến điểm lớn nhất ở epoch 15. Nhìn qua loss, đường val loss xuống đến đáy ở epoch 15 sau đó tăng nhẹ (có dấu hiệu overfit vì trong khi train loss tiếp tục giảm) và bị cắt đúng lúc nhờ early stopping.
- Trong khi đó, Row thì có val loss gần như đi ngang trong khi train loss thì tiếp tục giảm, khoảng cách giữa train-val ngày càng lớn, cho thấy dấu hiệu overfit.
- Patch7 có train và val di chuyển đồng thời với nhau, độ lớn train-val không lớn, cả hai còn đi xuống chậm, đây là dấu hiệu của mô hình bị underfit hoặc hạn chế về capacity, chưa bị overfit -> nên còn khả năng để tiếp tục optimize và tuning để tăng khả năng học của mô hình.
"""

COMMENTARY_MCNEMAR = """\
Kiểm định **McNemar** lấy nhãn thật và dự đoán của từng model, bỏ qua probs vì kiểm định này chỉ cần biết đúng hay sai.
- `b`: số ảnh scratch đoán đúng nhưng torch sai.
- `c`: ngược lại.
- Những ảnh cả hai cùng đoán đúng, hoặc cùng sai thì không mang thông tin so sánh, nên bỏ qua.

**Giả thuyết H0**: hai model tốt như nhau, khi đó mỗi ảnh "bất đồng" có xác suất 50/50 rơi vào nhóm b hoặc nhóm c, tức là b tuân theo phân phối Binominal(b+c, 0.5)

Qua bảng kết quả, `p` cho xác suất thấy độ lệch giữa b và c lớn nếu hai model thật sự ngang nhau. Nếu p < 0.05 tức là 5% thì mới tạo ra sự khác biệt thống kê, qua đó, giá trị của p ở các tokenzier khác nhau giữa scratch và torch đều lớn hơn nhiều so với 0.05
-> Không có bằng chừng nào cho thấy MSA scratch và torch dùng `nn.MultiheadAttention` cho chất lượng khác nhau.
"""

COMMENTARY_COST = """\
Thời gian train mỗi epoch:

- patch7 mất 2.3 s, row 2.8–3.1 s, cnn 6.3–6.4 s.
- Chi phí tăng nhanh hơn số token. Từ N = 16 lên N = 49 (gấp khoảng 3 lần), thời gian tăng khoảng 2.8 lần. Lý do là ma trận attention có kích thước N × N, cộng thêm chi phí của các lớp conv trong stem.
- Ở cùng một tokenizer, scratch và torch gần như bằng nhau, chỉ lệch khoảng 0.1–0.3 s.

Về latency, ở cả ba tokenizer thì bản torch luôn có độ trễ thấp hơn scratch, tức là torch nhanh hơn khoảng 5-10% so với scratch.

So với baseline E1 (MLP và CNN):

- Không model nào vượt qua được CNN của E1
- CNN-stem của E2 là gần với CNN của E1 nhất, nhưng dùng khoảng 94k tham số thay vì dùng khoảng 422k tham số của CNN E1.
- Row với khoảng 71~72k tham số gần bằng với MLP của E1 với hơn 530k tham số.
"""

COMMENTARY_CONFUSION = """\
Qua ba confusion matrix ở cả ba tokenizer trên, ta có thể thấy:
- cả ba đều đoán đúng tốt ở các lớp (trouser, sandal, sneaker, bag, boot), đều ≥94%. Tạm gọi đây là các nhóm/lớp dễ, vì các lớp này có hình dạng tổng thể khác biệt rõ.
- với các lớp khác như t-shirt, pullover, coat thì ở mức 80%-90%, đặc biệt là lớp shirt (66%, 66%, 71%). Tạm gọi là các lớp khó, tất cả đều thuộc nhóm áo.
- trong lớp khó, ở các nhóm áo thì CNN cải thiện tốt nhất ở coat (80 -> 86 -> 91%), pullover (82 -> 82 -> 86%) và shirt (66 -> 66 -> 71).
- đối với các cặp dự đoán sai:
  - Đúng là coat (dự đoán thành pullover), thì patch7 tệ nhất, đoán nhầm lên đến 10%, trong khi row và cnn là 5% và 3%.
  - shirt -> coat đây là điểm đáng chú ý, vì CNN có tỉ lệ đoán nhầm cao hơn cả patch7 và row, lần lượt là 6% -> 7% -> 10%.
  - trong đó, cập shirt <-> t-shirt vẫn là cặp có tỉ lệ đoán nhầm cao nhất, tức vẫn là ranh giới khó nhất, giống với kết qua trước đây.
"""

COMMENTARY_ERRORS = """\
Lỗi tự tin nhất, có nhiều ảnh mà chính mắt người cũng thấy nhãn đáng ngờ:
- “T: Sneaker” nhưng ảnh trông giống ankle boot cổ cao.
- “T: Shirt” nhưng hình chữ nhật trông giống túi.
- “T: T-shirt” nhưng là áo hai dây, bị đoán thành Dress.

=> Vì các lớp áo có cùng hình dạng tổng thể, muốn phân biệt chúng cần chi tiết cục bộ hơn như cổ áo, khuy, độ dài tay áo... Đây là thứ mà cnn-stem làm tốt, trích xuất tốt hơn một phép chiếu tuyến tính (patch, row). Điều này giải thích vì sao lợi thế cnn tập trung ở nhóm áo.
"""

COMMENTARY_ATTENTION = """\
Mỗi cột là một ảnh (một lớp ảnh), mỗi hàng là một tokenzier. Màu nóng cho viết token đó đóng góp nhiều vào biểu diễn của cls.

- Với row:

  - T-Shirt: dãy sáng tập trung vào hàng tay áo kết thúc, cho thấy độ dài tay áo là điểm phân biệt của T-shirt với Pullover.
  - Coat: dãy sáng tập trung ở phần cổ áo.
  - Trouser: dãy sáng tập trung ở phần thân quần, nửa dưới, thể hiện độ dài của quần.
  - Sandal, Sneaker, Boot: dãy sáng tập trung vào hàng đế và cổ giày.

- Với patch:

  - Attention khá phân tán, thường rơi vào các pattern nằm ở viền của vật thể.
  - Điều này cũng phản ánh độ thô và thưa của lưới 4x4 với mỗi ô patch là 7x7, không đủ chi tiết để khoanh vùng cho chi tiết.

- Với cnn:

  - Chia thành nhiều ô lưới hơn patch, nhưng attention cũng khá thưa, nhưng có những điểm tập trung nhất định vào chi tiết của các vật thể.
"""


# ---------------------------------------------------------------- tables (cũng được main in ra console)

def mcnemar_table(outputs: Dict[str, tuple], tokenizers: Sequence[str]) -> str:
    lines = [f"{'tokenizer':<10}{'agree':>8}{'b':>6}{'c':>6}{'p (McNemar)':>13}"]
    for tok in tokenizers:
        y, p_s, _ = outputs[f"{tok}_scratch"]
        _, p_t, _ = outputs[f"{tok}_torch"]
        b = int(np.sum((p_s == y) & (p_t != y)))   # scratch đúng, torch sai
        c = int(np.sum((p_s != y) & (p_t == y)))   # scratch sai, torch đúng
        pval = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        lines.append(f"{tok:<10}{(p_s == p_t).mean():>8.2%}{b:>6}{c:>6}{pval:>13.3f}")
    return "\n".join(lines)


def summary_table(results: List[dict]) -> str:
    lines = [f"{'model':<16}{'N':>4}{'params':>9}{'ep':>4}{'s/ep':>7}{'lat(ms)':>9}"
             f"{'train':>8}{'val':>8}{'test':>8}{'gap':>6}"]
    for r in results:
        lines.append(f"{r['model']:<16}{r['num_tokens']:>4}{r['params']:>9,}{r['best_epoch']:>4}{r['sec_per_epoch']:>7.1f}"
                     f"{r['latency_ms']:>9.2f}{r['train_acc']:>8.2%}{r['val_acc']:>8.2%}{r['test_acc']:>8.2%}"
                     f"{(r['train_acc'] - r['val_acc']) * 100:>6.1f}")
    return "\n".join(lines)


def recall_table(cms: Dict[str, np.ndarray]) -> str:
    lines = [f"{'Recall (%)':<26}" + "".join(f"{t:>8}" for t in cms)]
    for k, cls_name in enumerate(CLASS_NAMES):
        lines.append(f"{cls_name:<26}" + "".join(f"{cms[t][k, k]:>8.1%}" for t in cms))
    for true_name, pred_name in KEY_PAIRS:
        i, j = CLASS_NAMES.index(true_name), CLASS_NAMES.index(pred_name)
        lines.append(f"{true_name + ' -> ' + pred_name:<26}" + "".join(f"{cms[t][i, j]:>8.0%}" for t in cms))
    return "\n".join(lines)


def _learning_curve_stats(histories: Dict[str, dict], results: List[dict]) -> str:
    lines = []
    for r in results:
        h = histories[r["model"]]
        best_val_loss_ep = int(np.argmin(h["val_loss"]))
        lines.append(
            f"- {r['model']}: {len(h['val_loss'])} epochs run, best val acc {max(h['val_acc']):.2%} "
            f"at epoch {r['best_epoch']}; min val loss {h['val_loss'][best_val_loss_ep]:.3f} "
            f"at epoch {best_val_loss_ep + 1}; train loss {h['train_loss'][0]:.3f} -> {h['train_loss'][-1]:.3f}")
    return "\n".join(lines)


def _confident_errors(outputs: Dict[str, tuple], n: int = 16) -> str:
    lines = []
    for name, (y_true, y_pred, probs) in outputs.items():
        wrong = np.flatnonzero(y_pred != y_true)
        conf = np.sort(probs[wrong, y_pred[wrong]])[::-1][:n]
        lines.append(f"- {name}: {len(wrong):,} lỗi / {len(y_true):,}; top-{len(conf)} confident errors, "
                     f"{int(np.sum(conf >= 0.995))} at ~100% confidence, min {conf.min():.1%}")
    return "\n".join(lines)


def _figures(names: Sequence[str]) -> str:
    return "\n".join(f"- figures/{n}.png" for n in names)


def export_report(path: Path, results: List[dict], histories: Dict[str, dict], outputs: Dict[str, tuple],
                  cms: Dict[str, np.ndarray], tokenizers: Sequence[str]) -> Path:
    scratch_outs = {f"{t}_scratch": outputs[f"{t}_scratch"] for t in tokenizers}
    sections = [
        "## Evaluation",
        f"(generated {datetime.now():%Y-%m-%d %H:%M}, seed={CONFIG['seed']}, "
        f"epochs={CONFIG['epochs']}, patience={CONFIG['patience']})\n"
        "Số liệu trong các bảng được tính từ run hiện tại; phần nhận xét là phân tích cho run seed 42 trong notebook.",

        "### 1. Bảng kết quả training",
        format_results_table(results),
        COMMENTARY_TABLE,

        "### 2. Learning curves",
        _figures([f"E2_curves_{t}_scratch_vs_torch" for t in tokenizers] + ["E2_curves_tokenizers"]),
        "Số liệu learning curves (run hiện tại):\n" + _learning_curve_stats(histories, results),
        COMMENTARY_CURVES,

        "### 3. Kiểm định McNemar (scratch vs. torch)",
        mcnemar_table(outputs, tokenizers),
        COMMENTARY_MCNEMAR,

        "### 4. Accuracy vs. chi phí",
        summary_table(results),
        _figures(["E2_acc_vs_cost"]),
        COMMENTARY_COST,

        "### 5. Confusion matrix và recall theo lớp (MSA tự viết, % theo hàng)",
        recall_table(cms),
        _figures(["E2_confusion_matrices"]),
        COMMENTARY_CONFUSION,

        "### 6. Các lỗi tự tin nhất",
        _confident_errors(scratch_outs),
        _figures([f"E2_misclassified_{t}" for t in tokenizers]),
        COMMENTARY_ERRORS,

        "### 7. Attention maps (attention rollout từ CLS)",
        _figures(["E2_attention_rollout"]),
        COMMENTARY_ATTENTION,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections), encoding="utf-8")
    return path
