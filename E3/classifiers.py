"""Bộ phân loại LSTM / GRU: ảnh -> chuỗi -> RNN -> h_T -> Dropout -> Linear -> logits."""
import torch
import torch.nn as nn

from E1.classifiers import MLPClassifier, SoftmaxClassifier

from .config import E3_CONFIG, NUM_CLASSES
from .sequences import SEQUENCES, create_sequence, sequence_shape

RNNS = {
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}


class RNNClassifier(nn.Module):
    """Ảnh [B, C, H, W] -> chuỗi [B, T, D] -> LSTM/GRU -> vector đặc trưng -> logits [B, 10]"""
    def __init__(self, rnn_type="lstm", seq_spec=SEQUENCES["row"],
                 hidden_size=128, num_layers=1, bidirectional=False, dropout=0.1,
                 pool="last", head="softmax", batch_first=True):
        super().__init__()
        assert rnn_type.lower() in ["lstm", "gru"], "rnn_type must be either 'lstm' or 'gru'"
        self.seq_spec = dict(seq_spec)
        self.num_steps, input_size = sequence_shape(seq_spec)  # T, D
        self.pool = pool
        self.num_directions = 2 if bidirectional else 1
        self.rnn = RNNS[rnn_type.lower()](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first,
        )
        feat_dim = hidden_size * self.num_directions
        self.drop = nn.Dropout(dropout)
        if head == "softmax":
            self.head = SoftmaxClassifier(input_dim=feat_dim, num_classes=NUM_CLASSES)
        else:
            self.head = MLPClassifier(input_dim=feat_dim, hidden_dims=(feat_dim,), dropout=dropout, num_classes=NUM_CLASSES)

    def forward(self, x):
        # Tạo chuỗi từ ảnh với "spec" được cung cấp ở trên
        seq = create_sequence(x, **self.seq_spec)
        out, h_n = self.rnn(seq)  # out: [B, T, H*num_directions]
                                  # h_n: [num_layers*num_directions, B, H]
        if isinstance(h_n, tuple):  # LSTM trả về (h_n, c_n)
            h_n = h_n[0]

        if self.pool == "last":
            # hidden state cuối của tầng trên cùng, ghép 2 chiều nếu bidirectional
            feat = torch.cat(list(h_n[-self.num_directions:]), dim=-1)  # [B, dirs * H]
        else:
            # pooling theo mean
            feat = out.mean(dim=1)  # [B, dirs * H]

        feat = self.drop(feat)
        logits = self.head(feat)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.softmax(self(x), dim=1)


def build_model(rnn_type, seq_name, **overrides):
    cfg = {**E3_CONFIG, **overrides}
    return RNNClassifier(
        rnn_type=rnn_type,
        seq_spec=SEQUENCES[seq_name],
        **cfg,
    )
