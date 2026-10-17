"""Các kiểm tra trước khi train: shape của chuỗi, shape và số tham số của mô hình."""
import numpy as np
import torch

from common.datasets import FASHION_MNIST_SHAPE
from common.utils import count_parameters, set_seed

from .classifiers import RNNS, build_model
from .config import E3_CONFIG, NUM_CLASSES
from .sequences import SEQUENCES, create_sequence, sequence_shape


def check_sequence_shapes(train_loader):
    """Test — shape và số pixel"""
    xb, yb = next(iter(train_loader))
    print(f"{'sequence':<10}{'input':<20}{'output':<20}{'T':>5}{'D':>5}")
    for name, spec in SEQUENCES.items():
        seq = create_sequence(xb, **spec)
        T, D = sequence_shape(spec)
        assert seq.shape == (xb.size(0), T, D)
        assert T * D == np.prod(FASHION_MNIST_SHAPE), "Số pixel phải được giữ nguyên"
        print(f"{name:<10}{str(tuple(xb.shape)):<20}{str(tuple(seq.shape)):<20}{T:>5}{D:>5}")


def rnn_param_formula(rnn_type, D, H):
    gates = {"lstm": 4, "gru": 3}[rnn_type]
    return gates * (D * H + H * H + 2 * H)  # công thức tính số tham số RNN theo lý thuyết


def check_shapes_and_params():
    """Test 1 — shape và số tham số"""
    H = E3_CONFIG["hidden_size"]
    x = torch.randn(8, 1, 28, 28)
    print(f"{'model':<12}{'T':>4}{'D':>4}{'rnn':>9}{'head':>7}{'total':>9}")
    for rnn_type in RNNS:
        for seq_name in SEQUENCES:
            set_seed(0)
            model = build_model(rnn_type, seq_name)
            assert model(x).shape == (8, NUM_CLASSES), "Output shape is incorrect"
            T, D = sequence_shape(SEQUENCES[seq_name])
            n_rnn = count_parameters(model.rnn)
            assert n_rnn == rnn_param_formula(rnn_type, D, H), "Số tham số RNN không khớp công thức"
            print(f"{rnn_type + '_' + seq_name:<12}{T:>4}{D:>4}{n_rnn:>9,}{count_parameters(model.head):>7,}"
                  f"{count_parameters(model):>9,}")


def run_all_checks(train_loader):
    print("\n[Check] Shape của chuỗi và số pixel")
    check_sequence_shapes(train_loader)
    print("\n[Check] Shape và số tham số của RNNClassifier")
    check_shapes_and_params()
