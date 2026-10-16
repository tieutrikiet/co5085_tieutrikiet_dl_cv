"""Các kiểm tra trước khi train: MSA scratch == torch, shape/tham số, overfit một batch, benchmark MSA."""
import torch
import torch.nn.functional as F

from common.utils import count_parameters, set_seed

from .attention import ATTENTIONS, MSAPyTorch, MultiHeadSelfAttention, scratch_to_torch_state_dict
from .config import CONFIG
from .timing import bench_attention
from .tokenizers import TOKENIZERS
from .transformer import TransformerClassifier


@torch.no_grad()
def check_scratch_and_torch(B=4, N=20, D=64, h=4, atol=1e-5):
    """Kiểm tra hai mô hình Attention cho cùng output khi dùng chung trọng số."""
    set_seed(0)
    scratch = MultiHeadSelfAttention(embed_dim=D, num_heads=h).eval()
    torch_msa = MSAPyTorch(embed_dim=D, num_heads=h).eval()
    torch_msa.load_state_dict(scratch_to_torch_state_dict(scratch.state_dict()))

    # Tạo đầu vào ngẫu nhiên
    x = torch.randn(B, N, D)

    # Kiểm tra đầu ra của scratch và torch có gần nhau không
    out_scratch, attn_scratch = scratch(x, return_attn=True)
    out_torch, attn_torch = torch_msa(x, return_attn=True)
    assert torch.allclose(out_scratch, out_torch, atol=atol) and torch.allclose(attn_scratch, attn_torch, atol=atol), "Outputs are not close!"

    # Tính số tham số
    n_s, n_t = count_parameters(scratch), count_parameters(torch_msa)
    print(f"Number of parameters - Scratch: {n_s}, Torch: {n_t}")

    # Kiểm tra số tham số phải bằng nhau và bằng 4D^2 + 4D
    assert n_s == n_t == 4 * D**2 + 4 * D, "Number of parameters is not correct!"


def benchmark_attentions(device):
    # N = số token với ảnh 28×28:
    #   16  = patch size 7 → lưới 4×4
    #   28  = mỗi hàng là 1 token
    #   49  = patch size 4 → lưới 7×7
    #   196 = patch size 2 → lưới 14×14
    for N in (16, 28, 49, 196):
        times = {name: bench_attention(cls, N, device=device) for name, cls in ATTENTIONS.items()}
        print(f"N={N:>4}: " + "  ".join(f"{k}={v:6.2f} ms" for k, v in times.items()))


def check_shapes_and_params():
    """Test 1 — shape và bảng tham số"""
    for tok_name, make in TOKENIZERS.items():
        set_seed(0)
        model = TransformerClassifier(tokenizer=make(64))
        x = torch.randn(8, 1, 28, 28)

        print(f"Testing tokenizer: {tok_name}")
        assert model(x).shape == (8, 10), "Output shape is incorrect"

        print(f"{'name':<10} {'parameters':>8}")
        for name, part in [("tokenizer", model.tokenizer), ("1 block", model.blocks[0]),
                           ("  ├ attn", model.blocks[0].attention), ("  └ ffn", model.blocks[0].ffn),
                           ("head", model.head), ("TOTAL", model)]:
            print(f"{name:<10} {count_parameters(part):>8,}")
        print()


@torch.no_grad()
def check_classifier_equivalence(atol=1e-5, tokenizer_name="patch7"):
    """Test 2 — scratch và torch cho cùng logits khi dùng chung trọng số"""
    set_seed(0)
    tok = TOKENIZERS[tokenizer_name](64)
    m_scratch = TransformerClassifier(tok, attention="scratch").eval()
    m_torch = TransformerClassifier(tok, attention="torch").eval()
    m_torch.load_state_dict(scratch_to_torch_state_dict(m_scratch.state_dict()))   # strict=True
    x = torch.randn(8, 1, 28, 28)
    diff = (m_scratch(x) - m_torch(x)).abs().max().item()
    print(f"{tokenizer_name:<8} max |Δlogits| = {diff:.2e}")
    assert diff < atol, f"Difference {diff:.2e} exceeds tolerance {atol}"


def check_overfit_one_batch(train_loader, steps=200):
    """Test 3 — overfit một batch: loop train đúng thì loss phải về gần 0"""
    for name, make in TOKENIZERS.items():
        print(f"Testing tokenizer: {name}")
        xb, yb = next(iter(train_loader))
        xb, yb = xb[:64], yb[:64]
        tok = make(64)
        set_seed(0)
        model = TransformerClassifier(tok)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

        for step in range(steps):
            model.train()
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            if step % 50 == 0 or step == steps - 1:
                print(f"Step {step:>3}: loss = {loss.item():.4f}")
        assert loss.item() < 0.05, f"Không overfit được 1 batch: loss = {loss.item():.4f}"
        print()


def run_all_checks(train_loader, device):
    print("\n[Check] MSA scratch vs. torch")
    check_scratch_and_torch()
    print("\n[Check] Benchmark MSA (forward + backward, B=128)")
    benchmark_attentions(device)
    print("\n[Check] Shape và bảng tham số")
    check_shapes_and_params()
    print("[Check] TransformerClassifier scratch vs. torch")
    for tokenizer_name in TOKENIZERS:
        check_classifier_equivalence(tokenizer_name=tokenizer_name)
    print("\n[Check] Overfit một batch")
    check_overfit_one_batch(train_loader)
