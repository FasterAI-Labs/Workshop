import time
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from fastai.vision.all import *
from fasterai.sparse.all import *
from fasterai.prune.all import *
from fasterai.quantize.all import *
from fasterai.analyze.sensitivity import analyze_sensitivity
from fasterbench.benchmark import benchmark
from fasterbench.speed import compute_speed
from fasterbench.profiling import LayerProfiler
import copy


# ── Hardware targets (used by plot_ai's roofline ridge) ──────────────────────
# plot_ai falls back to the module-level TARGET when no target is passed.
# Because plot_ai is defined here, it resolves TARGET in *this* module's
# namespace — so the default must live here. Override via plot_ai(..., target=).
TARGETS = {
    "t4-colab":    dict(label="NVIDIA T4 (this Colab)", peak_TFLOPs=8.1,  bw_GBs=320),
    "a100":        dict(label="NVIDIA A100",            peak_TFLOPs=312,  bw_GBs=2039),
    "jetson-orin": dict(label="Jetson Orin",            peak_TFLOPs=275,  bw_GBs=204),
    "x86-cpu":     dict(label="x86 CPU (AVX-512/VNNI)", peak_TFLOPs=1.5,  bw_GBs=80),
    "arm-mobile":  dict(label="ARM mobile (NEON)",      peak_TFLOPs=0.5,  bw_GBs=25),
}
TARGET = TARGETS["x86-cpu"]


def benchm(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def compute_per_layer_ai(model, input_shape=(1,3,32,32), weight_bits=None, act_bits=None,
                         print_table=True, n=10, name_w=40):

    results, hooks = [], []
    def wbytes(count, wtensor):
        if weight_bits is not None: return count * weight_bits / 8
        return count * (wtensor.element_size() if wtensor is not None else 4)
    def abytes(t):
        return t.numel() * (act_bits / 8 if act_bits is not None else t.element_size())
    def get_w(m):
        w = getattr(m, 'weight', None)
        return w() if callable(w) else w
    def make_hook(name):
        def hook(module, inp, out):
            conv = hasattr(module, 'out_channels') and hasattr(module, 'kernel_size')
            lin  = hasattr(module, 'out_features')
            if conv:
                oh, ow = out.shape[2], out.shape[3]
                kops = module.in_channels * module.kernel_size[0] * module.kernel_size[1] / module.groups
                ops = 2 * out.shape[0] * module.out_channels * oh * ow * kops
                w_count = module.out_channels * kops
            elif lin:
                ops = 2 * module.in_features * module.out_features * out.shape[0]
                w_count = module.in_features * module.out_features
            else:
                return
            total = abytes(inp[0]) + abytes(out) + wbytes(w_count, get_w(module))
            results.append((name, ops, total, ops / total if total > 0 else 0))
        return hook
    for name, module in model.named_modules():
        if (hasattr(module, 'out_channels') and hasattr(module, 'kernel_size')) or hasattr(module, 'out_features'):
            if get_w(module) is not None:
                hooks.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    x = torch.randn(*input_shape).to('cpu')
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    if print_table:
        print(f"{'#':>4} {'Layer':<{name_w}} {'OPs':>15} {'Bytes':>15} {'AI (OPs/byte)':>18}")
        print("-" * (name_w + 55))
        for i, (nm, ops, by, ai) in enumerate(results[:n]):
            print(f"{i:>4} {nm[:name_w]:<{name_w}} {ops:>15,.0f} {by:>15,.0f} {ai:>18.2f}")
        print("...")
    return results


def plot_ai(layer_ai, target=None, ax=None, figsize=(14, 6), title=None, show=None):
    target = target or TARGET
    ridge = target["peak_TFLOPs"] * 1000 / target["bw_GBs"]
    names = [n for n, *_ in layer_ai]; ais = [t[-1] for t in layer_ai]
    colors = ['#e74c3c' if a < ridge else '#2ecc71' for a in ais]   # rouge=memory-bound
    created = ax is None
    fig, ax = (plt.subplots(figsize=figsize) if created else (ax.figure, ax))
    ax.bar(range(len(names)), ais, color=colors)
    ax.axhline(ridge, color='black', ls='--', lw=2,
               label=f'{target["label"]} ridge (~{ridge:.0f} FLOP/byte)')
    ax.set_yscale('log'); ax.set_ylabel('Arithmetic Intensity (FLOP/byte, log)')
    ax.set_xlabel('Layer index'); ax.set_title(title or f'Per-layer AI vs {target["label"]}'); ax.legend()
    n_c = sum(a >= ridge for a in ais)
    if created: fig.tight_layout()
    if show or (show is None and created): plt.show()
    return fig, ax, {'compute_bound': n_c, 'memory_bound': len(ais) - n_c, 'total': len(ais)}


@torch.no_grad()
def quick_acc(m, n_batches=10):
    m.eval()
    dev = next(m.parameters()).device
    correct = total = 0
    for i, (xb, yb) in enumerate(dls.valid):
        if i >= n_batches: break
        xb, yb = xb.to(dev), yb.to(dev)
        correct += (m(xb).argmax(dim=1) == yb).sum().item(); total += yb.numel()
    return correct / total

PruneCB = partial(PruneCallback, context='local', criteria=large_final, schedule=agp)
demo_pruner = partial(Pruner, context='local', criteria=large_final)

def prune(model, pruning_ratio, image_size=128):
    return demo_pruner(model, pruning_ratio, example_inputs=torch.randn(1, 3, image_size, image_size)).prune_model()

_quantizer = partial(Quantizer, backend='x86', method='static')
def quantize(model):
    return _quantizer().quantize(model, dls.valid)


_sens = partial(analyze_sensitivity, compression='pruning', level=50,
                criteria=large_final, metric_name='accuracy', verbose=False, layer_types=nn.Conv2d)

def sensitivity(model, image_size):
    return _sens(model, torch.randn(1, 3, image_size, image_size), quick_acc)

def score_model(model):
    inp = dls.one_batch()[0][0][None]
    return compute_speed(model, inp).p99_ms, quick_acc(model)

path = untar_data(URLs.PETS)
files = get_image_files(path/"images")

def get_model(model):
    learn = vision_learner(dls, model, metrics=accuracy)
    learn.unfreeze();
    learn.model.cpu();
    learn.model.eval();
    return learn.model

def label_func(f): return f[0].isupper()

dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(128))

_bench = partial(benchmark, metrics=["size", "speed", "compute"], speed_devices=["cpu"])
def demo_benchmark(model, image_size):
    return _bench(model, torch.randn(1, 3, image_size, image_size)).summary()
