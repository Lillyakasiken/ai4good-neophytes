"""Estimate the GPU memory of a 512 development run for another encoder.

Builds the U-Net of ``models/semseg_plm.py`` for each encoder, runs a few AdamW
training steps on random 512 px crops (batch 8, bf16) and one validation forward
pass on a random 2048 px tile (batch 2), and records the peak reserved memory.
The difference to mit_b2 is added to the measured peak of the real mit_b2 512 run,
because the real run also holds the CUDA context, metrics and data on the GPU.

    python scripts/queues/probe_gpu_mem.py --encoders mit_b2 mit_b4 --b2-peak-gib 34

The last line of the output is the estimated peak in GiB of the last encoder.
Random inputs only; no project data is read. Downloads the encoder weights into the
Hugging Face cache, so train.py does not have to.
"""
import argparse
import gc

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def probe(encoder_name, steps=3, crop=512, batch_train=8, tile=2048, batch_val=2,
          num_classes=7):
    model = smp.Unet(encoder_name=encoder_name, in_channels=3, classes=num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()

    model.train()
    for _ in range(steps):
        x = torch.randn(batch_train, 3, crop, crop, device='cuda')
        y = torch.randint(0, num_classes, (batch_train, crop, crop), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_gib = torch.cuda.max_memory_reserved() / 2**30

    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        model(torch.randn(batch_val, 3, tile, tile, device='cuda'))
    peak_gib = torch.cuda.max_memory_reserved() / 2**30

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return train_gib, peak_gib


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--encoders', nargs='+', default=['mit_b2', 'mit_b4'],
                    help='the first one is the reference with the known real peak')
    ap.add_argument('--b2-peak-gib', type=float, default=34.0,
                    help='measured peak of the real run of the first encoder')
    args = ap.parse_args()

    peaks = {}
    for enc in args.encoders:
        train_gib, peak_gib = probe(enc)
        peaks[enc] = peak_gib
        print(f'{enc}: training {train_gib:.1f} GiB, with validation {peak_gib:.1f} GiB')

    ref = args.encoders[0]
    for enc in args.encoders[1:]:
        est = args.b2_peak_gib + peaks[enc] - peaks[ref]
        print(f'{enc}: estimated peak of the real run {est:.1f} GiB '
              f'({ref} {args.b2_peak_gib:.0f} GiB + {peaks[enc] - peaks[ref]:+.1f} GiB)')
    print(f'{est:.1f}' if len(args.encoders) > 1 else f'{peaks[ref]:.1f}')


if __name__ == '__main__':
    main()
