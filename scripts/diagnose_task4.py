"""Read-only best-checkpoint diagnostics; no official test images are read."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from src.data.fs2k_dataset import FS2KDataset
from src.models.generator_unet import StyleUNet
from training.train_task4_gan import validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw/fs2k/FS2K'))
    p.add_argument('--splits-dir', type=Path, default=Path('data/splits/fs2k'))
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    architecture = saved['config'].get('architecture')
    if architecture == 'deep_unet_v1':
        from src.models.task4_candidate import CandidateGenerator
        model = CandidateGenerator(**saved['config']['generator'])
    elif architecture is None:
        model = StyleUNet(**saved['config']['generator'])
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    model.load_state_dict(saved['generator']); model.eval()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'epoch': saved['progress']['epoch'], 'official_test_evaluated': False,
              'note': 'Full unaugmented train/validation metrics. Cross-style outputs have no paired target except the annotated style.',
              'splits': {}}
    for split in ('train', 'val'):
        data = FS2KDataset(a.splits_dir / f'{split}.json', a.raw_dir)
        metrics = validation(model, data, torch.device('cpu'), 8, lambda: False)
        print(split, metrics, flush=True)
        canvas = Image.new('RGB', (660, 450), 'white'); draw = ImageDraw.Draw(canvas)
        differences = []
        for style in range(3):
            sample = data[next(i for i, r in enumerate(data.rows) if r['style'] == style)]
            with torch.inference_mode():
                outputs = model(sample['photo'][None].repeat(3, 1, 1, 1), torch.arange(3))
            differences.append({'image_id': sample['image_id'], 'annotated_style': style,
                'pairwise_output_l1': {f'{i}-{j}': (outputs[i]-outputs[j]).abs().mean().item()
                                       for i, j in ((0,1),(0,2),(1,2))}})
            draw.text((0, style*150), f'{sample["image_id"]} target style {style+1}', fill='black')
            for j, tensor in enumerate([sample['photo'], sample['sketch'], *outputs]):
                tile = Image.fromarray((tensor.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8'))
                canvas.paste(tile, (132*j, style*150+22))
                draw.text((132*j, style*150+11), ['Photo','Target','Style 1','Style 2','Style 3'][j], fill='black')
        canvas.save(a.output_dir / f'{split}_styles.png')
        report['splits'][split] = {'metrics': metrics, 'style_examples': differences}
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
