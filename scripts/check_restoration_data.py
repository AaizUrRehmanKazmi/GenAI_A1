"""Check real training batches and fixed validation replay; never load test images."""
import torch
from torch.utils.data import DataLoader
from src.data.pets_dataset import TrainingPetsDataset, EvaluationPetsDataset, REPO_ROOT


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = TrainingPetsDataset()
    first, second = training[0], training[0]
    assert first['spec_json'] != second['spec_json']
    assert torch.equal(first['target'], second['target'])
    loader = DataLoader(training, batch_size=16, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    batch = next(iter(loader))
    assert batch['input'].shape == batch['target'].shape == (16, 3, 128, 128)
    validation = EvaluationPetsDataset(REPO_ROOT / 'data/splits/pets_val.json',
                                      REPO_ROOT / 'data/manifests/pets_val_corruptions.json')
    for index in range(10):
        a, b = validation[index], validation[index]
        assert torch.equal(a['input'], b['input'])
        assert torch.equal(a['target'], b['target'])
    print(f'Training images: {len(training)}; input/target batch shape: {tuple(batch["input"].shape)}')
    print(f'Validation cases: {len(validation)}; first ten cases replay identically.')
    print('Fresh training settings and stable targets verified. Test images were not loaded.')


if __name__ == '__main__':
    main()
