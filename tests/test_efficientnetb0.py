"""Offline implementation checks; synthetic inputs are not accuracy results.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_efficientnetb0.py' -v
"""

import unittest

import torch
from src.models.efficientnetb0 import create_efficientnetb0


class EfficientNetB0Tests(unittest.TestCase):
    def test_classifier_updates_without_changing_frozen_backbone(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = create_efficientnetb0(pretrained=False, freeze_backbone=True)
        model.train()
        self.assertFalse(model.network.features.training)
        self.assertTrue(model.network.classifier.training)
        before = {k: v.clone() for k, v in model.network.features.state_dict().items()}
        classifier_before = model.network.classifier[1].weight.detach().clone()
        images = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([0, 9])
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=0.001
        )
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 10))
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.network.classifier[1].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.network.features.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(classifier_before, model.network.classifier[1].weight))
        for key, value in model.network.features.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

        restored = create_efficientnetb0(pretrained=False)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(images), restored(images))

        model.set_backbone_trainable(True)
        model.train()
        self.assertTrue(model.network.features.training)
        self.assertTrue(all(p.requires_grad for p in model.network.features.parameters()))


if __name__ == "__main__":
    unittest.main()
