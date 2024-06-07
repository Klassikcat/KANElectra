import random
import unittest
import torch
from transformers import AutoTokenizer
import kipsum
import sys
sys.path.append('../')
from src.ElectraKAN.datamodule import (
    ElectraClassificationDataset,
    ElectraKANDataModule,
    ElectraPretrainingDataset
)


TOKENIZER = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
KIPSUM_GENERATOR = kipsum.Kipsum()


class TestModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sentenceGeneratedNum = random.randint(1000, 10000)
        cls.maxLength = 512
        cls.generatedSentences = KIPSUM_GENERATOR.paragraphs(
            cls.sentenceGeneratedNum
        )
        cls.pretrainingDataset = ElectraPretrainingDataset(
            texts=cls.generatedSentences,
            tokenizer=TOKENIZER,
            max_length=cls.maxLength
        )
        cls.classificationDataset = ElectraClassificationDataset(
            texts_and_labels=[(i, random.choice(["0", "1", "2", "3"])) for i in cls.generatedSentences],
            tokenizer=TOKENIZER,
            labels=["0", "1", "2", "3"]
        )
        cls.dataModule = ElectraKANDataModule(
            train_dataset=cls.pretrainingDataset,
            val_dataset=cls.classificationDataset,
            test_dataset=cls.classificationDataset,
            batch_size=8,
            num_workers=2,
            pin_memory=True
        )
        cls.tokenizerArgs = {
            "return_attention_mask": True,
            "return_token_type_ids": True,
            "return_tensors": 'pt',
            "max_length": cls.maxLength,
            "padding": "max_length",
            "truncation": True
        }

    def test_ElectraPretrainingDataset(self):
        self.assertEqual(len(self.pretrainingDataset), self.sentenceGeneratedNum)
        self.assertEqual(len(self.pretrainingDataset[0]), 4)
        tokenizer_outputs = TOKENIZER(self.generatedSentences[0], **self.tokenizerArgs)
        self.assertNotEqual(self.pretrainingDataset[0][0].tolist(), tokenizer_outputs["input_ids"].squeeze(0).tolist())
        self.assertEqual(self.pretrainingDataset[0][1].tolist(), tokenizer_outputs["attention_mask"].squeeze(0).tolist())
        self.assertEqual(self.pretrainingDataset[0][2].tolist(), tokenizer_outputs["token_type_ids"].squeeze(0).tolist())
        self.assertEqual(self.pretrainingDataset[0][3].tolist(), tokenizer_outputs["input_ids"].squeeze(0).tolist())
        
    def test_ElectraClassificationDataset(self):
        tokenizer_outputs = TOKENIZER(self.generatedSentences[0], **self.tokenizerArgs)
        self.assertEqual(len(self.classificationDataset), self.sentenceGeneratedNum)
        self.assertEqual(len(self.classificationDataset[0]), 4)
        self.assertEqual(self.classificationDataset[0][0].tolist(), tokenizer_outputs["input_ids"].squeeze(0).tolist())
        self.assertEqual(self.classificationDataset[0][1].tolist(), tokenizer_outputs["attention_mask"].squeeze(0).tolist())
        self.assertEqual(self.classificationDataset[0][2].tolist(), tokenizer_outputs["token_type_ids"].squeeze(0).tolist())
        self.assertIsInstance(self.classificationDataset[0][3], torch.Tensor)
        

if __name__ == '__main__':
    unittest.main()