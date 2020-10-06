import unittest
from tqdm import tqdm
from datasets import GloveDataset

class TestGloveDataset(unittest.TestCase):
    def test_dataset(self):
        with open('tests/test_dataset.txt') as f:
            dataset = GloveDataset(f.read(), 10000000)
            count = 0
            for xij, i_idx, j_idx in dataset.get_batches(1024):
                count += 1
        
        self.assertTrue(count > 0)
        
if __name__ == '__main__':
    unittest.main()
    