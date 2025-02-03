import unittest
import torch
from easyroutine.interpretability.activation_cache import ActivationCache

class TestActivationCache(unittest.TestCase):

    def setUp(self):
        """
        Setup the test case by initializing necessary ActivationCache instances.
        """
        self.cache1 = ActivationCache()
        self.cache2 = ActivationCache()

        # Populate cache1
        self.cache1["values_0"] = torch.tensor([1, 2])
        self.cache1["mapping_index"] = [0, 1]

        # Populate cache2
        self.cache2["values_0"] = torch.tensor([3, 4])
        self.cache2["mapping_index"] = [2, 3]

    def test_cat(self):
        """
        Test the cat method to merge two ActivationCache objects.
        """
        self.cache1.cat(self.cache2)

        self.assertTrue(torch.equal(self.cache1["values_0"], torch.tensor([1, 2, 3, 4])))
        self.assertEqual(self.cache1["mapping_index"], [0, 1])

    def test_register_aggregation(self):
        """
        Test custom aggregation strategies.
        """
        self.cache1.register_aggregation("values_", lambda values: torch.stack(values, dim=0))
        self.cache1.cat(self.cache2)

        expected = torch.stack([torch.tensor([1, 2]), torch.tensor([3, 4])], dim=0)
        self.assertTrue(torch.equal(self.cache1["values_0"], expected))

    def test_deferred_mode(self):
        """
        Test the deferred_mode context manager.
        """
        with self.cache1.deferred_mode():
            self.cache1.cat(self.cache2)

        self.assertTrue(torch.equal(self.cache1["values_0"], torch.tensor([1, 2, 3, 4])))
        self.assertEqual(self.cache1["mapping_index"], [0, 1])

    def test_key_mismatch(self):
        """
        Test that a ValueError is raised for mismatched keys.
        """
        self.cache2["extra_key"] = torch.tensor([5])

        with self.assertRaises(ValueError):
            self.cache1.cat(self.cache2)

    def test_empty_initialization(self):
        """
        Test initializing an empty ActivationCache with another cache.
        """
        empty_cache = ActivationCache()
        empty_cache.cat(self.cache1)

        self.assertTrue(torch.equal(empty_cache["values_0"], torch.tensor([1, 2])))
        self.assertEqual(empty_cache["mapping_index"], [0, 1])
        
    def test_add_with_info(self):
        self.cache1.add_with_info("values_0", torch.tensor([5, 6]), "informative_string")
        self.assertTrue(torch.equal(self.cache1["values_0"].value(), torch.tensor([5, 6])))
        self.assertEqual(self.cache1["values_0"].info(), "informative_string")
        

if __name__ == "__main__":
    unittest.main()
