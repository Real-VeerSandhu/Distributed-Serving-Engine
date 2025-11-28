import os
import time
import unittest
from unittest.mock import patch

from src.kvstore import KVStore, create_kv_store, CacheEntry


class TestKVStore(unittest.TestCase):
    def setUp(self):
        self.kv = create_kv_store(max_size=3)
    
    def tearDown(self):
        self.kv.close()
        if hasattr(self, 'temp_db'):
            try:
                os.unlink(self.temp_db)
            except:
                pass
    
    def test_basic_operations(self):
        """Test basic set, get, and delete operations."""
        # Test set and get
        self.kv.set('key1', 'value1')
        self.assertEqual(self.kv.get('key1'), 'value1')
        
        # Test update
        self.kv.set('key1', 'new_value')
        self.assertEqual(self.kv.get('key1'), 'new_value')
        
        # Test non-existent key
        self.assertIsNone(self.kv.get('nonexistent'))
        self.assertEqual(self.kv.get('nonexistent', 'default'), 'default')
        
        # Test delete
        self.assertTrue(self.kv.delete('key1'))
        self.assertFalse(self.kv.delete('nonexistent'))
        self.assertIsNone(self.kv.get('key1'))
        
        # Test __contains__
        self.kv['key2'] = 'value2'
        self.assertIn('key2', self.kv)
        self.assertNotIn('nonexistent', self.kv)
    
    def test_ttl(self):
        """Test time-to-live functionality."""
        # Test with TTL
        self.kv.set('key1', 'value1', ttl=0.1)  # 100ms TTL
        self.assertEqual(self.kv.get('key1'), 'value1')
        
        # Test TTL expiration
        time.sleep(0.2)
        self.assertIsNone(self.kv.get('key1'))
        self.assertNotIn('key1', self.kv)
        
        # Test TTL with None (no expiration)
        self.kv.set('key2', 'value2', ttl=None)
        self.assertEqual(self.kv.get('key2'), 'value2')
    
    def test_lru_eviction(self):
        """Test least-recently-used eviction policy."""
        # Fill the cache
        self.kv.set('key1', 'value1')
        self.kv.set('key2', 'value2')
        self.kv.set('key3', 'value3')
        
        # Access key1 to make it most recently used
        self.kv.get('key1')
        
        # Add one more item - key2 should be evicted (least recently used)
        self.kv.set('key4', 'value4')
        
        self.assertIsNone(self.kv.get('key2'))  # Should be evicted
        self.assertEqual(self.kv.get('key1'), 'value1')  # Should still be there
        self.assertEqual(self.kv.get('key3'), 'value3')
        self.assertEqual(self.kv.get('key4'), 'value4')
        
        # Test that __len__ works correctly with eviction
        self.assertEqual(len(self.kv), 3)
    
    def test_persistence(self):
        """Test that data persists in memory as long as instance exists."""
        # Create a store and add some data
        kv = KVStore(max_size=10)
        kv.set('key1', 'value1')
        kv.set('key2', 'value2')
        
        # Data should be accessible
        self.assertEqual(kv.get('key1'), 'value1')
        self.assertEqual(kv.get('key2'), 'value2')
        self.assertEqual(len(kv), 2)
        
        # Clear the store
        kv.clear()
        self.assertEqual(len(kv), 0)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with create_kv_store() as kv:
            kv['key1'] = 'value1'
            self.assertEqual(kv['key1'], 'value1')
        
        # The store should be cleared when the context manager exits
        self.assertEqual(len(kv), 0)
        self.assertIsNone(kv.get('key1'))
    
    def test_clear(self):
        """Test clearing the store."""
        self.kv.set('key1', 'value1')
        self.kv.set('key2', 'value2')
        
        self.assertEqual(len(self.kv), 2)
        self.kv.clear()
        self.assertEqual(len(self.kv), 0)
        self.assertIsNone(self.kv.get('key1'))
        self.assertIsNone(self.kv.get('key2'))
    
    def test_serialization(self):
        """Test serialization of different data types."""
        test_data = [
            'string',
            123,
            3.14,
            True,
            None,
            {'key': 'value'},
            [1, 2, 3],
            (4, 5, 6),
            {'set', 'of', 'values'}
        ]
        
        for i, value in enumerate(test_data):
            key = f'key{i}'
            self.kv.set(key, value)
            self.assertEqual(self.kv.get(key), value)
    
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        kv = KVStore()
        
        # Test getting non-existent key with default
        self.assertIsNone(kv.get('nonexistent'))
        self.assertEqual(kv.get('nonexistent', 'default'), 'default')
        
        # Test deleting non-existent key
        self.assertFalse(kv.delete('nonexistent'))
        
        # Test accessing non-existent key with []
        with self.assertRaises(KeyError):
            _ = kv['nonexistent']
    
    def test_cache_entry(self):
        """Test CacheEntry dataclass."""
        now = time.time()
        entry = CacheEntry(value='test', created_at=now, ttl=60)
        self.assertEqual(entry.value, 'test')
        self.assertEqual(entry.created_at, now)
        self.assertEqual(entry.ttl, 60)
        
        # Test with no TTL
        entry = CacheEntry(value='test', created_at=now)
        self.assertIsNone(entry.ttl)


class TestCreateKVStore(unittest.TestCase):
    def test_create_kv_store(self):
        """Test the create_kv_store convenience function."""
        # Test with defaults
        kv = create_kv_store()
        self.assertIsInstance(kv, KVStore)
        self.assertEqual(kv.max_size, 1000)
        kv.close()
        
        # Test with custom max size
        kv = create_kv_store(max_size=500)
        self.assertEqual(kv.max_size, 500)
        kv.close()


if __name__ == '__main__':
    unittest.main()