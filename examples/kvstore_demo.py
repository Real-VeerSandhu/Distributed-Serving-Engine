"""
Interactive KVCache CLI for AI Inference.

This script provides a command-line interface to interact with the KVCache.
"""

import json
import time
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kvstore import KVCache, CacheEntry

class KVCacheCLI:
    """Interactive command-line interface for KVCache."""
    
    def __init__(self, cache: KVCache):
        self.cache = cache
        self.running = True
        self.commands = {
            "get": self.get_value,
            "set": self.set_value,
            "batch_get": self.batch_get,
            "batch_set": self.batch_set,
            "delete": self.delete_key,
            "list": self.list_keys,
            "clear": self.clear,
            "stats": self.show_stats,
            "ttl": self.set_ttl,
            "help": self.show_help,
            "exit": self.exit
        }
    
    def start(self):
        """Start the interactive CLI."""
        print(f"KVCache CLI (Max Size: {self.cache.max_size}, Policy: {self.cache.eviction_policy.upper()})")
        print("Type 'help' for available commands")
        
        while self.running:
            try:
                cmd = input("\n> ").strip().split(maxsplit=1)
                if not cmd:
                    continue
                    
                command = cmd[0].lower()
                args = cmd[1] if len(cmd) > 1 else ""
                
                if command in self.commands:
                    self.commands[command](args)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")

    def get_value(self, args: str):
        """Get a value from the cache."""
        if not args:
            print("Usage: get <key>")
            return
            
        key = args.strip('"\'')
        value = self.cache.get(key, "Key not found or expired")
        print(f"Value: {json.dumps(value, indent=2)}")
    
    def set_value(self, args: str):
        """Set a key-value pair in the cache."""
        parts = args.split(maxsplit=2)
        if len(parts) < 2:
            print("Usage: set <key> <value> [ttl_seconds]")
            return
            
        key = parts[0].strip('"\'')
        try:
            value = json.loads(parts[1])
            ttl = float(parts[2]) if len(parts) > 2 else None
            self.cache.set(key, value, ttl=ttl)
            print(f"Set '{key}' successfully")
        except json.JSONDecodeError:
            print("Error: Value must be valid JSON")
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")

    def batch_get(self, args: str):
        """Get multiple values at once."""
        try:
            keys = json.loads(args)
            if not isinstance(keys, list):
                print("Error: Expected a JSON array of keys")
                return
                
            results = self.cache.batch_get(keys)
            print("Results:")
            for key, value in results.items():
                print(f"  {key}: {json.dumps(value, indent=2)}")
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")

    def batch_set(self, args: str):
        """Set multiple key-value pairs at once."""
        try:
            items = json.loads(args)
            if not isinstance(items, dict):
                print("Error: Expected a JSON object with key-value pairs")
                return
                
            self.cache.batch_set(items)
            print(f"Successfully set {len(items)} items")
        except json.JSONDecodeError:
            print("Error: Invalid JSON input")

    def delete_key(self, args: str):
        """Delete a key from the cache."""
        if not args:
            print("Usage: delete <key>")
            return
            
        key = args.strip('"\'')
        if self.cache.delete(key):
            print(f"Deleted key '{key}'")
        else:
            print(f"Key '{key}' not found")
    
    def list_keys(self, _: str = ""):
        """List all keys in the cache."""
        # Note: This is a simple implementation that gets all keys
        # For large caches, you might want to implement pagination
        keys = list(self.cache.cache.keys())
        if keys:
            print("Keys in cache:")
            for key in keys:
                entry = self.cache.cache[key]
                ttl = f" (TTL: {entry.ttl}s)" if entry.ttl else ""
                print(f"- {key}{ttl}")
        else:
            print("Cache is empty")
    
    def clear(self, _: str = ""):
        """Clear all keys from the cache."""
        confirm = input("Are you sure you want to clear the cache? (y/N) ").lower()
        if confirm == 'y':
            self.cache.clear()
            print("Cache cleared")
    
    def set_ttl(self, args: str):
        """Set TTL for a key."""
        parts = args.split()
        if len(parts) < 2:
            print("Usage: ttl <key> <seconds>")
            return
            
        key = parts[0].strip('"\'')
        try:
            ttl = float(parts[1])
            if key in self.cache.cache:
                self.cache.cache[key].ttl = ttl
                print(f"Set TTL for '{key}' to {ttl} seconds")
            else:
                print(f"Key '{key}' not found")
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
    
    def show_stats(self, _: str = ""):
        """Show cache statistics."""
        stats = self.cache.get_stats()
        print("\n=== Cache Statistics ===")
        print(f"Size: {stats['size']}/{stats['max_size']} items")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Evictions: {stats['evictions']}")
        print(f"Eviction Policy: {stats['eviction_policy'].upper()}")
    
    def show_help(self, _: str = ""):
        """Show help message."""
        print("\nAvailable commands:")
        print("  get <key>               - Get value for key")
        print("  set <key> <value> [ttl] - Set key-value pair with optional TTL")
        print("  batch_get [key1,key2]   - Get multiple values (JSON array)")
        print("  batch_set {k:v, ...}    - Set multiple key-value pairs (JSON object)")
        print("  delete <key>            - Delete a key")
        print("  list                    - List all keys")
        print("  clear                   - Clear all keys")
        print("  ttl <key> <seconds>     - Set TTL for a key")
        print("  stats                   - Show cache statistics")
        print("  help                    - Show this help message")
        print("  exit                    - Exit the CLI")
    
    def exit(self, _: str = ""):
        """Exit the CLI."""
        print("Goodbye!")
        self.running = False

def main():
    """Run the KVCache CLI."""
    parser = argparse.ArgumentParser(description="Interactive KVCache CLI")
    parser.add_argument("--max-size", type=int, default=1000,
                      help="Maximum number of items to store")
    parser.add_argument("--policy", choices=["lru", "lfu", "fifo"], default="lru",
                      help="Cache eviction policy (default: lru)")
    parser.add_argument("--ttl", type=float, help="Default TTL in seconds")
    
    args = parser.parse_args()
    
    try:
        # Create and initialize the KVCache
        cache = KVCache(
            max_size=args.max_size,
            eviction_policy=args.policy,
            default_ttl=args.ttl
        )
        
        # Start the CLI
        cli = KVCacheCLI(cache)
        cli.start()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()