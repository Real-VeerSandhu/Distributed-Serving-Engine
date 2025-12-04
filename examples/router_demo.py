"""
Router Demo - Interactive demonstration of the routing system.

This script demonstrates the router's capabilities including:
- Worker registration
- Request routing with hash-based sharding
- Health checks and failover
- Worker status monitoring
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import Router
from src.model_registry import ModelRegistry


class RouterDemo:
    """Interactive demo for the router."""
    
    def __init__(self, router: Router, registry: ModelRegistry):
        self.router = router
        self.registry = registry
        self.running = True
        self.commands = {
            "register": self.register_worker,
            "unregister": self.unregister_worker,
            "route": self.route_request,
            "health": self.check_health,
            "stats": self.show_stats,
            "workers": self.list_workers,
            "register_model": self.register_model,
            "add_shard": self.add_shard,
            "help": self.show_help,
            "exit": self.exit
        }
    
    async def start(self):
        """Start the interactive demo."""
        # Start the router
        await self.router.start()
        
        print("=" * 60)
        print("Router Demo - Distributed Inference Engine")
        print("=" * 60)
        print("\nType 'help' for available commands")
        print("Example workflow:")
        print("  1. register_model - Register a model")
        print("  2. add_shard - Add shards to the model")
        print("  3. register - Register workers")
        print("  4. route - Route requests")
        print("  5. stats - View statistics")
        print()
        
        while self.running:
            try:
                cmd = input("\n> ").strip().split(maxsplit=1)
                if not cmd:
                    continue
                
                command = cmd[0].lower()
                args = cmd[1] if len(cmd) > 1 else ""
                
                if command in self.commands:
                    if asyncio.iscoroutinefunction(self.commands[command]):
                        await self.commands[command](args)
                    else:
                        self.commands[command](args)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def register_worker(self, args: str):
        """Register a worker: register <worker_id> <host:port>"""
        parts = args.split()
        if len(parts) < 2:
            print("Usage: register <worker_id> <host:port>")
            print("Example: register worker-1 127.0.0.1:9001")
            return
        
        worker_id = parts[0]
        address = parts[1]
        
        # Validate address format
        try:
            host, port = address.split(':')
            int(port)  # Validate port is a number
        except ValueError:
            print("Error: Address must be in format 'host:port'")
            return
        
        self.router.register_worker(worker_id, address)
        print(f"Registered worker {worker_id} at {address}")
    
    def unregister_worker(self, args: str):
        """Unregister a worker: unregister <worker_id>"""
        if not args:
            print("Usage: unregister <worker_id>")
            return
        
        worker_id = args.strip()
        self.router.unregister_worker(worker_id)
        print(f"Unregistered worker {worker_id}")
    
    def route_request(self, args: str):
        """Route a request: route <model_name> <version> <request_key>"""
        parts = args.split()
        if len(parts) < 3:
            print("Usage: route <model_name> <version> <request_key>")
            print("Example: route my-model 1.0 user-123")
            return
        
        model_name = parts[0]
        version = parts[1]
        request_key = parts[2]
        
        shard = self.router.route_request(model_name, version, request_key)
        
        if shard:
            worker_info = self.router.get_worker_info(shard.worker_id)
            print(f"\n=== Routing Result ===")
            print(f"Request Key: {request_key}")
            print(f"Model: {model_name}:{version}")
            print(f"Shard ID: {shard.shard_id}")
            print(f"Worker ID: {shard.worker_id}")
            print(f"Worker Address: {self.router.get_worker_address(shard.worker_id)}")
            if worker_info:
                print(f"Worker Health: {worker_info['health']}")
        else:
            print(f"Error: Could not route request for {model_name}:{version}")
            print("Make sure the model is registered and has shards with registered workers")
    
    async def check_health(self, args: str):
        """Manually trigger health check: health [worker_id]"""
        if args:
            worker_id = args.strip()
            if worker_id not in self.router.workers:
                print(f"Error: Worker {worker_id} not found")
                return
            await self.router._check_worker_health(worker_id)
            worker_info = self.router.get_worker_info(worker_id)
            if worker_info:
                print(f"\n=== Health Check Result ===")
                print(f"Worker: {worker_id}")
                print(f"Health: {worker_info['health']}")
                print(f"Consecutive Failures: {worker_info['consecutive_failures']}")
        else:
            print("Running health check on all workers...")
            await self.router._check_all_workers()
            print("Health check complete. Use 'stats' to see results.")
    
    def show_stats(self, _: str = ""):
        """Show router statistics."""
        stats = self.router.get_stats()
        print("\n=== Router Statistics ===")
        print(f"Total Workers: {stats['total_workers']}")
        print(f"Healthy Workers: {stats['healthy_workers']}")
        print(f"Unhealthy Workers: {stats['unhealthy_workers']}")
        print(f"Unknown Workers: {stats['unknown_workers']}")
        print(f"Failover Enabled: {stats['failover_enabled']}")
        print(f"Health Check Interval: {stats['health_check_interval']}s")
        
        if stats['total_workers'] > 0:
            print("\n=== Worker Details ===")
            for worker_id in self.router.workers.keys():
                info = self.router.get_worker_info(worker_id)
                if info:
                    print(f"\nWorker: {worker_id}")
                    print(f"  Address: {info['address']}")
                    print(f"  Health: {info['health']}")
                    print(f"  Failures: {info['consecutive_failures']}")
    
    def list_workers(self, _: str = ""):
        """List all registered workers."""
        if not self.router.workers:
            print("No workers registered")
            return
        
        print("\n=== Registered Workers ===")
        for worker_id, worker in self.router.workers.items():
            print(f"{worker_id}: {worker.address} ({worker.health.name})")
    
    def register_model(self, args: str):
        """Register a model: register_model <name> <version> <path>"""
        parts = args.split(maxsplit=2)
        if len(parts) < 3:
            print("Usage: register_model <name> <version> <path>")
            print("Example: register_model my-model 1.0 /path/to/model")
            return
        
        model_name = parts[0]
        version = parts[1]
        model_path = parts[2]
        
        self.registry.register_model(
            model_name=model_name,
            version=version,
            model_path=model_path,
            input_schema={"input": "string"},
            output_schema={"output": "string"},
            batch_size=1,
            max_batch_size=32
        )
        print(f"Registered model {model_name}:{version}")
    
    def add_shard(self, args: str):
        """Add a shard: add_shard <model_name> <version> <shard_id> <worker_id>"""
        parts = args.split()
        if len(parts) < 4:
            print("Usage: add_shard <model_name> <version> <shard_id> <worker_id>")
            print("Example: add_shard my-model 1.0 0 worker-1")
            return
        
        model_name = parts[0]
        version = parts[1]
        try:
            shard_id = int(parts[2])
        except ValueError:
            print("Error: shard_id must be an integer")
            return
        worker_id = parts[3]
        
        try:
            shard = self.registry.add_shard(
                model_name=model_name,
                version=version,
                shard_id=shard_id,
                worker_id=worker_id
            )
            print(f"Added shard {shard_id} for {model_name}:{version} on {worker_id}")
        except ValueError as e:
            print(f"Error: {e}")
    
    def show_help(self, _: str = ""):
        """Show help message."""
        print("\n=== Available Commands ===")
        print("  register <worker_id> <host:port>")
        print("                         - Register a worker")
        print("  unregister <worker_id> - Unregister a worker")
        print("  route <model> <version> <key>")
        print("                         - Route a request")
        print("  health [worker_id]     - Check worker health")
        print("  stats                  - Show router statistics")
        print("  workers                - List all workers")
        print("  register_model <name> <version> <path>")
        print("                         - Register a model")
        print("  add_shard <model> <version> <shard_id> <worker_id>")
        print("                         - Add a shard to a model")
        print("  help                   - Show this help")
        print("  exit                   - Exit the demo")
    
    def exit(self, _: str = ""):
        """Exit the demo."""
        print("Shutting down router...")
        self.running = False


async def main():
    """Run the router demo."""
    parser = argparse.ArgumentParser(description="Router Demo")
    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=5.0,
        help="Health check interval in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--health-check-timeout",
        type=float,
        default=2.0,
        help="Health check timeout in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Max consecutive failures before marking unhealthy (default: 3)"
    )
    parser.add_argument(
        "--no-failover",
        action="store_true",
        help="Disable automatic failover"
    )
    
    args = parser.parse_args()
    
    # Create registry and router
    registry = ModelRegistry()
    router = Router(
        registry=registry,
        health_check_interval=args.health_check_interval,
        health_check_timeout=args.health_check_timeout,
        max_consecutive_failures=args.max_failures,
        failover_enabled=not args.no_failover
    )
    
    # Create and run demo
    demo = RouterDemo(router, registry)
    
    try:
        await demo.start()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        await router.stop()


if __name__ == "__main__":
    asyncio.run(main())

