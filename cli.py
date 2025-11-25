from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress_bar import ProgressBar

class Dashboard:
    def __init__(self, sim):
        self.sim = sim
        self.layout = self._create_layout()

    def _create_layout(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(name="nodes", ratio=2),
            Layout(name="status", ratio=1)
        )
        return layout

    def _render_node_row(self, node):
        gpu_usage = node.gpu_load
        kv_usage = node.kv_used / node.kv_capacity
        
        gpu_color = "green"
        if gpu_usage > 0.7:
            gpu_color = "red"
        elif gpu_usage > 0.4:
            gpu_color = "yellow"
            
        kv_color = "green"
        if kv_usage > 0.7:
            kv_color = "red"
        elif kv_usage > 0.4:
            kv_color = "yellow"
            
        return [
            f"[bold]{node.id}[/]" + (" [red](active)[/]" if node.processing_request else ""),
            f"[bold {gpu_color}]{gpu_usage*100:3.0f}%[/] {self._progress_bar(gpu_usage, 10)}",
            f"[bold {kv_color}]{node.kv_used/1000:.1f}/{node.kv_capacity/1000:.1f}GB[/]",
            f"{len(node.request_queue)} waiting"
        ]
        
    def _progress_bar(self, percentage, width=10):
        filled = '█' * int(percentage * width)
        empty = '░' * (width - len(filled))
        return f"{filled}{empty}"

    def render(self):
        # Update layout
        self.layout["header"].update(Panel(
            "[bold blue]InferMesh[/] - Distributed LLM Inference Simulator | "
            f"[green]Nodes: {len(self.sim.nodes)}[/] | "
            f"[yellow]Auto-gen: {'ON' if self.sim.auto_generate else 'OFF'}",
            style="white on blue"
        ))
        
        # Nodes table
        nodes_table = Table(show_header=True, header_style="bold magenta")
        nodes_table.add_column("Node")
        nodes_table.add_column("GPU Load")
        nodes_table.add_column("KV Cache")
        nodes_table.add_column("Status")
        
        for node in sorted(self.sim.nodes, key=lambda n: int(n.id[1:])):
            nodes_table.add_row(*self._render_node_row(node))
            
        # Status panel
        status_text = Text()
        status_text.append("\n[bold]Controls:[/]\n")
        status_text.append("• [bold]a[/]: Add request\n")
        status_text.append("• [bold]s[/]: Toggle auto-gen\n")
        status_text.append("• [bold]q[/]: Quit\n\n")
        status_text.append(f"[bold]Requests:[/] {self.sim.request_id} total\n")
        
        self.layout["nodes"].update(Panel(nodes_table, title="Cluster Nodes"))
        self.layout["status"].update(Panel(status_text, title="Status"))
        
        return self.layout

