"""
eDEX-UI Integration for OpenHands Enhanced
Provides sci-fi themed interface with real-time system monitoring.
"""

import asyncio
import logging
import os
import json
import subprocess
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime
import websockets
import threading

logger = logging.getLogger(__name__)

class EdexUIIntegration:
    """Integration with eDEX-UI for sci-fi themed interface."""
    
    def __init__(self):
        self.enabled = os.getenv("EDEX_UI_ENABLED", "false").lower() == "true"
        self.port = int(os.getenv("EDEX_UI_PORT", "3001"))
        self.world_map_api = os.getenv("WORLD_MAP_API_ENABLED", "true").lower() == "true"
        
        self.edex_process: Optional[subprocess.Popen] = None
        self.websocket_server = None
        self.connected_clients = set()
        
        # System metrics
        self.metrics_thread = None
        self.metrics_running = False
        
    async def start(self):
        """Start eDEX-UI integration."""
        if not self.enabled:
            logger.info("eDEX-UI integration disabled")
            return
            
        try:
            # Check if eDEX-UI is available
            edex_path = os.path.join(os.getcwd(), "edex-ui")
            if not os.path.exists(edex_path):
                logger.warning("eDEX-UI not found. Please run setup script to install.")
                return
                
            # Start eDEX-UI process
            await self._start_edex_ui()
            
            # Start WebSocket server for real-time communication
            await self._start_websocket_server()
            
            # Start metrics collection
            self._start_metrics_collection()
            
            logger.info("eDEX-UI integration started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start eDEX-UI integration: {e}")
            
    async def stop(self):
        """Stop eDEX-UI integration."""
        try:
            # Stop metrics collection
            self.metrics_running = False
            if self.metrics_thread:
                self.metrics_thread.join(timeout=5)
                
            # Stop WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
                
            # Stop eDEX-UI process
            if self.edex_process:
                self.edx_process.terminate()
                self.edx_process.wait(timeout=10)
                
            logger.info("eDEX-UI integration stopped")
            
        except Exception as e:
            logger.error(f"Error stopping eDEX-UI integration: {e}")
            
    async def _start_edex_ui(self):
        """Start the eDEX-UI process."""
        edex_path = os.path.join(os.getcwd(), "edex-ui")
        
        # Create OpenHands integration config
        config = {
            "openhands": {
                "api_url": f"http://localhost:8000",
                "websocket_url": f"ws://localhost:{self.port + 1}",
                "world_map_enabled": self.world_map_api,
                "metrics_interval": 1000
            }
        }
        
        config_path = os.path.join(edex_path, "openhands-config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        # Start eDEX-UI with custom configuration
        env = os.environ.copy()
        env["OPENHANDS_CONFIG"] = config_path
        env["PORT"] = str(self.port)
        
        self.edex_process = subprocess.Popen(
            ["npm", "start"],
            cwd=edex_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Started eDEX-UI on port {self.port}")
        
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time communication."""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            logger.info(f"eDEX-UI client connected: {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)
                logger.info(f"eDEX-UI client disconnected: {websocket.remote_address}")
                
        self.websocket_server = await websockets.serve(
            handle_client,
            "localhost",
            self.port + 1
        )
        
        logger.info(f"WebSocket server started on port {self.port + 1}")
        
    def _start_metrics_collection(self):
        """Start metrics collection thread."""
        self.metrics_running = True
        self.metrics_thread = threading.Thread(target=self._metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
    def _metrics_loop(self):
        """Metrics collection loop."""
        while self.metrics_running:
            try:
                metrics = self._collect_system_metrics()
                asyncio.run(self._broadcast_metrics(metrics))
                threading.Event().wait(1)  # Wait 1 second
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                threading.Event().wait(5)  # Wait longer on error
                
    def _collect_system_metrics(self) -> Dict:
        """Collect comprehensive system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            # Sort by CPU usage and take top 10
            processes = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:10]
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except:
                pass
                
            # GPU metrics (if available)
            gpu_metrics = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_metrics = {
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature
                    }
            except ImportError:
                pass
                
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq._asdict() if cpu_freq else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                    "io": disk_io._asdict() if disk_io else None
                },
                "network": network._asdict() if network else None,
                "processes": processes,
                "temperature": temperature,
                "gpu": gpu_metrics,
                "uptime": psutil.boot_time()
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
            
    async def _broadcast_metrics(self, metrics: Dict):
        """Broadcast metrics to all connected clients."""
        if not self.connected_clients:
            return
            
        message = json.dumps({
            "type": "metrics",
            "data": metrics
        })
        
        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending metrics to client: {e}")
                disconnected.add(client)
                
        # Remove disconnected clients
        self.connected_clients -= disconnected
        
    async def send_openhands_event(self, event_type: str, data: Dict):
        """Send OpenHands-specific events to eDEX-UI."""
        if not self.connected_clients:
            return
            
        message = json.dumps({
            "type": "openhands_event",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending event to client: {e}")
                disconnected.add(client)
                
        self.connected_clients -= disconnected
        
    def get_world_map_data(self) -> Dict:
        """Get world map data for global visualization."""
        if not self.world_map_api:
            return {"enabled": False}
            
        # In a real implementation, this would fetch data from various APIs
        # For now, return mock data
        return {
            "enabled": True,
            "api_requests": {
                "total": 1234,
                "by_region": {
                    "North America": 456,
                    "Europe": 321,
                    "Asia": 234,
                    "Other": 223
                }
            },
            "active_connections": 42,
            "server_locations": [
                {"name": "US East", "lat": 40.7128, "lng": -74.0060, "load": 65},
                {"name": "US West", "lat": 37.7749, "lng": -122.4194, "load": 45},
                {"name": "Europe", "lat": 51.5074, "lng": -0.1278, "load": 78},
                {"name": "Asia", "lat": 35.6762, "lng": 139.6503, "load": 52}
            ]
        }
        
    def get_status(self) -> Dict:
        """Get eDEX-UI integration status."""
        return {
            "enabled": self.enabled,
            "port": self.port,
            "world_map_api": self.world_map_api,
            "edex_running": self.edex_process is not None and self.edex_process.poll() is None,
            "websocket_running": self.websocket_server is not None,
            "connected_clients": len(self.connected_clients),
            "metrics_running": self.metrics_running
        }

# Global eDEX-UI integration instance
edex_integration = EdexUIIntegration()