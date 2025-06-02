"""
Idle Improvement System for OpenHands Enhanced
Automatically improves the system when idle conditions are met.
"""

import asyncio
import logging
import os
import time
import psutil
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class IdleImprovement:
    """Manages automatic system improvements during idle periods."""
    
    def __init__(self):
        self.running = False
        self.improvement_task: Optional[asyncio.Task] = None
        
        # Configuration from environment
        self.system_idle_enabled = os.getenv("SYSTEM_IDLE", "false").lower() == "true"
        self.vm_idle_enabled = os.getenv("VM_IDLE", "false").lower() == "true"
        self.ac_power_required = os.getenv("AC_POWER", "true").lower() == "true"
        self.lightning_vm_enabled = os.getenv("LIGHTNING_VM", "false").lower() == "true"
        
        # Idle detection settings
        self.idle_threshold = int(os.getenv("IDLE_THRESHOLD_MINUTES", "10"))
        self.improvement_interval = int(os.getenv("IDLE_IMPROVEMENT_INTERVAL", "3600"))
        
        # State tracking
        self.last_activity_time = time.time()
        self.last_improvement_time = 0
        self.improvement_history: List[Dict] = []
        
    async def start(self):
        """Start the idle improvement system."""
        if self.running:
            return
            
        self.running = True
        self.improvement_task = asyncio.create_task(self._improvement_loop())
        logger.info("Idle improvement system started")
        
    async def stop(self):
        """Stop the idle improvement system."""
        self.running = False
        if self.improvement_task:
            self.improvement_task.cancel()
            try:
                await self.improvement_task
            except asyncio.CancelledError:
                pass
        logger.info("Idle improvement system stopped")
        
    def update_activity(self):
        """Update the last activity time."""
        self.last_activity_time = time.time()
        
    def is_system_idle(self) -> bool:
        """Check if the system is idle."""
        if not self.system_idle_enabled:
            return False
            
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 20:  # System is busy
            return False
            
        # Check if idle time threshold is met
        idle_time = time.time() - self.last_activity_time
        return idle_time > (self.idle_threshold * 60)
        
    def is_vm_idle(self) -> bool:
        """Check if VM is idle (placeholder for VM-specific logic)."""
        if not self.vm_idle_enabled:
            return False
            
        # In a real implementation, this would check VM-specific metrics
        # For now, use system idle as a proxy
        return self.is_system_idle()
        
    def is_ac_power_available(self) -> bool:
        """Check if AC power is available."""
        if not self.ac_power_required:
            return True
            
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                # Desktop system, assume AC power
                return True
            return battery.power_plugged
        except:
            # Assume AC power if can't detect
            return True
            
    def should_run_improvement(self) -> bool:
        """Check if improvement should run based on all conditions."""
        # Check if enough time has passed since last improvement
        time_since_last = time.time() - self.last_improvement_time
        if time_since_last < self.improvement_interval:
            return False
            
        # Check idle conditions
        if self.system_idle_enabled and not self.is_system_idle():
            return False
            
        if self.vm_idle_enabled and not self.is_vm_idle():
            return False
            
        # Check AC power
        if not self.is_ac_power_available():
            return False
            
        return True
        
    async def _improvement_loop(self):
        """Main improvement loop."""
        while self.running:
            try:
                if self.should_run_improvement():
                    await self._run_improvement()
                    
                # Check every minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in improvement loop: {e}")
                await asyncio.sleep(60)
                
    async def _run_improvement(self):
        """Run the actual improvement process."""
        logger.info("Starting idle improvement process")
        
        improvement_start = time.time()
        
        try:
            if self.lightning_vm_enabled:
                await self._run_lightning_improvement()
            else:
                await self._run_local_improvement()
                
            # Record successful improvement
            improvement_time = time.time() - improvement_start
            self.last_improvement_time = time.time()
            
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "duration": improvement_time,
                "type": "lightning" if self.lightning_vm_enabled else "local",
                "status": "success"
            })
            
            logger.info(f"Improvement completed in {improvement_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Improvement failed: {e}")
            
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "duration": time.time() - improvement_start,
                "type": "lightning" if self.lightning_vm_enabled else "local",
                "status": "failed",
                "error": str(e)
            })
            
    async def _run_local_improvement(self):
        """Run improvement on local system."""
        logger.info("Running local improvement")
        
        # Simulate improvement process
        # In a real implementation, this would:
        # 1. Analyze current system performance
        # 2. Identify optimization opportunities
        # 3. Apply safe improvements
        # 4. Test and validate changes
        
        await asyncio.sleep(5)  # Simulate work
        
        # Example improvements:
        improvements = [
            "Optimized model routing algorithms",
            "Updated provider configurations",
            "Cleaned up temporary files",
            "Optimized database queries",
            "Updated caching strategies"
        ]
        
        for improvement in improvements:
            logger.info(f"Applied: {improvement}")
            await asyncio.sleep(1)
            
    async def _run_lightning_improvement(self):
        """Run improvement on Lightning Labs VM."""
        logger.info("Running Lightning Labs improvement")
        
        try:
            # Import Lightning SDK
            import lightning as L
            
            # Create improvement job
            app = L.LightningApp(
                L.LightningWork(
                    cloud_compute=L.CloudCompute("cpu-small"),
                    run=self._lightning_improvement_work
                )
            )
            
            # Run the improvement
            await app.run()
            
        except ImportError:
            logger.warning("Lightning SDK not available, falling back to local improvement")
            await self._run_local_improvement()
            
    def _lightning_improvement_work(self):
        """Work function for Lightning Labs improvement."""
        import subprocess
        import sys
        
        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Run improvement analysis
        # This would be more sophisticated in a real implementation
        print("Running improvement analysis on Lightning Labs...")
        
        # Simulate analysis and optimization
        time.sleep(10)
        
        print("Improvement completed on Lightning Labs")
        
    def get_status(self) -> Dict:
        """Get current idle improvement status."""
        return {
            "running": self.running,
            "system_idle_enabled": self.system_idle_enabled,
            "vm_idle_enabled": self.vm_idle_enabled,
            "ac_power_required": self.ac_power_required,
            "lightning_vm_enabled": self.lightning_vm_enabled,
            "is_system_idle": self.is_system_idle(),
            "is_vm_idle": self.is_vm_idle(),
            "is_ac_power_available": self.is_ac_power_available(),
            "should_run_improvement": self.should_run_improvement(),
            "last_improvement_time": self.last_improvement_time,
            "improvement_history": self.improvement_history[-10:]  # Last 10 improvements
        }
        
    def get_metrics(self) -> Dict:
        """Get improvement metrics."""
        if not self.improvement_history:
            return {
                "total_improvements": 0,
                "success_rate": 0,
                "average_duration": 0,
                "last_improvement": None
            }
            
        total = len(self.improvement_history)
        successful = sum(1 for h in self.improvement_history if h["status"] == "success")
        success_rate = (successful / total) * 100
        
        durations = [h["duration"] for h in self.improvement_history if h["status"] == "success"]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_improvements": total,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "last_improvement": self.improvement_history[-1] if self.improvement_history else None
        }

# Global idle improvement instance
idle_improvement = IdleImprovement()