"""
Locust Load Testing Script for Urban Sound Classification API
Simulates flood of prediction requests to test system performance
"""

from locust import HttpUser, task, between, events
import os
import random
import time
from pathlib import Path

TEST_AUDIO_DIR = "data/test"


class UrbanSoundUser(HttpUser):
    """
    Simulates users making prediction requests
    """
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts"""
        self.audio_files = self._get_test_files()
        if not self.audio_files:
            print("Warning: No test audio files found!")
    
    def _get_test_files(self):
        """Get list of test audio files"""
        audio_dir = Path(TEST_AUDIO_DIR)
        if audio_dir.exists():
            return list(audio_dir.glob("**/*.wav"))
        return []
    
    @task(10)
    def predict_audio(self):
        """
        Main prediction task (highest weight = 10)
        Simulates single audio prediction
        """
        if not self.audio_files:
            return
        
        # Select random audio file
        audio_file = random.choice(self.audio_files)
        
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file.name, f, 'audio/wav')}
            
            with self.client.post(
                "/predict",
                files=files,
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Failed with status {response.status_code}")
    
    @task(3)
    def batch_predict(self):
        """
        Batch prediction task (weight = 3)
        Simulates batch prediction with multiple files
        """
        if not self.audio_files:
            return
        
        # Select 3-5 random files
        batch_size = random.randint(3, min(5, len(self.audio_files)))
        selected_files = random.sample(self.audio_files, batch_size)
        
        files = []
        for audio_file in selected_files:
            with open(audio_file, 'rb') as f:
                files.append(('files', (audio_file.name, f.read(), 'audio/wav')))
        
        with self.client.post(
            "/batch-predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(2)
    def check_health(self):
        """
        Health check task (weight = 2)
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed")
    
    @task(1)
    def get_metrics(self):
        """
        Get metrics task (weight = 1)
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics fetch failed")
    
    @task(1)
    def get_classes(self):
        """
        Get classes task (weight = 1)
        """
        with self.client.get("/classes", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Classes fetch failed")


class QuickStartUser(HttpUser):
    """
    Quick start user for testing API availability
    """
    
    wait_time = between(0.5, 1)
    
    @task
    def health_check(self):
        """Simple health check"""
        self.client.get("/health")


# Event hooks for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """
    Custom event listener to track request metrics
    """
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when load test starts
    """
    print("=" * 60)
    print("🚀 Starting Urban Sound API Load Test")
    print("=" * 60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when load test stops
    """
    print("\n" + "=" * 60)
    print("✅ Load Test Completed")
    print("=" * 60)
    
    # Print summary statistics
    stats = environment.runner.stats
    
    print("\n📊 SUMMARY STATISTICS:")
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min Response Time: {stats.total.min_response_time:.2f}ms")
    print(f"Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests per Second: {stats.total.total_rps:.2f}")
    print("=" * 60)


# Custom configuration for different load scenarios
class LightLoadUser(UrbanSoundUser):
    """Simulates light load (10-50 users)"""
    wait_time = between(2, 5)


class MediumLoadUser(UrbanSoundUser):
    """Simulates medium load (50-200 users)"""
    wait_time = between(1, 3)


class HeavyLoadUser(UrbanSoundUser):
    """Simulates heavy load (200-1000 users)"""
    wait_time = between(0.5, 2)


if __name__ == "__main__":
    print("""
    Urban Sound Classification API - Load Testing
    
    Run this script using Locust:
    
    # Light Load Test (10 users, spawn 1 per second)
    locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 1
    
    # Medium Load Test (50 users, spawn 5 per second)
    locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5
    
    # Heavy Load Test (200 users, spawn 10 per second)
    locust -f locustfile.py --host=http://localhost:8000 --users 200 --spawn-rate 10
    
    # Headless mode with CSV output
    locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 \\
           --run-time 5m --headless --csv=results/load_test
    
    # Web UI mode (access at http://localhost:8089)
    locust -f locustfile.py --host=http://localhost:8000
    """)
