from locust import HttpUser, task, between
import random

class RecommendationUser(HttpUser):
    wait_time = between(0.5, 2)
    
    user_ids = [
        '02703e242da346fc85f1aad21208d1103e62e705564c81ca2b6edf943032320a',
        '0350db312b33ec58e9b4e40c56998594d613476ec605b4c5a5a435e9ed172d70',
        '03fdb0bf2d9ff8ba23e1b4aef53709119aad5bc83691d89293a01a52b93d7370',
        '04dab48e5805e9c05272604ac78eb5eb941850ce307a7dd4bb5fe4652c0e4915',
        '0501552eaebc51b1199e9f900b5ba7bedf975624787d6bb42601293cb6743395',
        '057dddf8b97b949ae81c5e6f1269027ce573fca569b9c27c004ea42713dd46cb',
        '06878ff9b50fa7b950cb6f592d9d9369f1adc61ca6f62408aded14911cc91556',
        '06d23b72cac134851a761473b40c881c300814981b6b0a3c4692ddf5a93f2a48',
        '0bf4c6fd4e9d33f9bfb807bb78348cbf5c565846ff4006acf5c1b9aea77b0e54',
        '10411ac70c80005a535783ede440e0de5df6a1d7ff25fde5f58ac00ca97d8e55',
    ]
    
    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    @task
    def get_personal_recommendations(self):
        user_id = random.choice(self.user_ids)
        with self.client.get(
            f"/recommendations/personal_items/?user_id={user_id}",
            name="/recommendations/personal_items/",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Status {response.status_code}")
            elif not response.json().get("recommendations"):
                response.failure("Empty recommendations")

# locust -f locustfile.py --host http://127.0.0.1:8000
