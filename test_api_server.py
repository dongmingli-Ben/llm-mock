import requests
import threading
import time

HOST = "localhost"
PORT = 8000

def test_health():
    payload = {
            "prompt": "some random prompts",
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 100,
            "ignore_eos": True,
            "stream": False,
        }
    
    response = requests.post(f"http://{HOST}:{PORT}/generate",
                             json=payload)
    assert response.status_code == 200
    print(response.json())

def test_concurrent_requests():
    payload = {
            "prompt": "some random prompts",
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 100,
            "ignore_eos": True,
            "stream": False,
        }
    
    def make_request():
        response = requests.post(f"http://{HOST}:{PORT}/generate",
                                 json=payload)
        assert response.status_code == 200
        print(response.json())
    
    t1 = threading.Thread(target=make_request, daemon=True)
    t1.start()
    time.sleep(0.5)
    t2 = threading.Thread(target=make_request, daemon=True)
    t2.start()
    t1.join()
    t2.join()


if __name__ == "__main__":
    # test_health()
    test_concurrent_requests()
