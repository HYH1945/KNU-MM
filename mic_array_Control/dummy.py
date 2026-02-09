# dummy_sender.py
import redis, json, time

r = redis.Redis(host='localhost', port=6379, db=0)

while True:
    data = {"source": "TEST", "angle": 90, "msg": "Hello World!"}
    r.publish('security_events', json.dumps(data))
    print("Sent: ", data)
    time.sleep(1) # 1초마다 전송