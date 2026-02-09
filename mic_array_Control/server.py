import os
import aiosqlite
import json
import redis.asyncio as redis
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "security_logs.db")

# 2. 최신 표준: Lifespan 관리 (startup/shutdown 통합)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 서버 시작 시 DB 초기화
    print(f">>> [DB] 경로 확인: {DB_PATH}")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                type TEXT,
                angle REAL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
    print(">>> [DB] 연결 및 테이블 준비 완료!")
    yield

app = FastAPI(lifespan=lifespan)
# html 대시보드
@app.get("/")
async def get_dashboard():
    file_path = os.path.join(BASE_DIR, "index.html")

    if not os.path.exists(file_path):
        return {"error": f"파일을 찾을 수 없습니다. {file_path}"}
    return FileResponse(file_path)


# 데이터 규격
from pydantic import BaseModel
class SecurityEvent(BaseModel):
    source: str
    type: str
    data: dict

# 서버 시작 시 DB 테이블 생성

# redis 서버로 보내고, db 저장
@app.post("/publish")
async def publish_event(event: SecurityEvent):
    # HTTP POST로 보낸 데이터를 Redis로
    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    await r.publish('security_events', event.json())
    angle = event.data.get("angle", 0.0)
    confidence = event.data.get("confidence", 0.0)

    # 로그 DB 저장
    async with aiosqlite.connect(DB_PATH) as db:
        try:
            await db.execute(
                "INSERT INTO logs (source, type, angle, confidence) VALUES (?, ?, ?, ?)",
                (event.source, event.type, float(angle), float(confidence))
            )
            await db.commit()
        except Exception as e:
            print(f">>> DB 저장 에러 {e}")
    
    await r.aclose()
    return {"status": "success"}

# 과거 로그 50개 
@app.get("/logs")
async def get_recent_logs():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows] if rows else []
    except Exception as e:
        print(f">>> 로그 저장 에러 {e}")
        return {"error": str(e)}

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> 클라이언트가 웹소켓에 연결되었습니다.")
    
    # 비동기 Redis 연결
    r = redis.from_url("redis://localhost:6379", decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe('security_events')

    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                # Redis 데이터를 브라우저로
                data_str = message['data']
                await websocket.send_text(data_str)
                try:
                    data = json.loads(data_str)
                    async with aiosqlite.connect(DB_PATH) as db:
                        # 데이터 구조에 맞춰 추출
                        angle = data.get("target_pan") or data.get("angle") or 0.0
                        conf = data.get("confidence", 1.0)
                        

                        await db.execute(
                            "INSERT INTO logs (source, type, angle, confidence) VALUES (?, ?, ?, ?)",
                            (data.get("source", "UNKNOWN"), data.get("type", "EVENT"), float(angle), float(conf))
                        )

                        await db.commit()
                except Exception as e:
                    print(f">>> 서버 로그 자동 저장 에러: {e}")
    except WebSocketDisconnect:
        print(">>> 클라이언트 연결 종료")
    finally:
        await pubsub.unsubscribe('security_events')
        await r.aclose()

if __name__ == "__main__":
    # 포트 번호와 경로를 다시 한번 확인하세요
    uvicorn.run(app, host="0.0.0.0", port=8000)



    