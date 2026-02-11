from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sqlite3
import json
import datetime
import uvicorn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
app = FastAPI()

@app.get("/")
async def get_dashboard():
    # 파일이 실제로 존재하는지 확인하는 로직 추가 (디버깅용)
    if not os.path.exists(INDEX_PATH):
        return {"error": f"index.html을 찾을 수 없습니다. 예상 경로: {INDEX_PATH}"}
    return FileResponse(INDEX_PATH)

# --- DB 초기화 ---
def init_db():
    conn = sqlite3.connect("security_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            event_type TEXT,
            payload TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- 웹소켓 매니저 ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# --- API 엔드포인트 ---

# 1. AI 엔진으로부터 이벤트를 받는 곳
@app.post("/event")
async def receive_event(request: Request):
    data = await request.json()
    
    # 1. 시간 설정 (ISO 포맷 또는 읽기 편한 KST)
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # 2. 데이터 세분화 추출 (추출 안 되면 기본값 처리)
    payload = data.get('data', {})
    source = data.get('source', 'UNKNOWN')
    event_type = data.get('type', 'UNKNOWN')
    
    # 시각화를 위한 핵심 컬럼들
    priority = payload.get('priority', 'NORMAL')
    angle = payload.get('angle', -1)  # 각도 없으면 -1
    person_count = payload.get('count', 0)
    description = payload.get('situation', payload.get('text', ''))

    # 3. DB 저장 (더 많은 컬럼 사용)
    conn = sqlite3.connect("security_logs.db")
    cursor = conn.cursor()
    # 주의: 테이블 생성(init_db) 시 아래 컬럼들이 추가되어 있어야 합니다.
    cursor.execute("""
        INSERT INTO event_logs 
        (timestamp, source, event_type, priority, angle, person_count, description, payload) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp_str, source, event_type, priority, angle, person_count, description, json.dumps(payload)))
    
    conn.commit()
    conn.close()

    await manager.broadcast(data)
    return {"status": "success"}

# 2. 과거 로그 조회 (대시보드 초기 로딩용)
@app.get("/logs")
async def get_logs():
    conn = sqlite3.connect("security_logs.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM event_logs ORDER BY id DESC LIMIT 50")
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

# 3. 실시간 웹소켓 연결
@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # 연결 유지만 함
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)