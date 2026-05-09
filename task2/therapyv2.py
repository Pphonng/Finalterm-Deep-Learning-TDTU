from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import torch
import pickle
from transformers import AutoTokenizer
from datetime import datetime
from modelsv2 import (
    EMOTION_SCORE, TherapyChatbot, Retriever, EmotionModel, LLM, WeeklySummary,
    device
)

BASE_PATH = "./mychat"

def init_bot() -> TherapyChatbot:
    with open(f"{BASE_PATH}/label_maps.pkl", "rb") as f:
        maps = pickle.load(f)

    id2emotion  = maps["id2emotion"]
    id2behavior = maps["id2behavior"]

    retriever     = Retriever.load(BASE_PATH)
    emo_tokenizer = AutoTokenizer.from_pretrained(f"{BASE_PATH}/emo_tokenizer")

    emotion_model = EmotionModel(n_emo=32, n_behavior=len(id2behavior))
    emotion_model.load_state_dict(
        torch.load(f"{BASE_PATH}/emotion_model.pt", map_location=device)
    )
    emotion_model.to(device).eval()

    return TherapyChatbot(
        retriever=retriever,
        emotion_model=emotion_model,
        emo_tokenizer=emo_tokenizer,
        id2emotion=id2emotion,
        id2behavior=id2behavior,
        llm=LLM(),
    )

bot = init_bot()

api = FastAPI(title="MindSpace API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# --- ENDPOINTS ---

@api.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        return {"reply": "Bạn muốn nói gì với mình không? 😊", "emotion": None}

    try:
        reply = bot.chat(req.message, verbose=True)
        
        current_emo = "Neutral"
        if bot.mood_tracker.memory:
            last_entry = bot.mood_tracker.memory[-1]
            current_emo = last_entry.get("emotion") or last_entry.get("label") or "Neutral"

        return {
            "reply": reply, 
            "emotion": current_emo
        }

    except Exception as e:
        print(f" Lỗi Chat: {e}")
        return {"reply": f"Lỗi hệ thống: {e}", "emotion": None}

@api.post("/api/reset")
def reset_endpoint():
    bot.reset()
    return {"status": "success", "message": "Đã reset lịch sử hội thoại."}

@api.get("/api/mood-timeline")
def mood_timeline():
    """Trả về dữ liệu thô để Frontend vẽ biểu đồ (ví dụ bằng Chart.js)"""
    if not bot.mood_tracker.memory:
        return {"points": []}

    points = []
    for i, item in enumerate(bot.mood_tracker.memory):
        emotion = item.get("emotion", "neutral").lower()
       
        score = EMOTION_SCORE.get(emotion, 0) 
        points.append({
            "step": i + 1,
            "score": score,
            "emotion": emotion
        })
    return {"points": points}
@api.get("/api/emotion-graph")
def get_emotion_graph():
    emotions = []
    for item in bot.mood_tracker.memory:
        emo = item.get("emotion") or item.get("label") or "Neutral"
        emotions.append(emo)
    
    if len(emotions) < 2:
        return {"emotions": [], "status": "need_more_data"}
        
    return {"emotions": emotions, "status": "success"}
@api.get("/api/weekly")
def weekly_endpoint():
    """Trả về báo cáo dạng văn bản và dữ liệu biểu đồ tuần"""
    if not bot.mood_tracker.memory:
        return {"days": [], "report": "Chưa có đủ dữ liệu để tổng kết."}
        
    summary = WeeklySummary(bot.mood_tracker)
    report_text = summary.text_report()

    day_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    
    if hasattr(summary, "daily_scores") and summary.daily_scores:
        days_data = [{"label": day_names[i % 7], "score": s} for i, s in enumerate(summary.daily_scores)]
    else:
        # fake
        days_data = [
            {"label": "T2", "score": 1}, {"label": "T3", "score": 2}, 
            {"label": "T4", "score": -1}, {"label": "T5", "score": 0},
            {"label": "T6", "score": 2.5}, {"label": "T7", "score": 1.5}, {"label": "CN", "score": 2}
        ]
    
    return {"report": report_text, "chart_data": days_data}

api.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    from pyngrok import ngrok
    
    NGROK_AUTH_TOKEN = "3A4vKXoIEkxBNJFr6ToaZWrk86r_27cfZisckuzptBErqxc6M"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    public_url = ngrok.connect(8000).public_url
    print("\n" + "="*55)
    print(f" 🌿 MindSpace Backend đang chạy tại: {public_url}")
    print(f" {public_url}/docs")
    print("="*55 + "\n")

    uvicorn.run(api, host="0.0.0.0", port=8000)