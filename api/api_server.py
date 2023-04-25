from config import attribute
from threading import Thread
from logger.logger import logger
from api.api import Model_Setter, Model_Getter, Model_AI
from api.api import setter, getter, ai_action
from fastapi import BackgroundTasks, Request, FastAPI
import uvicorn

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()
@app.get("/")
def health_check():
    return "AI Engine Server Running"


@app.post("/ai_engine/ai")
async def response_get(body: Request, background_tasks: BackgroundTasks):
    try:
        body = await body.json()
        if dict(body)['id'] == 'ai_engine.001':
            body = Model_AI.parse_obj(body)
            background_tasks.add_task(ai_action, body)
            return {"Success": True}
        else:
            return {"Success": False}
    except:
        return {"Success": False}

@app.post("/ai_engine/ai_model_list")
async def response_get(body: Request, background_tasks: BackgroundTasks):
    try:
        body = await body.json()
        if dict(body)['id'] == 'ai_engine.001':
            body = Model_AI.parse_obj(body)
            background_tasks.add_task(getter, body)
            return {"Success": True}
        else:
            return {"Success": False}
    except:
        return {"Success": False}

def serve():
    uvicorn.run(app, host="0.0.0.0", port=int(attribute.PORT))


