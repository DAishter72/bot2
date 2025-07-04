from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
app = FastAPI()

# instancia model
model = init_chat_model("command-r-plus", model_provider="cohere")


class Bot(BaseModel):
    query: str


messages = [
    SystemMessage(content="""Eres Mark, un experto en videojuegos y consolas.
                  Te encargas de responder preguntas sobre videojuegos, consolas y temas relacionados.
                  Responde de manera clara y concisa, agregando alguna broma o comentario divertido si es apropiado.
                  No respondas preguntas que no se relacionen con videojuegos o consolas.
                  Si no sabes la respuesta, di que no lo sabes y sugiere buscar en Google.
                  No inventes respuestas y si hacen preguntas sobre algo relacionado al videojuego,
                  pero que existe en la vida real,
                  solo responde si es relevante para el videojuego.""")
]


@app.post("/bot/")
async def bot(q: Bot):
    question = q.query
    messages.append(HumanMessage(content=question))
    response = model.invoke(messages)
    messages.append(response)
    return {"response": response.content}
