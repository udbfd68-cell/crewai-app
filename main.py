import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from crewai import LLM, Agent, Crew, Task, Process

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="CrewAI API", version="1.0.0", lifespan=lifespan)


class CrewRequest(BaseModel):
    topic: str


class CrewResponse(BaseModel):
    result: str


def build_crew(topic: str) -> Crew:
    llm = LLM(
        model="openrouter/google/gemini-2.0-flash-001",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    researcher = Agent(
        role="Chercheur Senior",
        goal=f"Trouver des informations clés sur : {topic}",
        backstory="Tu es un chercheur expérimenté avec un talent pour dénicher les informations les plus pertinentes.",
        verbose=False,
        llm=llm,
    )

    writer = Agent(
        role="Rédacteur",
        goal=f"Rédiger un résumé clair et concis sur : {topic}",
        backstory="Tu es un rédacteur talentueux qui transforme des recherches complexes en textes accessibles.",
        verbose=False,
        llm=llm,
    )

    research_task = Task(
        description=f"Recherche approfondie sur le sujet suivant : {topic}. Trouve les points clés, tendances et faits importants.",
        expected_output="Une liste structurée des points clés et informations importantes.",
        agent=researcher,
    )

    write_task = Task(
        description=f"À partir des recherches, rédige un résumé complet sur : {topic}.",
        expected_output="Un résumé bien structuré de 3-5 paragraphes.",
        agent=writer,
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=False,
    )


@app.get("/")
def health():
    return {"status": "ok", "service": "crewai-app"}


@app.post("/crew/run", response_model=CrewResponse)
def run_crew(request: CrewRequest):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY non configurée")
    try:
        crew = build_crew(request.topic)
        result = crew.kickoff()
        return CrewResponse(result=str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
