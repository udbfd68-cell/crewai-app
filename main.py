import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CrewAI App</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.container{max-width:700px;width:90%;padding:2rem}
h1{font-size:2.5rem;text-align:center;margin-bottom:.5rem;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#94a3b8;margin-bottom:2rem}
.input-group{display:flex;gap:.75rem;margin-bottom:1.5rem}
input{flex:1;padding:.85rem 1rem;border-radius:12px;border:1px solid #334155;background:#1e293b;color:#e2e8f0;font-size:1rem;outline:none;transition:border .2s}
input:focus{border-color:#60a5fa}
button{padding:.85rem 1.5rem;border-radius:12px;border:none;background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff;font-size:1rem;font-weight:600;cursor:pointer;transition:opacity .2s;white-space:nowrap}
button:hover{opacity:.9}
button:disabled{opacity:.5;cursor:not-allowed}
.result{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.5rem;white-space:pre-wrap;line-height:1.7;min-height:100px;display:none}
.loading{text-align:center;color:#94a3b8;display:none}
.loading .spinner{display:inline-block;width:20px;height:20px;border:3px solid #334155;border-top-color:#60a5fa;border-radius:50%;animation:spin 1s linear infinite;margin-right:.5rem;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.error{color:#f87171}
</style>
</head>
<body>
<div class="container">
<h1>CrewAI</h1>
<p class="subtitle">Posez une question, deux agents IA collaborent pour y r&eacute;pondre.</p>
<div class="input-group">
<input type="text" id="topic" placeholder="Ex: Les tendances de l'IA en 2026" autocomplete="off">
<button id="btn" onclick="runCrew()">Lancer</button>
</div>
<div class="loading" id="loading"><span class="spinner"></span>Les agents travaillent... (peut prendre 30-60s)</div>
<div class="result" id="result"></div>
</div>
<script>
async function runCrew(){
const topic=document.getElementById('topic').value.trim();
if(!topic)return;
const btn=document.getElementById('btn');
const loading=document.getElementById('loading');
const result=document.getElementById('result');
btn.disabled=true;
loading.style.display='block';
result.style.display='none';
result.className='result';
try{
const res=await fetch('/crew/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic})});
const data=await res.json();
if(!res.ok)throw new Error(data.detail||'Erreur serveur');
result.textContent=data.result;
result.style.display='block';
}catch(e){
result.textContent='Erreur: '+e.message;
result.className='result error';
result.style.display='block';
}finally{
btn.disabled=false;
loading.style.display='none';
}
}
document.getElementById('topic').addEventListener('keydown',e=>{if(e.key==='Enter')runCrew()});
</script>
</body>
</html>"""


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
