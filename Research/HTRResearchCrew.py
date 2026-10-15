import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from crewai_tools import DuckDuckGoSearchTool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from dotenv import load_dotenv

# Ładowanie kluczy i konfiguracja modeli
load_dotenv()

# Gemini 1.5 Flash do researchu i pisania
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.2, # Niska temperatura dla bardziej precyzyjnych i spójnych odpowiedzi, ważne przy researchu
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Llama 3 - darmowa lokalna inteligencja (do precyzyjnej analizy technicznej)
llama_llm = Ollama(model="llama3",
                   verbose=True,
                   temperature=0.1) # Bardzo niska temperatura dla analizy technicznej

# Narzędzia
search_tool = DuckDuckGoSearchTool()
arxiv_tool = ArxivQueryRun()

# Zespół ekspertów (po Angielsku, ponieważ przeszukują internet w poszukiwaniu wiedzy i analizują głównie anglojęzyczne źródła)
scout = Agent(
    role='Lead Academic Researcher',
    goal='Identify and retrieve the most recent English-language papers on CRNN, CapsNet, and Transformers in HTR.',
    backstory='You are a world-class AI researcher specializing in deep learning and handwriting recognition.',
    tools=[search_tool, arxiv_tool],
    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)

analyst = Agent(
    role='Senior Technical Analyst',
    goal='Extract technical parameters, CER/WER metrics, and architectural innovations in English.',
    backstory='You are an expert in neural network architectures. You transform complex papers into structured data.',
    llm=llama_llm,
    verbose=True,
    allow_delegation=False
)

# Research
research_task = Task(
    description="""Perform an in-depth academic research on the following technical domains:
    
    1. **ResNet-CRNN and CapsNet Synergy:** 
       Investigate the 'geometric arbiter' role of Capsule Networks when placed after a ResNet-CRNN backbone. 
       Focus on how CapsNet handles spatial relationships and resolves character ambiguity in low-confidence 
       HTR sequences where standard CNNs fail due to pooling-induced information loss.

    2. **Byte-level vs. Word-level Transformers for Polish:** 
       Analyze why Byte-level tokenization is superior for morphologically rich and inflected languages like Polish. 
       Focus on sub-word feature extraction, handling of Polish diacritics, and mitigating the 
       'Out-of-Vocabulary' problem in semantic correction.

    3. **Uncertainty-Aware Active Learning in HTR:** 
       Research the distinction between aleatoric and epistemic uncertainty quantification in handwriting recognition. 
       Identify 'Hard Mining' strategies and 'Human-in-the-loop' frameworks that utilize model uncertainty 
       to prioritize samples for manual validation and experience replay.""",
       
    expected_output="""A structured technical intelligence report including:
    - Summaries of at least 5 key academic papers.
    - Specific mathematical/architectural advantages for each topic.
    - A list of BibTeX citations for direct use in the thesis bibliography.""",
    agent=scout
)

# Analiza (po polsku, ponieważ analizujemy dane i wyniki, które mogą być w języku polskim, a także dla lepszej spójności z kontekstem pracy magisterskiej)
analysis_task = Task(
    description="""Przeanalizuj dane od Scouta. Wyciągnij:
    - Konkretne wyniki CER/WER dla modeli hybrydowych.
    - Porównanie efektywności tokenizacji bajtowej vs słownej dla j. polskiego.
    - Metody implementacji niepewności w modelach bayesowskich.""",
    expected_output="Techniczne zestawienie danych i porównań w punktach.",
    agent=analyst,
    context=[research_task]
)

# Optymalizacja
optimization_task = Task(
    description="""Critically evaluate the current 3-stage pipeline:
    1. ResNet-CRNN (Sequential module)
    2. Capsule Network (Morphological arbiter)
    3. Byte-level Transformer (Semantic correction)
    
    Identify potential bottlenecks in this specific 'Cascade' logic. 
    Research and suggest 3 improvements, such as:
    - Diffusion Models for more realistic 'Ink-Physics' data augmentation.
    - Evidential Deep Learning as an alternative to Bayesian uncertainty for faster calibration.
    - Test-Time Augmentation strategies specifically for CapsNet-based verification.
    - Integration of Graph Neural Networks for better character-ligature relationship modeling.""",
    
    expected_output="""A critical review report suggesting 3 concrete technical upgrades with scientific justification.""",
    agent=analyst,
    context=[analysis_task]
)


# Uruchomienie całego procesu
htr_crew = Crew(
    agents=[scout, analyst],
    tasks=[research_task, analysis_task, optimization_task], 
    process=Process.sequential,
    memory=True, # Agenci pamiętają kontekst całej rozmowy, co jest ważne dla spójności i głębi analizy
    cache=True,  # Oszczędność czasu przy powtórnych zapytaniach
    verbose=True
)

print("Uruchamianie HTR Research Crew.")
result = htr_crew.kickoff()

# Zapis do pliku
with open("research_results.md", "w", encoding="utf-8") as f:
    f.write(str(result))

print('Sprawdź research_results.md')