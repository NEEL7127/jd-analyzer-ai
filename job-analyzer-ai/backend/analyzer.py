# backend/analyzer.py
# 
# KEY IMPROVEMENT: All 4 analyses run in PARALLEL
# using Python's asyncio + ThreadPoolExecutor
#
# Before: Call 1 → wait → Call 2 → wait → Call 3 → wait → Call 4 = 20s
# After:  Call 1 ↘
#         Call 2 → all run at same time → 6-8s ✅
#         Call 3 ↗
#         Call 4 ↗

import os
from groq                   import Groq
from dotenv                 import load_dotenv
from backend.rag_engine     import retrieve, get_full_context
from concurrent.futures     import ThreadPoolExecutor, as_completed

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


# ============================================
# HELPER: Call Groq
# ============================================
def call_groq(prompt: str, temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {
                "role"   : "system",
                "content": """You are an expert career coach and technical recruiter 
                with 10 years of experience in the Indian tech industry.
                You give honest, specific, actionable advice.
                also have deep knowledge of how ATS systems work and how to optimize resumes for them.
                You always base answers ONLY on the job description provided.
                Never make up information not present in the JD.
                Format your response with clear sections using these markers:
                Use ## for section headers
                Use • for bullet points  
                Use ✅ for positive/required items
                Use ❌ for missing/negative items
                Use 🔥 for most important items
                Use 💡 for tips and insights"""
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        max_tokens  = 1500,
        temperature = temperature,
    )
    return response.choices[0].message.content


# ============================================
# JOB 1: SKILL GAP ANALYSIS
# ============================================
def analyze_skills(collection_id: str) -> dict:
    skills_context = retrieve(
        "required skills technologies programming languages frameworks tools",
        collection_id, n_results=4
    )
    exp_context = retrieve(
        "years experience qualifications education degree",
        collection_id, n_results=2
    )
    context = f"{skills_context}\n\n{exp_context}"

    prompt = f"""
    Analyze this job description and extract ALL skill requirements.
    
    JOB DESCRIPTION CONTEXT:
    {context}
    
    Format your response EXACTLY like this:
    
    ## 🎯 Must-Have Skills
    • [skill] — [why critical, how often mentioned]
    • [skill] — [why critical]
    (list ALL must-have technical skills)
    
    ## 💡 Nice-to-Have Skills  
    • [skill] — [advantage it gives you]
    • [skill] — [advantage]
    
    ## 📋 Experience & Education
    ✅ Experience: [exact requirement]
    ✅ Education: [exact requirement]
    
    ## 🔥 Top 3 Most Critical Skills
    1. [skill] — [why this is #1 priority]
    2. [skill] — [why #2]
    3. [skill] — [why #3]
    
    ## 💬 Honest Assessment
    [2-3 sentences: how competitive is this role? what level candidate are they really looking for?]
    
    Only use information from the JD. Be specific and brutally honest.
    """

    return {
        "type"   : "skill_analysis",
        "content": call_groq(prompt, temperature=0.2),
        "status" : "success"
    }


# ============================================
# JOB 2: 30-DAY LEARNING ROADMAP
# ============================================
def generate_roadmap(collection_id: str) -> dict:
    skills_context = retrieve(
        "required skills must know technologies tools",
        collection_id, n_results=3
    )
    resp_context = retrieve(
        "responsibilities duties what you will do",
        collection_id, n_results=2
    )
    context = f"{skills_context}\n\n{resp_context}"

    prompt = f"""
    Based on this job description, create a realistic 30-day learning roadmap.
    
    JOB DESCRIPTION CONTEXT:
    {context}
    
    Format EXACTLY like this:
    
    ## 📅 Week 1 — Foundation (Days 1–7)
    • Day 1–2: [specific topic] | Resource: [free resource name]
    • Day 3–4: [specific topic] | Resource: [free resource]
    • Day 5–7: [mini project to build] | Goal: [what to build]
    🔥 Week 1 Target: [what you should be able to do]
    
    ## 📅 Week 2 — Core Skills (Days 8–14)
    • Day 8–10: [topic] | Resource: [resource]
    • Day 11–12: [topic] | Resource: [resource]
    • Day 13–14: [build this] | Goal: [outcome]
    🔥 Week 2 Target: [what you should have built]
    
    ## 📅 Week 3 — Advanced + Projects (Days 15–21)
    • Day 15–17: [topic] | Resource: [resource]
    • Day 18–19: [topic] | Resource: [resource]
    • Day 20–21: [project] | Goal: [outcome]
    🔥 Week 3 Target: [portfolio piece ready]
    
    ## 📅 Week 4 — Interview Prep (Days 22–30)
    • Day 22–24: [prep area] | How: [method]
    • Day 25–27: [mock projects] | Build: [what]
    • Day 28–30: [final prep] | Do: [what]
    🔥 Week 4 Target: [ready to apply]
    
    ## 🚀 Final Project Idea
    [One specific project that demonstrates ALL required skills from this JD]
    
    ## ⏱️ Honest Timeline
    [Is 30 days realistic? What's the real timeframe for a beginner vs intermediate?]
    
    Be specific — real tool names, real websites, real project ideas. Prioritize JD skills.
    """

    return {
        "type"   : "roadmap",
        "content": call_groq(prompt, temperature=0.4),
        "status" : "success"
    }


# ============================================
# JOB 3: INTERVIEW QUESTIONS
# ============================================
def generate_interview_questions(collection_id: str) -> dict:
    tech_context    = retrieve(
        "technical skills programming frameworks tools required",
        collection_id, n_results=3
    )
    culture_context = retrieve(
        "company culture team responsibilities collaboration",
        collection_id, n_results=2
    )
    context = f"{tech_context}\n\n{culture_context}"

    prompt = f"""
    Predict the most likely interview questions for this job.
    
    JOB DESCRIPTION CONTEXT:
    {context}
    
    Format EXACTLY like this:
    
    ## 💻 Technical Questions
    
    🔥 Q1: [most likely technical question]
    Why asked: [what skill it tests]
    💡 Key points to cover: [2-3 bullet hints]
    
    🔥 Q2: [technical question]
    Why asked: [reason]
    💡 Key points: [hints]
    
    Q3: [technical question]
    Why asked: [reason]
    💡 Key points: [hints]
    
    Q4: [technical question]
    Why asked: [reason]
    💡 Key points: [hints]
    
    Q5: [technical question]
    Why asked: [reason]
    💡 Key points: [hints]
    
    ## 🖥️ Coding / Assignment Round
    
    Q6: [likely coding problem or take-home]
    Why asked: [what it tests]
    
    Q7: [another coding challenge]
    Why asked: [reason]
    
    ## 🤝 Behavioral Questions
    
    Q8: [behavioral question]
    Why asked: [what they evaluate]
    💡 STAR tip: [how to structure answer]
    
    Q9: [behavioral question]
    💡 STAR tip: [structure]
    
    Q10: [behavioral question]
    💡 STAR tip: [structure]
    
    ## 🎯 Most Likely Opening Question
    [The question they almost certainly start with + exactly how to nail it]
    
    ## ❌ Red Flags to Avoid
    • [common mistake for this role]
    • [another mistake]
    • [another]
    
    Base ALL questions on this specific JD only.
    """

    return {
        "type"   : "interview_questions",
        "content": call_groq(prompt, temperature=0.3),
        "status" : "success"
    }


# ============================================
# JOB 4: RESUME TIPS
# ============================================
def generate_resume_tips(collection_id: str) -> dict:
    full_context = get_full_context(collection_id)

    prompt = f"""
    Analyze this job description and give specific resume optimization advice.
    
    JOB DESCRIPTION:
    {full_context[:2000]}
    
    Format EXACTLY like this:
    
    ## 🔑 ATS Keywords (Use These Exact Words)
    🔥 Critical: [keyword], [keyword], [keyword], [keyword]
    ✅ Important: [keyword], [keyword], [keyword]
    💡 Bonus: [keyword], [keyword]
    
    ## 📝 Perfect Resume Headline
    "[Write the exact headline they should use for THIS job]"
    
    ## 🛠️ Skills Section
    ✅ Add immediately: [skill1], [skill2], [skill3]
    ❌ Remove or hide: [irrelevant skills]
    💡 Order them as: [recommended order based on JD priority]
    
    ## 📊 Experience Bullets — Transform Yours
    ❌ WEAK: "Worked on machine learning projects"
    ✅ STRONG: "[rewritten with JD keywords, metrics, impact]"
    
    ❌ WEAK: "Built APIs using Python"
    ✅ STRONG: "[rewritten version]"
    
    ❌ WEAK: "Collaborated with team members"  
    ✅ STRONG: "[rewritten version]"
    
    ## 🚀 Projects to Highlight
    🔥 Must show: [type of project based on JD]
    ✅ Good to have: [another project type]
    💡 How to describe them: [tips]
    
    ## 👤 Resume Summary (Copy-Paste Ready)
    "[Write a complete 3-line summary optimized for this exact JD]"
    
    ## 📈 ATS Score Tips
    • [specific tip to improve ATS score]
    • [tip]
    • [tip]
    
    ## 🔥 #1 Thing Most Candidates Miss
    [The single most important resume tip specific to THIS JD]
    """

    return {
        "type"   : "resume_tips",
        "content": call_groq(prompt, temperature=0.3),
        "status" : "success"
    }


# ============================================
# JOB 5: CHAT WITH JD
# ============================================
def chat_with_jd(question: str, collection_id: str) -> dict:
    context = retrieve(question, collection_id, n_results=4)

    prompt = f"""
    Answer the user's question about this job description.
    
    JD CONTEXT:
    {context}
    
    USER QUESTION: {question}
    
    Rules:
    1. Answer ONLY from the JD context above
    2. If not in JD → say "This isn't mentioned in the JD"
    3. Be direct — 2-4 sentences max
    4. Use ✅ ❌ 🔥 💡 emojis where relevant
    5. If asked about qualifications → be honest
    """

    return {
        "type"    : "chat",
        "question": question,
        "answer"  : call_groq(prompt, temperature=0.2),
        "status"  : "success"
    }


# ============================================
# MASTER FUNCTION — PARALLEL EXECUTION
# Runs all 4 analyses at the SAME TIME
# ============================================
def run_full_analysis(collection_id: str) -> dict:
    """
    Run all 4 analyses in parallel using ThreadPoolExecutor.
    Instead of 20s sequential → 6-8s parallel ✅
    """

    results = {}

    # Define all 4 tasks
    tasks = {
        "skill_analysis"     : lambda: analyze_skills(collection_id),
        "learning_roadmap"   : lambda: generate_roadmap(collection_id),
        "interview_questions": lambda: generate_interview_questions(collection_id),
        "resume_tips"        : lambda: generate_resume_tips(collection_id),
    }

    # Run all tasks in parallel using thread pool
    # max_workers=4 → 4 threads run simultaneously
    with ThreadPoolExecutor(max_workers=4) as executor:

        # Submit all tasks at once
        future_to_key = {
            executor.submit(func): key
            for key, func in tasks.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result      = future.result()
                results[key] = result["content"]
                print(f"✅ {key} done!")
            except Exception as e:
                print(f"❌ {key} failed: {e}")
                results[key] = f"Analysis failed: {str(e)}"

    results["status"] = "success"
    return results


# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    from backend.rag_engine import index_jd

    test_jd = """
    AI/ML Engineer — Groww, Bangalore
    Required: Python 3+ years, scikit-learn, PyTorch, FastAPI, SQL, Git
    Nice to have: LLMs, RAG, LangChain, Docker, AWS, MLflow
    Responsibilities: Build ML models, deploy to production, write clean code
    Qualifications: B.Tech CS, freshers with strong projects welcome, 1-3 years
    Salary: 12-22 LPA | Bangalore Hybrid | 3 days remote
    """

    print("=" * 50)
    print("Testing PARALLEL analyzer...")
    print("=" * 50)

    import time
    index_result  = index_jd(test_jd)
    collection_id = index_result["collection_id"]

    start  = time.time()
    result = run_full_analysis(collection_id)
    end    = time.time()

    print(f"\n⚡ Total time: {end-start:.1f} seconds")
    print(f"✅ All 4 analyses complete!")
    print("\n--- SKILL ANALYSIS PREVIEW ---")
    print(result["skill_analysis"][:300])