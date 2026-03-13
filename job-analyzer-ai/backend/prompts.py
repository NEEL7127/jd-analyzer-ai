# Centralized prompt definitions


def build_prompt(job_description: str, resume_text: str | None, context: list[str]) -> str:
    context_block = "\n".join(context)
    resume_block = resume_text or "(no resume provided)"

    return f"""
You are a career assistant analyzing a job description.
analyze the job description and also tell that what need to be added in resume.

Job Description:\n{job_description}\n
Resume:\n{resume_block}\n
Context:\n{context_block}\n
Provide a concise summary, strengths, gaps, and recommendations.
""".strip()
