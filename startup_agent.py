import streamlit as st
import openai
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

# ---- Load Groq API key from .env ----
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---- Streamlit Setup ----
st.set_page_config(page_title="Startup Trend Agent with News", page_icon="💼")
st.title("💼 AI Startup Trend Analyzer + News")
st.caption("Real-time news + Groq LLaMA3 agents = Better insights, faster decisions.")

# ---- Inputs ----
topic = st.text_input("📌 Enter a startup interest area (e.g., Fintech, AI in Education):")

# ---- Groq client init ----
def get_groq_client(api_key):
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

# ---- Agent runner ----
def run_agent(client, system_role, user_prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.65,
        max_tokens=1000,
    )
    return response.choices[0].message.content

# ---- News Search Agent ----
def fetch_recent_news(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, region="wt-wt", max_results=max_results):
            results.append(f"- [{r['title']}]({r['url']})\n  {r['body']}")
    return "\n\n".join(results)

# ---- Button logic ----
if st.button("Generate Analysis"):
    if not groq_api_key:
        st.error("🚫 Groq API key not found. Please set it in your environment variables.")
    elif not topic:
        st.warning("Please enter a topic to analyze.")
    else:
        try:
            client = get_groq_client(groq_api_key)

            # Step 1: Search News
            with st.spinner("📰 News Collector Agent gathering real-time news..."):
                news_snippets = fetch_recent_news(topic)
                st.markdown("#### 🗞️ Latest News Articles Found")
                st.markdown(news_snippets)

            # Step 2: Summarize News
            with st.spinner("✍️ Summarizing news content..."):
                summarized_news = run_agent(
                    client,
                    "You are a startup-savvy news summarizer.",
                    f"Summarize the following news related to {topic}:\n\n{news_snippets}"
                )

            # Step 3: Run Multi-Agent Flow
            with st.spinner("🔍 Market Research Agent working..."):
                market_summary = run_agent(
                    client,
                    "You are a market research expert.",
                    f"Using this news summary:\n{summarized_news}\n\nWhat market trends and startup activity are emerging in {topic}?"
                )

            with st.spinner("💡 Idea Generator Agent working..."):
                idea_suggestions = run_agent(
                    client,
                    "You are a creative startup idea generator.",
                    f"Suggest 2–3 innovative startup ideas in {topic} based on the latest news and trends."
                )

            with st.spinner("🛠️ Execution Planner Agent working..."):
                roadmap = run_agent(
                    client,
                    "You are a startup mentor.",
                    f"Create a roadmap from idea → MVP → launch for a founder in {topic}."
                )

            with st.spinner("📢 GTM Strategy Agent working..."):
                gtm = run_agent(
                    client,
                    "You are a go-to-market expert.",
                    f"Suggest marketing and distribution strategies for startups in {topic}."
                )

            with st.spinner("⚠️ Risk Analyst Agent working..."):
                risks = run_agent(
                    client,
                    "You are a startup risk advisor.",
                    f"What are common risks in this space and how to avoid them?"
                )

            # Final Report
            st.subheader("📊 Complete Startup Analysis Report")
            st.markdown(f"### 📰 Summarized News\n{summarized_news}")
            st.markdown(f"### 🔍 Market Overview\n{market_summary}")
            st.markdown(f"### 💡 Startup Ideas\n{idea_suggestions}")
            st.markdown(f"### 🛠️ Roadmap\n{roadmap}")
            st.markdown(f"### 📢 Go-to-Market\n{gtm}")
            st.markdown(f"### ⚠️ Challenges & Risks\n{risks}")

        except Exception as e:
            st.error(f"❌ An error occurred:\n\n{e}")
else:
    st.info("💡 Enter a topic to run the full agentic analysis. No API key required from user.")

