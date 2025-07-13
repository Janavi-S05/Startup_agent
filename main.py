import streamlit as st
import openai

# ---- Streamlit UI Setup ----
st.set_page_config(page_title="Startup Trend Analysis (Groq)", page_icon="📈")
st.title("🚀 AI Startup Trend Analysis Agent")
st.caption("Powered by LLaMA3-70B via Groq — Blazingly fast + FREE!")

# ---- Input Fields ----
topic = st.text_input("📌 Enter your startup interest area (e.g., AI in Healthcare):")
groq_api_key = st.sidebar.text_input("🔐 Enter your Groq API Key", type="password")

# ---- Run on Button Click ----
if st.button("Generate Analysis"):
    if not groq_api_key or not topic:
        st.warning("Please enter both the topic and your Groq API key.")
    else:
        try:
            # ---- Initialize OpenAI-style Client for Groq ----
            client = openai.OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            prompt = f"""
You are a startup trend analyst.
Analyze the latest market trends and suggest startup opportunities in the domain: {topic}.

Please provide:
1. A concise summary of recent developments or innovations
2. 2-3 promising startup ideas with brief descriptions
3. Actionable advice for new founders interested in this space
"""

            with st.spinner("Analyzing with LLaMA3-70B via Groq..."):
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a helpful startup trend analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=800,
                )

                result = response.choices[0].message.content
                st.subheader("📊 Trend Analysis & Startup Opportunities")
                st.write(result)

        except Exception as e:
            st.error(f"❌ An error occurred:\n\n{e}")
else:
    st.info("💡 Enter your topic and Groq API key to get started.")
