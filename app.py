import os
import webbrowser

import streamlit as st
import subprocess
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.tools.firecrawl import FirecrawlTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

st.set_page_config(page_title="AI-Powered Assistant", layout="wide")
st.title("AI-Powered Smart Assistant")

st.sidebar.title("🔍 Choose a Feature")
option = st.sidebar.radio(
    "Select an option:",
    ["Shopping Assistant 🛍️", "Finance & Stocks 📈", "General Knowledge 🔍", "Image Generator 🎨"]
)

# ---- Shopping Assistant ----
if option == "Shopping Assistant 🛍️":
    st.header("🛍️ Shopping Assistant")
    
    shopping_agent = Agent(
        name="Shopping Partner",
        model=Gemini(id="gemini-2.0-flash-exp"),
        instructions=[
            "You are a product recommender specializing in trusted platforms like Amazon, Flipkart, Myntra, and Nike.",
            "Ensure the product is available and meets user criteria (minimum 50% match).",
            "Clearly display key attributes (price, brand, features) in an easy-to-read format.",
        ],
        tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
    )

    user_query = st.text_area("Enter your shopping preferences:")
    
    if st.button("Find Products"):
        with st.spinner("Searching for products..."):
            response = shopping_agent.run(user_query)
            st.markdown("## 🛒 Recommended Products")
            st.markdown(response.content if hasattr(response, 'content') else str(response))

# ---- Finance & Stocks ----
elif option == "Finance & Stocks 📈":
    st.header("📈 Finance & Stock Market")

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True
            )
        ],
        instructions=[
            "Provide stock prices, trends, and financial metrics.",
            "Use tables to enhance readability.",
        ],
        markdown=True,
        debug_mode=True,
    )

    stock_question = st.text_area("Ask a financial or stock-related question:")

    if st.button("Get Finance Insights"):
        with st.spinner("Fetching financial insights..."):
            response = finance_agent.run(stock_question)
            st.markdown("## 📊 Finance Insights")
            st.markdown(response.content if hasattr(response, 'content') else str(response))

# ---- General Knowledge Search ----
elif option == "General Knowledge 🔍":
    st.header("Real Time LLM Feeder")

    web_agent = Agent(
        name="Web Agent",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[DuckDuckGo(search=True)],
        instructions=[
            "Provide detailed answers with multiple sources and citations.",
            "Use bullet points, headings, and structured formats.",
        ],
        markdown=True,
        debug_mode=True,
    )

    user_question = st.text_area("Ask any question:")

    if st.button("Search"):
        with st.spinner("Fetching response..."):
            response = web_agent.run(user_question)
            st.markdown("## 🌍 Answer")
            st.markdown(response.content if hasattr(response, 'content') else str(response))

# ---- Image Generator (Runs `run.bat`) ----

elif option == "Image Generator 🎨":
    st.header("🎨 AI Image Generator")

    st.write("Click below to open the image generator.")




# ---- Debug Info in Sidebar ----
st.sidebar.header("🔧 Debug Info")
st.sidebar.write("Debug mode is enabled for detailed responses.")
