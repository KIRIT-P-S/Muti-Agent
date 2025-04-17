import os
import webbrowser
import streamlit as st
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

if option == "Shopping Assistant 🛍️":
    st.header("🛍️ Shopping Assistant")

    user_location = st.text_input("📍 Enter your location (city or area):", placeholder="e.g., Coimbatore")

    shopping_agent = Agent(
        name="Shopping Partner",
        model=Gemini(id="gemini-2.0-flash-exp"),
        instructions=[
            "You are a smart shopping assistant that searches the web for products based on user preferences like product type, budget, and city.",
            "If the user query is general (like 'shoes under 2000 in Coimbatore'), split the recommendations into three categories: Men, Women, and Kids.",
            "Under each category, divide products into Online Stores and Offline Stores.",
            "For Online Stores:",
            "- List product name, price, website/store name, and delivery date to the specified location (if available).",
            "- Include trusted and emerging platforms (e.g., Amazon, Flipkart, Ajio, etc).",
            "For Offline Stores:",
            "- List product name, price, store name, and approximate address in the city mentioned.",
            "Use only public data found on the web.",
            "Only show 2-3 top results per section.",
            "Ensure product match is at least 70%.",
            "Use headings like: '### 👟 For Men', '#### 🛒 Online Stores', '#### 🏪 Offline Stores'.",
            "No unnecessary text — keep it clean and structured."
        ],
        tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
    )

    user_query = st.text_area("Enter your shopping preferences:", placeholder="e.g., I need a shoe under 2000 in Coimbatore")

    if st.button("Find Products"):
        if not user_query or not user_location:
            st.warning("Please enter both your shopping preferences and your location.")
        else:
            with st.spinner("Searching for products..."):
                full_query = f"{user_query.strip()} in {user_location.strip()}"
                response = shopping_agent.run(full_query)
                st.markdown("## 🛒 Recommended Products")
                st.markdown(response.content if hasattr(response, 'content') else str(response))

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

elif option == "Image Generator 🎨":
    st.header("🎨 AI Image Generator")

    st.write("Click below to open the image generator.")

    if st.button("Generate Image"):
        webbrowser.open("http://127.0.0.1:7860/")
        st.success("Opening Image Generator...")

st.sidebar.header("🔧 Debug Info")
st.sidebar.write("Debug mode is enabled for detailed responses.")

st.sidebar.markdown("### Share this App:")
st.sidebar.markdown("[Click Here to Visit the App](https://muti-agent.streamlit.app/)")
