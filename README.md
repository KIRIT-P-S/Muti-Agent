# AI-Powered Multi-Agent Assistant

A modular, LLM-driven multi-agent assistant that autonomously handles real-world tasks including EV charging station lookup, smart shopping assistance, question answering, financial data analysis, and AI image generation — all through a clean, interactive Streamlit UI.

##  Features

- **🔌 EV Charging Station Finder**  
  Search nearby EV charging stations with name, address, and charger types using real-time web search.

- **🛍️ Smart Shopping Assistant**  
  Get product recommendations from both online and offline stores based on budget, location, and preferences.

- **🔍 General Question Answering**  
  Ask any question and receive structured, trustworthy answers using LLMs and DuckDuckGo search.

- **📈 Financial Insights & Stock Analysis**  
  View real-time stock prices, fundamentals, and analyst recommendations powered by YFinance.

- **🎨 AI Image Generator**  
  Launches a local or external AI image generation tool (e.g., Stable Diffusion) via UI interaction.

## 🧠 How It Works

- Prompt-engineered behaviors
- Tool-augmented reasoning
- Contextual LLM responses

Agents utilize real-time tools:
- 🔍 DuckDuckGo API for general and location-based search
- 🔥 Firecrawl API for shopping product scraping
- 📊 YFinance Tools for financial insights

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLMs:** Groq (LLaMA-3), Google Gemini
- **Agent Framework:** PHI
- **APIs/Tools:** DuckDuckGo, Firecrawl, YFinance
- **Deployment:** Streamlit Cloud / Localhost
