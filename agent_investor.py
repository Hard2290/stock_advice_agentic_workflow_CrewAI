#Importing libraries
from crewai import Agent, Task
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from crewai import LLM
from crewai_tools import EXASearchTool
import os

# Setting up Gemini LLM
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_KEY") #Define this in .env file

#Defining Gemini LLM parameters
gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.2,
    api_key = GOOGLE_API_KEY,
    provider="google_gemini"
)


# Current date for context
Now = datetime.now()
Today = Now.strftime("%d-%b-%Y")

'''
# Defining a web search tool using DuckDuckGo
@tool("DuckDuckGo Search")
def search_tool(search_query: str):
    """Search the internet for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)
'''

#Defining a web search tool using EXASearch
@tool("EXASearch")
def search_tool(search_query: str):
    """Search the internet for information on a given topic"""
    return EXASearchTool(api_key=os.getenv("EXA_API_KEY")).run(search_query)

#Defining custom tools
from crewai.tools import tool
import json

# Import Yahoo Finance APIs to fetch stock market parameters
import yfinance as yf
from curl_cffi import requests
import time

session = requests.Session(impersonate="chrome")

# custom tool built to fetch current stock price
@tool ("Get current stock price")
def get_current_stock_price(symbol: str) -> str:
    """Use this function to get the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: The current stock price or error message.
    """
    try:
        time.sleep(0.5)
        stock = yf.Ticker(symbol, session=session)

        current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        return f"{current_price:.2f}" if current_price else f"Could not fetch current price for {symbol}"
    except Exception as e:
        return f"Error fetching current price for {symbol}: {e}"

# Custom tool built to fetch details and market performance paramters of the company's stock
@tool
def get_company_info(symbol: str):
    """Use this function to get company information and current financial snapshot for a given stock symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        JSON containing company profile and current financial snapshot.
    """
    try:
        company_info_full = yf.Ticker(symbol,session=session).info
        if company_info_full is None:
            return f"Could not fetch company info for {symbol}"

        company_info_cleaned = {
            "Name": company_info_full.get("shortName"),
            "Symbol": company_info_full.get("symbol"),
            "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
            "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
            "Sector": company_info_full.get("sector"),
            "Industry": company_info_full.get("industry"),
            "City": company_info_full.get("city"),
            "Country": company_info_full.get("country"),
            "EPS": company_info_full.get("trailingEps"),
            "P/E Ratio": company_info_full.get("trailingPE"),
            "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
            "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
            "50 Day Average": company_info_full.get("fiftyDayAverage"),
            "200 Day Average": company_info_full.get("twoHundredDayAverage"),
            "Employees": company_info_full.get("fullTimeEmployees"),
            "Total Cash": company_info_full.get("totalCash"),
            "Free Cash flow": company_info_full.get("freeCashflow"),
            "Operating Cash flow": company_info_full.get("operatingCashflow"),
            "EBITDA": company_info_full.get("ebitda"),
            "Revenue Growth": company_info_full.get("revenueGrowth"),
            "Gross Margins": company_info_full.get("grossMargins"),
            "Ebitda Margins": company_info_full.get("ebitdaMargins"),
        }
        return json.dumps(company_info_cleaned)
    except Exception as e:
        return f"Error fetching company profile for {symbol}: {e}"

# Custom tool built to get income statement of the company
@tool
def get_income_statements(symbol: str):

    """Use this function to get income statements for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    JSON containing income statements or an empty dictionary.
    """
    try:
        stock = yf.Ticker(symbol,session=session)
        financials = stock.financials
        return financials.to_json(orient="index")
    except Exception as e:
        return f"Error fetching income statements for {symbol}: {e}"

#Defining agents
from crewai import Agent

# Agent for gathering company news and information
news_info_explorer = Agent(
    role='News and Info Researcher',
    goal='Gather and provide the latest news and information about a company from the internet',
    #llm='gpt-4o',
    llm=gemini_llm,
    verbose=True,
    backstory=(
        'You are an expert researcher, who can gather detailed information about a company. '
        'Consider you are on: ' + Today
    ),
    tools=[search_tool],
    cache=True,
    max_iter=5,
)

# Agent for gathering financial data
data_fin_explorer = Agent(
    role='Data Researcher',
    goal='Gather and provide financial data and company information about a stock',
    #llm='gpt-4o',
    llm=gemini_llm,
    verbose=True,
    backstory=(
        'You are an expert researcher, who can gather detailed information about a company or stock. '
        'When using tools, use the stock symbol and add a suffix ".NS" to it. try with and without the suffix and see what works'
        'Consider you are on: ' + Today
    ),
    tools=[get_company_info, get_income_statements],
    cache=True,
    max_iter=5,
)

# Agent for analyzing data
analyst = Agent(
    role='Data Analyst',
    goal='Consolidate financial data, stock information, and provide a summary',
    #llm='gpt-4o',
    llm=gemini_llm,
    verbose=True,
    backstory=(
        'You are an expert in analyzing financial data, stock/company-related current information, and '
        'making a comprehensive analysis. Use Indian units for numbers (lakh, crore). '
        'Consider you are on: ' + Today
    ),
)

# Agent for financial recommendations
fin_expert = Agent(
    role='Financial Expert',
    goal='Considering financial analysis of a stock, make investment recommendations',
    #llm='gpt-4o',
    llm=gemini_llm,
    verbose=True,
    tools=[get_current_stock_price],
    max_iter=5,
    backstory=(
        'You are an expert financial advisor who can provide investment recommendations. '
        'Consider the financial analysis, current information about the company, current stock price, '
        'and make recommendations about whether to buy/hold/sell a stock along with reasons.'
        'When using tools, try with and without the suffix ".NS" to the stock symbol and see what works. '
        'Consider you are on: ' + Today
    ),
)

#Defining tasks
from crewai import Task

# Task to gather latest company news and business information
get_company_news = Task(
    description="Get latest news and business information about company: {stock}",
    expected_output="Latest news and business information about the company. Provide a summary also.",
    agent=news_info_explorer,
)

# Task to gather financial data of the company and its stock
get_company_financials = Task(
    description="Get financial data like income statements and other fundamental ratios for stock: {stock}",
    expected_output="Detailed information from income statement, key ratios for {stock}. "
                    "Indicate also about current financial status and trend over the period.",
    agent=data_fin_explorer,
)

# Task to analyze financial data and news
analyse = Task(
    description="Make thorough analysis based on given financial data and latest news of a stock",
    expected_output="Comprehensive analysis of a stock outlining financial health, stock valuation, risks, and news. "
                    "Mention currency information and number units in Indian context (lakh/crore).",
    agent=analyst,
    context=[get_company_financials, get_company_news],
    output_file='Analysis.md',
)

# Task to provide financial advice
advise = Task(
    description="Make a recommendation about investing in a stock, based on analysis provided and current stock price. "
                "Explain the reasons.",
    expected_output="Recommendation (Buy / Hold / Sell) of a stock backed with reasons elaborated."
                    "Response in Mark down format.",
    agent=fin_expert,
    context=[analyse],
    output_file='Fin_Recommendation.md',
)

#Setting up th Crew
from crewai import Crew, Process
from datetime import datetime

# Callback function to print a timestamp
def timestamp(Input):
    print(datetime.now())

# Defining the crew with agents and tasks in sequential process
crew = Crew(
    agents=[data_fin_explorer, news_info_explorer, analyst, fin_expert],
    tasks=[get_company_financials, get_company_news, analyse, advise],
    verbose=True,
    Process=Process.sequential,
    step_callback=timestamp,
)

#Running the crew to get results
# Run the crew with a specific stock
result = crew.kickoff(inputs={'stock': 'RELIANCE'})

# Print the final result
print("Final Result:", result)