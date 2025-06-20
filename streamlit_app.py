import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
from langchain.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, YouTubeSearchTool
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
import json

# Page configuration
st.set_page_config(
    page_title="🌍 AI Travel Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .cost-breakdown {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'travel_agent' not in st.session_state:
    st.session_state.travel_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom Tools
@tool
def addition(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def division(a: int, b: int) -> float:
    """Divide two integers."""
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b

@tool
def substraction(a: int, b: int) -> float:
    """Subtract two integers."""
    return a - b

@tool
def get_weather(city: str) -> str:
    """Fetches the current weather of the city from OpenWeatherMap."""
    try:
        weather = OpenWeatherMapAPIWrapper()
        return weather.run(city)
    except Exception as e:
        return f"Weather data unavailable for {city}. Error: {str(e)}"

@tool
def search_google(query: str) -> str:
    """Fetches details about attractions, restaurants, hotels, etc. from Google Serper API."""
    try:
        search_serper = GoogleSerperAPIWrapper()
        return search_serper.run(query)
    except Exception as e:
        return f"Search unavailable. Error: {str(e)}"

@tool
def search_duck(query: str) -> str:
    """Fetches details using DuckDuckGo search."""
    try:
        search_d = DuckDuckGoSearchRun()
        return search_d.invoke(query)
    except Exception as e:
        return f"Search unavailable. Error: {str(e)}"

@tool
def youtube_search(query: str) -> str:
    """Fetches YouTube videos about travel destinations."""
    try:
        youtubetool = YouTubeSearchTool()
        return youtubetool.run(query)
    except Exception as e:
        return f"YouTube search unavailable. Error: {str(e)}"

# Advanced calculation tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell for complex calculations. Input should be a valid python command.",
    func=python_repl.run,
)

def initialize_travel_agent():
    """Initialize the travel agent with all tools and configurations."""
    try:
        # Check for required API keys
        required_keys = ['OPENAI_API_KEY']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            st.error(f"Missing API keys: {', '.join(missing_keys)}")
            st.error("Please set up your API keys in the .env file")
            return None
        
        # Initialize OpenAI model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=SecretStr(openai_api_key)
        )
        
        # System prompt
        system_prompt = SystemMessage("""
        You are a professional AI Travel Agent. You MUST follow this EXACT process for every travel query:

        STEP 1: ALWAYS call get_weather tool first for the destination city

        STEP 2: ALWAYS call search_google or search_duck to find:
           - Hotels with specific prices per night
           - Top attractions with entry fees
           - Restaurants with price ranges
           - Transportation options with costs
           - CURRENCY CONVERSION: If user needs different currency, search for:
             "current exchange rate [from_currency] to [to_currency] today"

        STEP 3: ALWAYS use arithmetic tools (addition, multiply) to calculate:
           - Hotel cost = daily_rate × number_of_days
           - Total food cost = daily_food_budget × number_of_days
           - Total attraction costs = sum of all entry fees
           - Currency conversion = amount × exchange_rate (from search)
           - Grand total = hotel + food + attractions + transport

        STEP 4: ALWAYS call youtube_search for relevant travel videos

        STEP 5: Create detailed day-by-day itinerary with REAL costs from your searches

        MANDATORY RULES:
        - For currency conversion: SEARCH for current exchange rates, don't guess
        - Use ACTUAL data from tool calls, never make up prices
        - Show detailed cost breakdown with calculations
        - Include weather information from the weather tool
        - Provide YouTube video links from your search

        FORMAT your response as:
        ## 🌤️ Weather Information
        ## 💱 Currency Conversion  
        ## 🏛️ Attractions & Activities
        ## 🏨 Hotels & Accommodation
        ## 📅 Daily Itinerary
        ## 💰 Cost Breakdown
        ## 🎥 YouTube Resources
        ## 📋 Summary
        """)
        
        # Create tools list
        tools = [addition, multiply, division, substraction, get_weather, 
                search_google, search_duck, repl_tool, youtube_search]
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Create graph function
        def function_1(state: MessagesState):
            user_question = state["messages"]
            input_question = [system_prompt] + user_question
            response = llm_with_tools.invoke(input_question)
            return {"messages": [response]}
        
        # Build the graph
        builder = StateGraph(MessagesState)
        builder.add_node("llm_decision_step", function_1)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "llm_decision_step")
        builder.add_conditional_edges("llm_decision_step", tools_condition)
        builder.add_edge("tools", "llm_decision_step")
        
        # Compile the graph
        react_graph = builder.compile()
        return react_graph
        
    except Exception as e:
        st.error(f"Error initializing travel agent: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 AI Travel Agent & Expense Planner</h1>
        <p>Plan your perfect trip with real-time data and detailed cost calculations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Setup")
        
        # API Key Status
        st.subheader("📡 API Status")
        api_status = {}
        api_keys = {
            "OpenAI": "OPENAI_API_KEY",
            "Serper": "SERPER_API_KEY", 
            "Weather": "OPENWEATHERMAP_API_KEY"
        }
        
        for name, key in api_keys.items():
            if os.getenv(key):
                st.success(f"✅ {name} API")
                api_status[name] = True
            else:
                st.warning(f"⚠️ {name} API Missing")
                api_status[name] = False
        
        # Features
        st.subheader("🚀 Features")
        features = [
            "🌤️ Real-time Weather",
            "🏨 Hotel Price Calculator", 
            "💱 Currency Conversion",
            "🎯 Attraction Finder",
            "📅 Itinerary Generator",
            "💰 Cost Breakdown",
            "🎥 Travel Videos"
        ]
        
        for feature in features:
            st.markdown(f"<div class='feature-box'>{feature}</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Travel Query")
        
        # Pre-filled example queries
        example_queries = {
            "🏖️ Beach Vacation": """I want to visit Goa for 5 days in December.
My budget is 30,000 INR.
Get current weather for Goa.
Find hotels under 3,000 INR per night.
I want to know about beaches, water sports, and nightlife.
Calculate exact costs including food (500 INR per day).
Show me travel videos about Goa.""",
            
            "🌍 International Trip": """I want to visit Thailand for 4 days.
My budget is 800 USD.
Convert all costs to Indian Rupees.
Get current weather for Bangkok.
Find budget hotels under 30 USD per night.
Include street food and restaurant costs.
Show temple entry fees and transportation costs.
Calculate total trip cost in both USD and INR.""",
            
            "🏔️ Mountain Adventure": """Plan a 3-day trip to Manali.
Budget: 15,000 INR total.
Current weather in Manali?
Hotels under 2,000 INR per night.
Adventure activities and costs.
Local food expenses.
YouTube videos about Manali travel."""
        }
        
        selected_example = st.selectbox("🎯 Choose Example Query:", 
                                       ["Custom Query"] + list(example_queries.keys()))
        
        if selected_example != "Custom Query":
            query = st.text_area("✍️ Your Travel Query:", 
                                value=example_queries[selected_example],
                                height=200)
        else:
            query = st.text_area("✍️ Your Travel Query:", 
                                placeholder="E.g., I want to visit Paris for 7 days...",
                                height=200)
        
        # Process button
        if st.button("🚀 Plan My Trip", type="primary", use_container_width=True):
            if not query.strip():
                st.warning("Please enter your travel query!")
                return
            
            # Check if travel agent is initialized
            if st.session_state.travel_agent is None:
                with st.spinner("🔧 Initializing AI Travel Agent..."):
                    st.session_state.travel_agent = initialize_travel_agent()
            
            if st.session_state.travel_agent is None:
                st.error("❌ Failed to initialize travel agent. Please check your API keys.")
                return
            
            # Process the query
            with st.spinner("🤖 Planning your perfect trip..."):
                try:
                    response = st.session_state.travel_agent.invoke({
                        "messages": [HumanMessage(query)]
                    })
                    
                    # Display the response
                    if response and "messages" in response:
                        final_response = response["messages"][-1].content
                        st.success("✅ Your travel plan is ready!")
                        st.markdown(final_response)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "response": final_response
                        })
                    else:
                        st.error("❌ No response received. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing your request: {str(e)}")
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Display some quick info
        stats_container = st.container()
        with stats_container:
            st.metric("🎯 Queries Processed", len(st.session_state.chat_history))
            st.metric("🔧 Tools Available", "9")
            st.metric("🌍 Destinations", "Any City Worldwide")
        
        # Recent queries
        if st.session_state.chat_history:
            st.subheader("📝 Recent Queries")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
                with st.expander(f"Query {len(st.session_state.chat_history) - i}"):
                    st.text(chat["query"][:100] + "..." if len(chat["query"]) > 100 else chat["query"])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🌟 Made with ❤️ using LangChain, LangGraph & Streamlit | 
        🔗 <a href='https://github.com/yourusername/ai-travel-agent'>View on GitHub</a></p>
        <p>Created by Ashu Mishra - linkedin: <a href='https://www.linkedin.com/in/ashumish/'>https://www.linkedin.com/in/ashumish/</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
