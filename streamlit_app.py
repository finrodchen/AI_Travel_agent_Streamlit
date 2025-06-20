import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, YouTubeSearchTool
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition

# Page configuration
st.set_page_config(
    page_title="🌍 AI Travel Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

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
        weather_api_key = st.secrets.get("OPENWEATHERMAP_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
        if weather_api_key:
            os.environ["OPENWEATHERMAP_API_KEY"] = weather_api_key
            weather = OpenWeatherMapAPIWrapper()
            return weather.run(city)
        else:
            return f"Weather API key not available. Cannot get weather for {city}."
    except Exception as e:
        return f"Weather data unavailable for {city}. Error: {str(e)}"

@tool
def search_google(query: str) -> str:
    """Fetches details about attractions, restaurants, hotels, etc. from Google Serper API."""
    try:
        serper_api_key = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
        if serper_api_key:
            os.environ["SERPER_API_KEY"] = serper_api_key
            search_serper = GoogleSerperAPIWrapper()
            return search_serper.run(query)
        else:
            # Fallback to duck search if serper not available
            return search_duck(query)
    except Exception as e:
        return f"Google search unavailable, trying alternative search. Error: {str(e)}"

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
        # Get OpenAI API key from Streamlit secrets or environment
        openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets.")
            st.info("💡 Go to Settings → Secrets and add: OPENAI_API_KEY = \"your-key-here\"")
            return None
        
        # Initialize OpenAI model - FIXED: Removed SecretStr
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=2000,
            api_key=openai_api_key  # Direct string, not SecretStr
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
        st.error(f"❌ Error initializing travel agent: {str(e)}")
        st.info("💡 Check your API keys and internet connection")
        return None

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>🌍 AI Travel Agent & Expense Planner</h1>
        <p>Plan your perfect trip with real-time data and detailed cost calculations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status Check
    st.sidebar.header("📡 API Status")
    
    # Check API keys
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    serper_key = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
    weather_key = st.secrets.get("OPENWEATHERMAP_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
    
    if openai_key:
        st.sidebar.success("✅ OpenAI API")
    else:
        st.sidebar.error("❌ OpenAI API Missing")
        st.sidebar.info("Required for the app to work")
    
    if serper_key:
        st.sidebar.success("✅ Serper API")
    else:
        st.sidebar.warning("⚠️ Serper API Missing")
        st.sidebar.info("Will use DuckDuckGo as fallback")
        
    if weather_key:
        st.sidebar.success("✅ Weather API")
    else:
        st.sidebar.warning("⚠️ Weather API Missing")
        st.sidebar.info("Weather feature won't work")
    
    # Main content
    st.header("💬 Travel Query")
    
    # Example queries
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
Calculate total trip cost in both USD and INR."""
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
        
        if not openai_key:
            st.error("❌ OpenAI API key is required. Please add it to Streamlit secrets.")
            return
        
        # Initialize travel agent
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
                st.info("💡 Try refreshing the page or check your internet connection")

if __name__ == "__main__":
    main()
