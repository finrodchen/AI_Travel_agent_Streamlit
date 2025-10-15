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

# 頁面設定
st.set_page_config(
    page_title="🌍 AI 旅遊規劃師",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 載入環境變數
load_dotenv()

# 初始化 Session State
if 'travel_agent' not in st.session_state:
    st.session_state.travel_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 自訂工具 (工具名稱和描述保持英文，以利 LLM 準確調用)
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
    """初始化旅遊規劃師，包含所有工具和組態。"""
    try:
        # 從 Streamlit secrets 或環境變數取得 OpenAI API key
        openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            st.error("❌ 找不到 OpenAI API 密鑰。請將其加入 Streamlit secrets。")
            st.info("💡 前往 Settings → Secrets 並新增：OPENAI_API_KEY = \"your-key-here\"")
            return None
        
        # 初始化 OpenAI 模型
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=2000,
            api_key=openai_api_key
        )
        
        # 系統提示 (System Prompt) - **已中文化**
        system_prompt = SystemMessage("""
        您是一位專業的 AI 旅遊規劃師。您必須對每個旅遊查詢遵循以下**精確的**流程：

        步驟 1：**總是**先調用 `get_weather` 工具查詢目的地的天氣。

        步驟 2：**總是**調用 `search_google` 或 `search_duck` 來尋找：
            - 飯店及每晚特定價格
            - 頂級景點及門票費用
            - 餐廳及價格範圍
            - 交通選項及費用
            - **貨幣換算**：如果使用者需要不同貨幣，請搜尋："current exchange rate [from_currency] to [to_currency] today" (今日 [來源貨幣] 兌 [目標貨幣] 匯率)。

        步驟 3：**總是**使用算術工具 (`addition`, `multiply`) 計算：
            - 飯店費用 = 每日價格 × 天數
            - 總食物費用 = 每日食物預算 × 天數
            - 總景點費用 = 所有門票費用的總和
            - 貨幣換算 = 金額 × 匯率（從搜尋結果取得）
            - 總計 = 飯店 + 食物 + 景點 + 交通

        步驟 4：**總是**調用 `Youtube` 查詢相關的旅遊影片。

        步驟 5：使用您搜尋到的**真實費用**創建詳細的逐日行程。

        **強制規則：**
        - 貨幣換算：**必須**搜尋當前匯率，不可猜測。
        - 使用來自工具調用的**實際**數據，絕不編造價格。
        - 顯示包含計算細節的詳細費用明細。
        - 包含天氣工具提供的天氣資訊。
        - 提供來自您搜尋的 YouTube 影片連結。

        您的回覆格式必須是：
        ## 🌤️ 天氣資訊
        ## 💱 貨幣換算
        ## 🏛️ 景點與活動
        ## 🏨 住宿與飯店
        ## 📅 每日行程
        ## 💰 費用明細
        ## 🎥 YouTube 資源
        ## 📋 總結
        """)
        
        # 創建工具列表
        tools = [addition, multiply, division, substraction, get_weather, 
                 search_google, search_duck, repl_tool, youtube_search]
        
        # 將工具綁定到 LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # 創建圖函數
        def function_1(state: MessagesState):
            user_question = state["messages"]
            input_question = [system_prompt] + user_question
            response = llm_with_tools.invoke(input_question)
            return {"messages": [response]}
        
        # 構建圖 (Graph)
        builder = StateGraph(MessagesState)
        builder.add_node("llm_decision_step", function_1)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "llm_decision_step")
        builder.add_conditional_edges("llm_decision_step", tools_condition)
        builder.add_edge("tools", "llm_decision_step")
        
        # 編譯圖
        react_graph = builder.compile()
        return react_graph
        
    except Exception as e:
        st.error(f"❌ 初始化旅遊規劃師時發生錯誤: {str(e)}")
        st.info("💡 請檢查您的 API 密鑰和網路連線")
        return None

def main():
    # 標頭 (Header) - **已中文化**
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>🌍 AI 旅遊規劃師與費用預算助手</h1>
        <p>利用即時數據和詳細費用計算，規劃您的完美旅程</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API 狀態檢查 (Sidebar) - **已中文化**
    st.sidebar.header("📡 API 狀態")
    
    # 檢查 API 密鑰
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    serper_key = st.secrets.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
    weather_key = st.secrets.get("OPENWEATHERMAP_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
    
    if openai_key:
        st.sidebar.success("✅ OpenAI API")
    else:
        st.sidebar.error("❌ OpenAI API 密鑰缺失")
        st.sidebar.info("應用程式運作必需")
    
    if serper_key:
        st.sidebar.success("✅ Serper API")
    else:
        st.sidebar.warning("⚠️ Serper API 密鑰缺失")
        st.sidebar.info("將使用 DuckDuckGo 作為備選")
        
    if weather_key:
        st.sidebar.success("✅ 天氣 API")
    else:
        st.sidebar.warning("⚠️ 天氣 API 密鑰缺失")
        st.sidebar.info("天氣功能將無法運作")
    
    # 主要內容 - **已中文化**
    st.header("💬 旅遊查詢")
    
    # 範例查詢 - **已中文化**
    example_queries = {
        "🏖️ 海灘度假": """我計畫在 12 月去印度果阿 (Goa) 玩 5 天。
我的預算是 30,000 印度盧比 (INR)。
請查詢果阿的當前天氣。
尋找每晚低於 3,000 印度盧比的飯店。
我想知道海灘、水上活動和夜生活。
請計算精確的總費用，包括食物 (每天 500 印度盧比)。
請提供關於果阿的旅遊影片。""",
        
        "🌍 國際旅行": """我想去泰國玩 4 天。
我的預算是 800 美元 (USD)。
請將所有費用換算成新台幣 (TWD)。
查詢曼谷的當前天氣。
尋找每晚低於 30 美元的平價飯店。
包含街頭小吃和餐廳的費用。
提供寺廟門票和交通費用。
計算以美元 (USD) 和新台幣 (TWD) 計價的總旅行費用。"""
    }
    
    selected_example = st.selectbox("🎯 選擇範例查詢:", 
                                    ["自訂查詢"] + list(example_queries.keys()))
    
    if selected_example != "自訂查詢":
        query = st.text_area("✍️ 您的旅遊查詢:", 
                             value=example_queries[selected_example],
                             height=200)
    else:
        query = st.text_area("✍️ 您的旅遊查詢:", 
                             placeholder="例如：我想在 5 月去巴黎玩 7 天...",
                             height=200)
    
    # 處理按鈕 - **已中文化**
    if st.button("🚀 規劃我的行程", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("請輸入您的旅遊查詢！")
            return
        
        if not openai_key:
            st.error("❌ 需要 OpenAI API 密鑰。請將其加入 Streamlit secrets。")
            return
        
        # 初始化旅遊規劃師
        if st.session_state.travel_agent is None:
            with st.spinner("🔧 正在初始化 AI 旅遊規劃師..."):
                st.session_state.travel_agent = initialize_travel_agent()
        
        if st.session_state.travel_agent is None:
            st.error("❌ 初始化旅遊規劃師失敗。請檢查您的 API 密鑰。")
            return
        
        # 處理查詢
        with st.spinner("🤖 正在規劃您的完美旅程..."):
            try:
                response = st.session_state.travel_agent.invoke({
                    "messages": [HumanMessage(query)]
                })
                
                # 顯示回覆
                if response and "messages" in response:
                    final_response = response["messages"][-1].content
                    st.success("✅ 您的旅遊計畫已準備就緒！")
                    st.markdown(final_response)
                    
                    # 加入聊天紀錄 (如果需要顯示歷史紀錄，您可以取消註釋此部分)
                    # st.session_state.chat_history.append({
                    #     "query": query,
                    #     "response": final_response
                    # })
                else:
                    st.error("❌ 未收到回覆。請再試一次。")
                    
            except Exception as e:
                st.error(f"❌ 處理您的請求時發生錯誤: {str(e)}")
                st.info("💡 嘗試重新整理頁面或檢查您的網路連線")

if __name__ == "__main__":
    main()
