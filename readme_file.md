# 🌍 AI Travel Agent & Expense Planner

An intelligent travel planning agent built with **LangChain**, **LangGraph**, and **OpenAI** that provides comprehensive trip planning with real-time data and detailed cost calculations.

## 🚀 Features

- **🌤️ Real-time Weather Information** - Current weather conditions for any destination
- **🏛️ Top Attractions & Activities** - Discover must-visit places and experiences  
- **🏨 Hotel Cost Calculation** - Find budget accommodations with price calculations (daily rate × total days)
- **💱 Currency Conversion** - Convert costs to your native currency using real-time exchange rates
- **📅 Complete Itinerary Generation** - Day-by-day travel plans with actual costs
- **💰 Total Expense Calculation** - Detailed cost breakdown including hotels, food, attractions, and transport
- **🎥 Travel Videos** - Relevant YouTube videos for your destination
- **📋 Trip Summary** - Comprehensive travel plan with actionable recommendations

## 🛠️ Tech Stack

- **LangChain** - Framework for building AI applications
- **LangGraph** - State management and workflow orchestration
- **OpenAI GPT-4** - Large Language Model for intelligent responses
- **Google Serper API** - Web search for real-time information
- **OpenWeatherMap API** - Weather data
- **DuckDuckGo Search** - Alternative search engine
- **YouTube Search** - Video recommendations

## 📋 Prerequisites

Before running this project, you'll need API keys for:

1. **OpenAI API** - [Get your API key](https://platform.openai.com/api-keys)
2. **Google Serper API** - [Get your API key](https://serper.dev/)
3. **OpenWeatherMap API** - [Get your API key](https://openweathermap.org/api)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-travel-agent.git
cd ai-travel-agent
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create a `.env` file** in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
```

## 🚀 Usage

1. **Start Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open `Assignment.ipynb`**

3. **Run all cells** to initialize the travel agent

4. **Test with a travel query:**
```python
response = react_graph.invoke({
    "messages": [HumanMessage("""
    I want to visit Thailand for 5 days.
    My budget is 800 USD.
    Please convert all costs to Indian Rupees.
    Get current weather for Bangkok.
    Find budget hotels and calculate total costs.
    Show me travel videos and create an itinerary.
    """)]
})

for m in response["messages"]:
    m.pretty_print()
```

## 💡 Example Queries

### Query 1: Detailed Trip Planning
```
I want to visit Goa for 4 days in December. 
My budget is 25,000 INR total. 
I need current weather information.
Find hotels under 3000 INR per night.
I want to know about beaches, water sports, and nightlife.
Calculate the exact costs including food (500 INR per day).
Show me travel videos about Goa.
Give me a day-by-day itinerary with real prices.
```

### Query 2: Currency Conversion Focus
```
Plan a 3-day trip to Bangkok, Thailand.
My budget is 500 USD.
Convert all costs to Indian Rupees.
Get current weather for Bangkok.
Find budget hotels, street food costs, and temple entry fees.
Include YouTube videos about Bangkok travel.
```

## 🔧 Tools Available

The agent uses these specialized tools:

- **`get_weather(city)`** - Real-time weather data
- **`search_google(query)`** - Web search for hotels, restaurants, attractions
- **`search_duck(query)`** - Alternative web search
- **`youtube_search(query)`** - Find relevant travel videos
- **`addition/multiply/division/substraction`** - Cost calculations
- **`python_repl`** - Complex mathematical operations

## 📁 Project Structure

```
ai-travel-agent/
├── Assignment.ipynb          # Main Jupyter notebook
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
└── .gitignore               # Git ignore file
```

## 🎯 How It Works

1. **Weather Check** - Gets current weather for destination
2. **Search Phase** - Finds hotels, restaurants, attractions with real prices
3. **Cost Calculation** - Uses mathematical tools to calculate total expenses
4. **Currency Conversion** - Searches for current exchange rates
5. **Video Resources** - Finds relevant YouTube travel content
6. **Itinerary Generation** - Creates day-by-day plans with costs

## 🚫 Limitations

- Requires internet connection for real-time data
- API rate limits may apply
- Costs are estimates based on search results
- Currency rates are approximate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues:

**API Key Errors:**
- Ensure all API keys are correctly set in `.env` file
- Check API key validity and quotas

**Tool Errors:**
- Verify internet connection
- Check if search results are being returned properly

**Import Errors:**
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Try restarting Jupyter kernel

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/ai-travel-agent/issues) page
2. Create a new issue with detailed description
3. Include error messages and environment details

## 🙏 Acknowledgments

- **LangChain** team for the amazing framework
- **OpenAI** for the powerful language models
- **Serper** for web search capabilities
- **OpenWeatherMap** for weather data

---

**Made with ❤️ for travelers worldwide** 🌍✈️