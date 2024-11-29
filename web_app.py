import streamlit as st
import pandas as pd
import requests
import time
import re
import json
import plotly.graph_objs as go
import plotly.express as px
from huggingface_hub import InferenceClient
import random

def extract_sentiment(text):
    """
    Extract sentiment from the text, case-insensitive
    """
    sentiments = ["stronglyagree","stronglydisagree" , "disagree", "agree"]
    text = text.replace(" ", "").lower()
    for sentiment in sentiments:
        if sentiment in text :
            return sentiment.lower()
    return "not found"

def parse_results():
    """
    Parse results and categorize sentiments
    """
    if "results" not in st.session_state:
        st.warning("No results found. Please run the LLM queries first.")
        return None

    # Initialize dictionaries to store results
    sentiment_counts = {
        "strongly agree": 0,
        "agree": 0,
        "disagree": 0,
        "strongly disagree": 0,
        "not found": 0
    }
    
    # Storing detailed results for later reference
    detailed_results = {
        "strongly agree": [],
        "agree": [],
        "disagree": [],
        "strongly disagree": [],
        "not found": []
    }

    # Parse results
    for idx,response in st.session_state.results:
        try:
            # Extract text from the result
            text = response if isinstance(response, str) else str(response)
            
            sentiment = extract_sentiment(text)
            
            if sentiment == "stronglyagree":
                sentiment = "strongly agree"
            elif sentiment == "stronglydisagree":
                sentiment = "strongly disagree"

            sentiment_counts[sentiment] += 1
            detailed_results[sentiment].append(text)
        except Exception as e:
            st.warning(f"Error processing result: {e}")

    return sentiment_counts, detailed_results

def plot_sentiment_distribution():
    """
    Create bar plot of sentiment distribution
    """
    results = parse_results()
    if not results:
        return None

    sentiment_counts, detailed_results = results

    # Create interactive bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=list(sentiment_counts.keys()), 
            y=list(sentiment_counts.values()),
            text=list(sentiment_counts.values()),
            textposition='auto',
            hovertemplate='%{x}: %{y}<extra></extra>'
        )
    ])
    fig.update_layout(
        title='Sentiment Distribution Across Prompts',
        xaxis_title='Sentiment',
        yaxis_title='Count'
    )

    # Add click event to show details
    return fig, detailed_results

def plot_llm_results():
    """
    Visualize LLM results with multiple charts
    """
    if "results" not in st.session_state:
        st.warning("No LLM results found.")
        return None

    # If it's the first time running, initialize llm_results
    if "llm_results" not in st.session_state:
        st.session_state.llm_results = {}

    # Add current results to llm_results if model_name exists
    if hasattr(st.session_state, 'model_name'):
        st.session_state.llm_results[st.session_state.model_name] = st.session_state.results

    # Ensure we have results
    if not st.session_state.llm_results:
        st.warning("No LLM results found.")
        return None

    # Count responses per LLM
    llm_response_counts = {}
    llm_sentiment_counts = {}

    for model, results in st.session_state.llm_results.items():
        # Count total responses per LLM
        llm_response_counts[model] = len(results)

        # Count sentiments per LLM
        sentiment_counts = {
            "strongly agree": 0,
            "agree": 0,
            "disagree": 0,
            "strongly disagree": 0,
            "not found": 0
        }
        
        for idx,response in results:
            try:
                # Extract text from the result
                text = response if isinstance(response, str) else str(response)
                
                sentiment = extract_sentiment(text)

                if sentiment == "stronglyagree":
                    sentiment = "strongly agree"
                elif sentiment == "stronglydisagree":
                    sentiment = "strongly disagree"
                
                sentiment_counts[sentiment] += 1
            except Exception as e:
                st.warning(f"Error processing result for {model}: {e}")
        
        llm_sentiment_counts[model] = sentiment_counts

    # Create response count bar plot
    response_fig = go.Figure(data=[
        go.Bar(
            x=list(llm_response_counts.keys()), 
            y=list(llm_response_counts.values()),
            text=list(llm_response_counts.values()),
            textposition='auto'
        )
    ])
    response_fig.update_layout(
        title='Number of Responses per LLM',
        xaxis_title='LLM Model',
        yaxis_title='Response Count'
    )

    # Create stacked bar plot for sentiments
    sentiments = ["strongly agree", "agree", "disagree", "strongly disagree", "not found"]
    
    # Prepare data for stacked bar plot
    stacked_data = []
    for model in llm_sentiment_counts.keys():
        stacked_data.append([llm_sentiment_counts[model][sent] for sent in sentiments])

    stacked_fig = go.Figure(data=[
        go.Bar(
            name=sent, 
            x=list(llm_sentiment_counts.keys()), 
            y=[row[i] for row in stacked_data]
        ) for i, sent in enumerate(sentiments)
    ])
    stacked_fig.update_layout(
        title='Sentiment Distribution per LLM',
        xaxis_title='LLM Model',
        yaxis_title='Sentiment Count',
        barmode='stack'
    )

    return response_fig, stacked_fig


def update_comprehensive_results():
    """
    Create a comprehensive results structure tracking sentiments across different LLMs
    """
    # Check if required session states exist
    if not hasattr(st.session_state, 'list_variable') or \
       not hasattr(st.session_state, 'results') or \
       not hasattr(st.session_state, 'model_name'):
        st.warning("Missing required session state variables.")
        return None

    # Initialize the comprehensive results list if not exists
    if 'comprehensive_results' not in st.session_state:
        st.session_state.comprehensive_results = []

    # Iterate through list variable items
    for idx, item in enumerate(st.session_state.list_variable):
        # Check if this item already has an entry
        response = st.session_state.results[idx][1]
        response_str = str(response) if not isinstance(response, str) else response
        item_exists = False
        for existing_dict in st.session_state.comprehensive_results:
            if item in existing_dict:
                item_exists = True
                # Update or add model results for this item
                existing_dict[item][st.session_state.model_name] = extract_sentiment(response_str)
                break

        # If item doesn't exist, create a new dictionary entry
        if not item_exists:
            new_item_dict = {
                item: {
                    st.session_state.model_name: extract_sentiment(response_str)
                }
            }
            st.session_state.comprehensive_results.append(new_item_dict)

    return st.session_state.comprehensive_results


def calculate_disagreement_degree(comprehensive_results):
    """
    Calculate degree of disagreement across models for each prompt
    
    Disagreement Calculation Logic:
    1. Sentiment Hierarchy: 
       strongly disagree < disagree < not found < agree < strongly agree
    2. Assign numerical weights to sentiments
    3. Calculate variance of sentiment weights across models
    4. Consider completeness of responses
    5. Normalize disagreement score
    """
    # Sentiment weight mapping
    sentiment_weights = {
        "strongly disagree": 1,
        "disagree": 2,
        "not found": 3,
        "agree": 4,
        "strongly agree": 5
    }
    
    # Store disagreement results
    disagreement_results = []
    
    for prompt_dict in comprehensive_results:
        for prompt, model_sentiments in prompt_dict.items():
            # Extract sentiment weights for this prompt
            sentiment_weights_list = []
            model_count = 0
            
            for model, sentiment in model_sentiments.items():
                if sentiment in sentiment_weights:
                    sentiment_weights_list.append(sentiment_weights[sentiment])
                    model_count += 1
            
            # Skip if insufficient data
            if model_count < 2:
                continue
            
            # Calculate disagreement metrics
            # 1. Variance of sentiment weights
            import numpy as np
            weight_variance = np.var(sentiment_weights_list)
            
            # 2. Range of sentiments (max - min)
            weight_range = max(sentiment_weights_list) - min(sentiment_weights_list)
            
            # 3. Unique sentiment count
            unique_sentiments = len(set(model_sentiments.values()))
            
            # Combine metrics into a disagreement score
            # Higher score means more disagreement
            disagreement_score = (
                weight_variance * 0.4 +  # Variance of weights
                weight_range * 0.3 +     # Range of weights
                (unique_sentiments / model_count) * 0.3  # Diversity of sentiments
            )
            
            # Normalize to 0-100 scale
            normalized_score = min(max(disagreement_score * 20, 0), 100)
            
            disagreement_results.append({
                "prompt": prompt,
                "disagreement_score": normalized_score,
                "model_sentiments": model_sentiments,
                "unique_sentiment_count": unique_sentiments,
                "model_count": model_count
            })
    
    # Sort by disagreement score in descending order
    disagreement_results.sort(key=lambda x: x['disagreement_score'], reverse=True)
    
    # Return top 5 most disagreed prompts
    return disagreement_results[:5]

# Example usage in Streamlit
def display_disagreement_analysis(comprehensive_results):
    """
    Wrapper function to display disagreement analysis in Streamlit
    """

    
    # Calculate disagreement
    top_disagreements = calculate_disagreement_degree(comprehensive_results)
    
    # Display results
    for idx, result in enumerate(top_disagreements, 1):
        st.subheader(f"{idx}. Prompt with High Disagreement")
        st.write(f"Prompt: {result['prompt']}")
        st.write(f"Disagreement Score: {result['disagreement_score']:.2f}")
        
        # Create a table of model sentiments
        disagreement_df = pd.DataFrame.from_dict(result['model_sentiments'], orient='index', columns=['Sentiment'])
        disagreement_df.index.name = 'Model'
        st.table(disagreement_df)
        
        st.write(f"Unique Sentiments: {result['unique_sentiment_count']}")
        st.write(f"Models Analyzed: {result['model_count']}")
        st.markdown("---")

    return top_disagreements

#-----------------------------------------------------------------------------------------------------------------------------------------#

# -----------------------
# Part 1: Title
# -----------------------
st.title("DHAI Political bias in LLM")

# -----------------------
# Part 2: Select Data Loading Method
# -----------------------
st.header("1. Load Data")
load_method = st.radio("Choose a method to load data:", ("Load from Google Drive", "Upload a file locally"))

data = None
generated_prompts = []

if load_method == "Load from Google Drive":
    drive_url = st.text_input("Enter Google Drive file link:")
    if st.button("Load Data"):
        try:
            # Extract file ID using regex
            match = re.search(r"(?<=/d/|id=)[^/?&]+", drive_url)
            if not match:
                raise ValueError("Invalid Google Drive URL format.")
            file_id = match.group()
            download_url = f"https://drive.google.com/uc?id={file_id}"
            data = pd.read_csv(download_url)
            st.success("Data loaded successfully!")
            st.write("Preview of the data:", data.head())
        except Exception as e:
            st.error(f"Failed to load data: {e}")

elif load_method == "Upload a file locally":
    uploaded_file = st.file_uploader("Upload a CSV file:", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            st.write("Preview of the data:", data.head())
        except Exception as e:
            st.error(f"Failed to load data: {e}")

if data is not None:
    column_options = list(data.columns)
    selected_column = st.selectbox("Select a column to save as a list variable:", column_options)
    list_variable = data[selected_column].dropna().tolist()
    if len(list_variable) > 50:
        list_variable = random.sample(list_variable, 50)
    st.session_state.list_variable = list_variable

# -----------------------
# Part 3: Prompt Generation
# -----------------------
st.header("2. Prompt Generation")

if "list_variable" not in st.session_state:
    st.session_state.list_variable = ["resolution_1", "resolution_2", "resolution_3"]

if st.session_state.list_variable:
    st.write("Available items in the list:")
    st.write(st.session_state.list_variable)

user_input = st.text_input("Enter text to concatenate with the list items:", value="")
question = st.text_input("Enter the question", value="")
if st.button("Generate Prompts"):
    st.session_state.generated_prompts = [
        {
            "system": "You are a political analyst with expertise in evaluating resolutions.",
            "user": f"{user_input} Question: '{question}' {item}"
        }
        for item in st.session_state.list_variable
    ]
    st.success("Prompts have been successfully generated!")
    st.write("Generated Prompts:")
    st.write(st.session_state.generated_prompts)

# -----------------------
# Part 4: Hugging Face API Integration
# -----------------------
st.header("3. Hugging Face API Integration")
if 'results' not in st.session_state:
    st.session_state.results = []

if "generated_prompts" in st.session_state:
    generated_prompts = st.session_state.generated_prompts

    st.subheader("API Configuration")
    api_key = st.text_input("Enter your HuggingFace API key:", type="password")
    predefined_models = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "tiiuae/falcon-7b-instruct",
    ]
    model_name = st.selectbox("Select a Hugging Face model:", options=predefined_models + ["Custom"])
    if model_name == "Custom":
        model_name = st.text_input("Enter custom model name:")

    send_interval = st.number_input("Interval between requests (seconds):", 0.0, 10.0, 0.5, 0.1)
    max_tokens = st.number_input("Maximum tokens to return:", 1, 1024, 100)
    temperature = st.slider("Temperature (Randomness):", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Top-p (Nucleus Sampling):", 0.0, 1.0, 0.9, 0.1)

    if api_key and model_name and st.button("Send Prompts to Model"):
        results = []
        answers = []
        try:
            client = InferenceClient(model=model_name,api_key=api_key)
            

            
            #huggingface_api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            #headers = {
                #"Authorization": f"Bearer {api_key}",
                #"Content-Type": "application/json"
            #}
            
            for idx, prompt in enumerate(generated_prompts):
                st.write(f"Processing prompt {idx + 1}/{len(generated_prompts)}")

                try:
                  stream = client.chat_completion(
                                    messages=[
                                    {"role": "user",
    		                        "content": prompt["user"] #you can choose from the prompt templates list
                                     }
                                     ],    	
    	                             max_tokens=max_tokens,
                                     temperature=temperature,
                                     top_p=top_p,
    	                             stream=True
                                     )
                  chat=[]
                  for chunk in stream:
                      chat.append(chunk.choices[0].delta.content)
                  response_text = ' '.join(chat)
                  temp=idx, response_text
                  answers.append(temp)
                except Exception as stream_err:
                # 如果流式调用失败，记录错误
                  st.error(f"Stream processing error for Prompt {idx + 1}: {stream_err}")
                  answers.append((idx, {"error": str(stream_err)}))
                    
                time.sleep(send_interval)
            st.success("All prompts processed!")
            st.write("Final Results:", answers)

            # 保存会话状态
            st.session_state.results = answers
            st.session_state.model_name = model_name
            st.session_state.generated_prompts = generated_prompts
        except Exception as e:
        # 捕获顶层异常
            st.error(f"Unexpected error: {e}")

                
                # Modify payload based on model type
                #if 'Instruct' in model_name or 'zephyr' in model_name:
                    # For instruction-tuned models
                    #payload = {
                        #"inputs": prompt["user"],
                        #“parameters": {
                            #"max_new_tokens": max_tokens,
                            #"temperature": temperature,
                            #"top_p": top_p
                        #}
                    #}
                #else:
                    # For other models (like GPT-2)
                    #payload = {
                        #"inputs": prompt["user"],
                        #"parameters": {
                            #"max_length": max_tokens,
                            #"temperature": temperature,
                            #"top_p": top_p
                        #}
                    #}
                
                #try:
                    #response = requests.post(
                        #huggingface_api_url, 
                        #headers=headers, 
                        #data=json.dumps(payload)
                    #)
                    
                    #if response.status_code == 200:
                        #try:
                            #result = response.json()
                            #results.append(result)
                            #st.success(f"Response for Prompt {idx + 1}:")
                            #st.write(result)
                        #except ValueError as json_err:
                            #st.error(f"JSON Decoding Error for Prompt {idx + 1}: {json_err}")
                            #st.error(f"Response content: {response.text}")
                    #else:
                        #error_message = response.text
                        #st.error(f"API request failed for Prompt {idx + 1}: {error_message}")
                        #results.append({"error": error_message})
                
                #except requests.exceptions.RequestException as req_err:
                    #st.error(f"Request failed for Prompt {idx + 1}: {req_err}")
            
            #st.success("All prompts processed!")
            #st.write("Final Results:", results)

            #st.session_state.results = results
            #st.session_state.model_name = model_name
            #st.session_state.generated_prompts = generated_prompts
        
        #except Exception as e:
            #st.error(f"Unexpected error: {e}")

# -----------------------
# Part 5: Statistical Charts
# -----------------------
st.header("4. Statistical Charts")

# Sentiment Distribution Visualization
st.subheader("Sentiment Distribution")
sentiment_plot = plot_sentiment_distribution()
if sentiment_plot:
    fig, detailed_results = sentiment_plot
    st.plotly_chart(fig, use_container_width=True)

    # Sidebar for detailed results
    st.sidebar.header("Sentiment Details")
    selected_sentiment = st.sidebar.selectbox(
        "Select Sentiment", 
        list(detailed_results.keys())
    )
    st.sidebar.write(f"Resolution/Decision for {selected_sentiment}:")
    
    idx_list = []
    results_list = []
    
    for item in st.session_state.results:
        results_list.append(item[1])
    
    for result in detailed_results[selected_sentiment] :
       for idx, response_text in enumerate(results_list):
          if str(result).strip() == str(response_text).strip() :
              idx_list.append(idx)
    idx_list = list(set(idx_list))
    for idx in idx_list:
      st.sidebar.write(idx, st.session_state.list_variable[idx])

# LLM Performance Visualization
st.subheader("LLM Performance")
llm_plots = plot_llm_results()
if llm_plots:
    response_fig, stacked_fig = llm_plots
    
    # Tabs for different visualizations
    tab1, tab2 = st.tabs(["Responses per LLM", "Sentiment Distribution per LLM"])
    
    with tab1:
        st.plotly_chart(response_fig, use_container_width=True)
    
    with tab2:
        st.plotly_chart(stacked_fig, use_container_width=True)

# -----------------------
# Part 6: Digree of disagreement
# -----------------------

st.header("Disagreement Analysis")
update_comprehensive_results = update_comprehensive_results()
filtered_data = calculate_disagreement_degree(update_comprehensive_results)
display_disagreement_analysis(filtered_data)

st.header("Answers of LLM")
st.write(st.session_state.results)
