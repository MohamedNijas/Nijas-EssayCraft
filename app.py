import os
#from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain,SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

api_key = os.environ.get('OPENAI_API_KEY')
#os.environ["OPENAI_API_KEY"] = apikey
#memory

title_memory = ConversationBufferMemory(input_key = "topic",memory_key = "chat_history" )
script_memory = ConversationBufferMemory(input_key = "title",memory_key = "chat_history")
#APP
page_bg_img = '''
<style>
body {
background-image: url("https://dms.uom.lk/s/mcZXCLp3NASrpKt");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Nijas EssayCraft")
st.subheader("Only enter the keyword.I will give the title and Article for you")
prompt = st.text_input("Enter your keyword")

submitted = st.button("submit")

#prompt templates
title_template =  PromptTemplate(
    input_variables = ["topic"],
    template = "give me the title to write a Article for a webpage on this topic: {topic}"
)

script_template = PromptTemplate(
    input_variables = ["title","wikipedia_research"],
    template = "generate a large Article completly for a webpage in this title for engineers :{title} while leverage this wikipedia research:{wikipedia_research}"
)
llm = OpenAI(temperature = 0.9)

title_chain = LLMChain(llm = llm,prompt = title_template,verbose = True,output_key = "title",memory = title_memory )
script_chain = LLMChain(llm = llm,prompt = script_template,verbose = True,output_key = "script",memory = script_memory)
#sequential_chain = SequentialChain(chains = [title_chain,script_chain],input_variables = ["topic"],output_variables= ["title","script"],verbose = True)
#show the prompt
wiki = WikipediaAPIWrapper()
if prompt and submitted:
    #response = sequential_chain({"topic":prompt})
    title = title_chain.run(prompt )
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title,wikipedia_research = wiki_research)
    #st.write(response["title"])
    #st.write(response["script"])
    st.write(title)
    st.write(script)


    with st.expander("Title History"):
        st.info(title_memory.buffer)
    with st.expander("Script History"):
        st.info(script_memory.buffer)
    with st.expander("wiki research"):
        st.info(wiki_research)
