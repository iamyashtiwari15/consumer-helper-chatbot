# import os
# from PIL import Image as PILImage
# from dotenv import load_dotenv
# from langchain.agents import Tool
# from langchain_core.documents import Document
# from langchain_core.tools import tool
# from langchain_core.output_parsers import StrOutputParser
# from langchain_tavily import TavilySearch
# # from langchain_tavily import TavilySearchResults
# # from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate

# load_dotenv()

# # ✅ Vision model from Groq (e.g., LLaVA or LLaMA with vision support)
# llm = ChatGroq(
#     model="llama3-8b-8192",  # Replace with mixtral-8x7b-32768 if needed
#     temperature=0.3,
# )

# # ✅ Tavily tool
# search_tool = TavilySearch()
# # search_tool = TavilySearchResults()
# # search_tool = Tool(
# #     name="tavily_search",
# #     func=TavilySearch(max_results=5).invoke,
# #     description="Search the web for medical or scientific information"
# # )

# # ✅ Prompt
# IMAGE_ANALYSIS_QUERY = """
# You are a top-tier medical imaging expert. Carefully analyze the uploaded medical image and provide a structured markdown report:

# ### 1. Image Type & Region
# - What imaging modality is used?
# - What part of the body is shown?
# - Is the image of good quality?

# ### 2. Key Findings
# - Describe any abnormalities.
# - Include precise details (sizes, density, color differences).

# ### 3. Diagnosis
# - Most likely condition(s).
# - Alternate possibilities with reasoning.

# ### 4. Layman's Terms
# - Explain the above in simple language.

# ### 5. Recent Research
# - Use Tavily to search for latest research/treatments on this condition.
# - Provide 2-3 bullet points from the findings.

# Respond in clean markdown format.
# """

# # ✅ Analyzer function
# def analyze_image(filepath: str) -> str:
#     try:
#         # Open and validate
#         image = PILImage.open(filepath)
#         if image.format not in ["JPEG", "PNG", "BMP", "GIF"]:
#             return "❌ Unsupported image format. Please upload JPG, PNG, BMP, or GIF."
#         if os.path.getsize(filepath) > 10 * 1024 * 1024:
#             return "❌ Image too large. Please upload an image smaller than 10MB."
#         if image.mode in ("RGBA", "P"):
#             image = image.convert("RGB")

#         # Resize
#         width, height = image.size
#         new_width = 500
#         new_height = int((new_width / width) * height)
#         resized = image.resize((new_width, new_height))
#         temp_path = "temp_resized_image.jpg"
#         resized.save(temp_path)

#         # Prepare image data
#         with open(temp_path, "rb") as f:
#             image_bytes = f.read()

#         # LangChain vision models (if available)
#         prompt = ChatPromptTemplate.from_messages([
#             ("human", IMAGE_ANALYSIS_QUERY)
#         ])

#         chain = prompt | llm | StrOutputParser()
#         result = chain.invoke({"input": IMAGE_ANALYSIS_QUERY, "images": [image_bytes]})

#         # Add Tavily findings
#         docs = search_tool.invoke("latest research on " + result[:100])
#         if docs:
#             result += "\n\n---\n\n### 🔍 Additional Research from Tavily:\n"
#             for doc in docs[:3]:
#                 result += f"- [{doc['title']}]({doc['url']})\n"

#         os.remove(temp_path)
#         return result

#     except PILImage.UnidentifiedImageError:
#         return "❌ The uploaded file is not a valid image."
#     except Exception as e:
#         return f"⚠️ Error analyzing image: {str(e)}"


import os
from PIL import Image as PILImage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# ✅ Vision model from Groq
llm = ChatGroq(
    model="llama3-8b-8192",  # or mixtral-8x7b-32768
    temperature=0.3,
)

# ✅ Tavily tool
search_tool = TavilySearch()

# ✅ Prompt
IMAGE_ANALYSIS_QUERY = """
You are a top-tier medical imaging expert. Carefully analyze the uploaded medical image and provide a structured markdown report:

### 1. Image Type & Region
- What imaging modality is used?
- What part of the body is shown?
- Is the image of good quality?

### 2. Key Findings
- Describe any abnormalities.
- Include precise details (sizes, density, color differences).

### 3. Diagnosis
- Most likely condition(s).
- Alternate possibilities with reasoning.

### 4. Layman's Terms
- Explain the above in simple language.

### 5. Recent Research
- Use Tavily to search for latest research/treatments on this condition.
- Provide 2-3 bullet points from the findings.

Respond in clean markdown format.
"""

# ✅ Analyzer function
def analyze_image(filepath: str) -> str:
    try:
        # Open and validate
        image = PILImage.open(filepath)
        if image.format not in ["JPEG", "PNG", "BMP", "GIF"]:
            return "❌ Unsupported image format. Please upload JPG, PNG, BMP, or GIF."
        if os.path.getsize(filepath) > 10 * 1024 * 1024:
            return "❌ Image too large. Please upload an image smaller than 10MB."
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # Resize
        width, height = image.size
        new_width = 500
        new_height = int((new_width / width) * height)
        resized = image.resize((new_width, new_height))
        temp_path = "temp_resized_image.jpg"
        resized.save(temp_path)

        # Prepare image data
        with open(temp_path, "rb") as f:
            image_bytes = f.read()

        # LangChain vision models (if available)
        prompt = ChatPromptTemplate.from_messages([("human", IMAGE_ANALYSIS_QUERY)])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"input": IMAGE_ANALYSIS_QUERY, "images": [image_bytes]})

        # ✅ Tavily search (handling proper dict response)
        query = "latest research on " + result[:100]
        search_result = search_tool.invoke(query)

        if search_result and isinstance(search_result, dict) and "results" in search_result:
            result += "\n\n---\n\n### 🔍 Additional Research from Tavily:\n"
            for doc in search_result["results"][:3]:
                result += f"- [{doc.get('title', 'No Title')}]({doc.get('url', '#')})\n"

        os.remove(temp_path)
        return result

    except PILImage.UnidentifiedImageError:
        return "❌ The uploaded file is not a valid image."
    except Exception as e:
        return f"⚠️ Error analyzing image: {str(e)}"
