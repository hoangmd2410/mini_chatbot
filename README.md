## Installation
### Ollama for local run
- Download and install ollama from [original website](https://ollama.com/download/linux)
- Run ```ollama run gemma2``` to serve model gemma2 as local chat bot

### Environment install
- Install new environment using Anaconda ```conda create -n chatbot python=3.11```
- Activate and install packages ```conda activate chatbot && pip install -r requirements.txt```
- Run the program ```python main.py```
- Open link ```http://127.0.0.1:7860``` Send the OpenAI api key and start asking questions

### Approach 
- Build a RAG system ingesting files in 'data' folder and scraping urls to a vector database. Text in files and urls are split into chunks, encode to a vectors by embedding model. Vectors are used to search with cosine similarity to get top relevant chunks to user's query
- When users submit a query, top relevant chunks are retrieved and they are sent to the LLM as additional information so the LLM can give the best answer 
- With questions about particular file such as summarizing, the program will find if the file exists in the database and only get chunk from the text. Otherwise, the answer will indicate not file name


Example: 
- "summarize cast in anchor file" -> find exact path /Users/hoangmd/EH/mycode/chatbot/data/Cast-in_Anchor.pdf -> The document "Cast-in_Anchor.pdf" provides guidelines for the design, installation, and maintenance of cast-in anchors on the external walls of new buildings. It highlights the importance of safety features such as gondola systems, service platforms, and anchor devices to ensure worker safety during repair, maintenance, and alterations. The guidelines include technical design aspects like the positioning of cast-in anchors on reinforced concrete walls with specific thickness and material requirements. It also details operational and maintenance requirements, including inspection protocols, the proper use of fall protection equipment, and secure fencing of platforms. The document emphasizes compliance with relevant standards and regulations to ensure safety and reliability.
- "summarize the file instruction about sport" -> no file -> I couldn't find the file related to "instruction about sport." Could you please provide more details or check the file name?