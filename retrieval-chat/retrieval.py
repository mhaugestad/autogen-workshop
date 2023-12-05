import autogen
from autogen import config_list_from_json
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from qdrant_client import QdrantClient

# Documentation: https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/qdrant_retrieve_user_proxy_agent
# Official example: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_qdrant_RetrieveChat.ipynb
# Supported embedding models from Qdrant: https://qdrant.github.io/fastembed/examples/Supported_Models/

config_list = config_list_from_json(env_or_file = "../OAI_CONFIG_LIST")

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

svqtech_docs = [
            "https://github.com/svqtech/web/blob/master/content/comunidades/Agile%20Sur.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/AiSaturdaysSevilla.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/BitnamiSevilla.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/CloudNativeSevilla.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/DatabeersSVQ.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/Ethereum_svq.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/SVQ%20JUG.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/Sevilla%20Maker%20Society.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/Sevilla%20R.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/Sevilla%20Ruby%20On%20Rails.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/SevillaDotNet.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/SevillaSalesforceDeveloper.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/SevillaTypeScript.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/WordPress%20Sevilla.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/svq_mug.md",
            "https://github.com/svqtech/web/blob/master/content/comunidades/ttnsevilla.md"
            ]



ragproxyagent = QdrantRetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "qa",
        "docs_path": svqtech_docs,
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "client": QdrantClient(":memory:"),
        "embedding_model": "intfloat/multilingual-e5-large",
    },
)

# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

qa_problem = "a que grupo debo reunirme si me gusta la estadistica?"

if __name__ == '__main__':
    ragproxyagent.initiate_chat(assistant, problem=qa_problem)