from huggingface_hub import snapshot_download

# Define the local directory with a short path to store the model
local_dir = r"D:\Users\ziad.mahmoud\last_rag\genai_chatbot\genai_chatbot\models\stella_en_400M_v5"

# Download the model
snapshot_download(repo_id="dunzhang/stella_en_400M_v5", local_dir=local_dir)

print(f"Model downloaded to: {local_dir}")
