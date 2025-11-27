import torch
import shutil
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def check_hardware():
    """Test Nvidia GPU and CUDA availability"""
    print("\n--- 1. Hardware & CUDA Check ---")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ CUDA Not Found (Running on CPU)")

def check_poppler():
    """Test if Poppler is in the System PATH for PDF processing"""
    print("\n--- 2. PDF Tools (Poppler) Check ---")
    path = shutil.which("pdftoppm")
    if path:
        print(f"✅ Poppler Found: {path}")
    else:
        print("❌ Poppler Not Found (Add 'bin' to PATH)")

def check_groq():
    """Test connection to LLM Inference API"""
    print("\n--- 3. Groq API Connection Check ---")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY missing in .env")
        return
    
    try:
        llm = ChatGroq(api_key=key, model_name="llama-3.3-70b-versatile")
        llm.invoke("Ping")
        print("✅ Groq API Connected")
    except Exception as e:
        print(f"❌ Groq Connection Failed: {e}")

if __name__ == "__main__":
    print("🏥 STARTING SYSTEM HEALTH CHECK...")
    check_hardware()
    check_poppler()
    check_groq()
    print("\n--- End of Check ---")