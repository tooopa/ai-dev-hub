// Konfiguracja kategorii z ikonami SVG
const categoryConfig = {
  API: {
    label: "API",
    icon: `
      <svg viewBox="0 0 24 24">
        <rect x="3" y="5" width="18" height="6" rx="2"></rect>
        <rect x="3" y="13" width="8" height="6" rx="2"></rect>
        <rect x="13" y="13" width="8" height="6" rx="2"></rect>
      </svg>
    `
  },
  Console: {
    label: "Console",
    icon: `
      <svg viewBox="0 0 24 24">
        <rect x="3" y="4" width="18" height="14" rx="2"></rect>
        <path d="M7 9l3 3-3 3" stroke-width="1.5" fill="none"></path>
        <path d="M13 15h4" stroke-width="1.5"></path>
      </svg>
    `
  },
  Chat: {
    label: "Chat",
    icon: `
      <svg viewBox="0 0 24 24">
        <path d="M4 5h16v9H8l-4 4z"></path>
        <circle cx="10" cy="9" r="1"></circle>
        <circle cx="13" cy="9" r="1"></circle>
        <circle cx="16" cy="9" r="1"></circle>
      </svg>
    `
  },
  IDE: {
    label: "IDE / Notebooks",
    icon: `
      <svg viewBox="0 0 24 24">
        <rect x="4" y="4" width="16" height="16" rx="2"></rect>
        <path d="M9 9l-3 3 3 3" stroke-width="1.5" fill="none"></path>
        <path d="M15 9l3 3-3 3" stroke-width="1.5" fill="none"></path>
      </svg>
    `
  },
  CLI: {
    label: "CLI",
    icon: `
      <svg viewBox="0 0 24 24">
        <rect x="3" y="4" width="18" height="14" rx="2"></rect>
        <path d="M7 9l3 3-3 3" stroke-width="1.5" fill="none"></path>
        <path d="M12 15h5" stroke-width="1.5"></path>
      </svg>
    `
  },
  Tools: {
    label: "Tools / Frameworks",
    icon: `
      <svg viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M12 4v3"></path>
        <path d="M12 17v3"></path>
        <path d="M4 12h3"></path>
        <path d="M17 12h3"></path>
        <path d="M6.2 6.2l2.1 2.1"></path>
        <path d="M15.7 15.7l2.1 2.1"></path>
        <path d="M17.8 6.2l-2.1 2.1"></path>
        <path d="M8.3 15.7l-2.1 2.1"></path>
      </svg>
    `
  }
};

// Dane narzędzi (pełny JSON z kategoriami)
const tools = [
  // API (20)
  { id: "openai_api", name: "OpenAI API", url: "https://api.openai.com/v1/chat/completions", category: "API" },
  { id: "anthropic_api", name: "Anthropic API", url: "https://api.anthropic.com/v1/messages", category: "API" },
  { id: "google_gemini_api", name: "Google Gemini API", url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", category: "API" },
  { id: "x_grok_api", name: "X.ai Grok API", url: "https://api.x.ai/v1", category: "API" },
  { id: "hf_inference_api", name: "HuggingFace Inference API", url: "https://api-inference.huggingface.co/models/bert-base-uncased", category: "API" },
  { id: "replicate_api", name: "Replicate API", url: "https://api.replicate.com/v1/predictions", category: "API" },
  { id: "groq_api", name: "Groq API", url: "https://api.groq.com/openai/v1/chat/completions", category: "API" },
  { id: "cohere_api", name: "Cohere API", url: "https://api.cohere.ai/v1/chat", category: "API" },
  { id: "stability_api", name: "Stability AI API", url: "https://api.stability.ai/v2beta/stable-image/generate/core", category: "API" },
  { id: "elevenlabs_api", name: "ElevenLabs API", url: "https://api.elevenlabs.io/v1/text-to-speech", category: "API" },
  { id: "deepgram_api", name: "Deepgram API", url: "https://api.deepgram.com/v1/listen", category: "API" },
  { id: "assemblyai_api", name: "AssemblyAI API", url: "https://api.assemblyai.com/v2/transcript", category: "API" },
  { id: "tavily_api", name: "Tavily Search API", url: "https://api.tavily.com/search", category: "API" },
  { id: "pinecone_api", name: "Pinecone API", url: "https://controller.us-east1-gcp.pinecone.io/actions/whoami", category: "API" },
  { id: "weaviate_api", name: "Weaviate API", url: "https://localhost:8080/v1/schema", category: "API" },
  { id: "qdrant_api", name: "Qdrant API", url: "http://localhost:6333/collections", category: "API" },
  { id: "milvus_api", name: "Milvus REST API", url: "http://localhost:19530", category: "API" },
  { id: "modal_api", name: "Modal API", url: "https://api.modal.com/v1", category: "API" },
  { id: "runpod_api", name: "RunPod API", url: "https://api.runpod.io/graphql", category: "API" },
  { id: "nvidia_nim_api", name: "NVIDIA NIM API", url: "https://integrate.api.nvidia.com/v1/chat/completions", category: "API" },

  // Console (20)
  { id: "openai_console", name: "OpenAI Platform Console", url: "https://platform.openai.com", category: "Console" },
  { id: "anthropic_console", name: "Anthropic Console", url: "https://console.anthropic.com", category: "Console" },
  { id: "google_ai_studio", name: "Google AI Studio", url: "https://aistudio.google.com", category: "Console" },
  { id: "hf_spaces_console", name: "HuggingFace Spaces Console", url: "https://huggingface.co/spaces", category: "Console" },
  { id: "replicate_dashboard", name: "Replicate Dashboard", url: "https://replicate.com", category: "Console" },
  { id: "groq_console", name: "Groq Console", url: "https://console.groq.com", category: "Console" },
  { id: "nvidia_build_console", name: "NVIDIA Build Console", url: "https://build.nvidia.com", category: "Console" },
  { id: "aws_sagemaker_console", name: "AWS SageMaker Studio", url: "https://console.aws.amazon.com/sagemaker", category: "Console" },
  { id: "azure_ai_studio", name: "Azure AI Studio", url: "https://ai.azure.com", category: "Console" },
  { id: "ibm_watson_studio", name: "IBM Watson Studio", url: "https://dataplatform.cloud.ibm.com", category: "Console" },
  { id: "databricks_console", name: "Databricks MLflow Console", url: "https://community.cloud.databricks.com", category: "Console" },
  { id: "modal_console", name: "Modal Dashboard", url: "https://modal.com", category: "Console" },
  { id: "runpod_console", name: "RunPod Console", url: "https://www.runpod.io/console", category: "Console" },
  { id: "pinecone_console", name: "Pinecone Console", url: "https://app.pinecone.io", category: "Console" },
  { id: "qdrant_console", name: "Qdrant Cloud Console", url: "https://cloud.qdrant.io", category: "Console" },
  { id: "weaviate_console", name: "Weaviate Console", url: "https://console.weaviate.io", category: "Console" },
  { id: "supabase_dashboard", name: "Supabase Dashboard", url: "https://app.supabase.com", category: "Console" },
  { id: "firebase_console", name: "Firebase Console", url: "https://console.firebase.google.com", category: "Console" },
  { id: "mongodb_atlas", name: "MongoDB Atlas Console", url: "https://cloud.mongodb.com", category: "Console" },
  { id: "github_models_console", name: "GitHub Models Console", url: "https://github.com/marketplace/models", category: "Console" },

  // Chat (20)
  { id: "chatgpt", name: "ChatGPT", url: "https://chat.openai.com", category: "Chat" },
  { id: "claude_chat", name: "Claude Chat", url: "https://claude.ai", category: "Chat" },
  { id: "gemini_chat", name: "Gemini Chat", url: "https://gemini.google.com", category: "Chat" },
  { id: "grok_chat", name: "Grok (X.ai)", url: "https://x.ai", category: "Chat" },
  { id: "perplexity", name: "Perplexity AI", url: "https://www.perplexity.ai", category: "Chat" },
  { id: "meta_ai_chat", name: "Meta AI Chat", url: "https://www.meta.ai", category: "Chat" },
  { id: "mistral_chat", name: "Mistral Chat", url: "https://chat.mistral.ai", category: "Chat" },
  { id: "pi_ai", name: "Pi.ai", url: "https://pi.ai", category: "Chat" },
  { id: "huggingchat", name: "HuggingChat", url: "https://huggingface.co/chat", category: "Chat" },
  { id: "chatsonic", name: "ChatSonic", url: "https://writesonic.com/chat", category: "Chat" },
  { id: "poe", name: "Poe", url: "https://poe.com", category: "Chat" },
  { id: "character_ai", name: "Character.ai", url: "https://character.ai", category: "Chat" },
  { id: "replika", name: "Replika", url: "https://replika.ai", category: "Chat" },
  { id: "janitor_ai", name: "JanitorAI", url: "https://janitorai.com", category: "Chat" },
  { id: "autogpt_chat", name: "AutoGPT Chat Mode", url: "https://autogpt.net", category: "Chat" },
  { id: "llamachat", name: "LlamaChat", url: "https://lmstudio.ai", category: "Chat" },
  { id: "lm_studio_chat", name: "LM Studio Chat", url: "https://lmstudio.ai", category: "Chat" },
  { id: "ollama_chat", name: "Ollama Chat", url: "https://ollama.com", category: "Chat" },
  { id: "qwen_chat", name: "Qwen Chat", url: "https://chat.qwenlm.ai", category: "Chat" },
  { id: "deepseek_chat", name: "Deepseek Chat", url: "https://chat.deepseek.com", category: "Chat" },

  // IDE (20)
  { id: "vscode", name: "VS Code", url: "https://code.visualstudio.com", category: "IDE" },
  { id: "jetbrains", name: "JetBrains IDEs", url: "https://www.jetbrains.com", category: "IDE" },
  { id: "cursor_ide", name: "Cursor IDE", url: "https://cursor.sh", category: "IDE" },
  { id: "copilot_vscode", name: "GitHub Copilot (VS Code)", url: "https://github.com/features/copilot", category: "IDE" },
  { id: "copilot_jetbrains", name: "GitHub Copilot (JetBrains)", url: "https://docs.github.com/en/copilot", category: "IDE" },
  { id: "copilot_workspace", name: "Copilot Workspace", url: "https://github.com/features/copilot-workspace", category: "IDE" },
  { id: "codewhisperer", name: "Amazon CodeWhisperer", url: "https://aws.amazon.com/codewhisperer", category: "IDE" },
  { id: "codeium", name: "Codeium", url: "https://codeium.com", category: "IDE" },
  { id: "tabnine", name: "TabNine", url: "https://www.tabnine.com", category: "IDE" },
  { id: "warp_terminal", name: "Warp AI Terminal", url: "https://www.warp.dev", category: "IDE" },
  { id: "zed_editor", name: "Zed Editor", url: "https://zed.dev", category: "IDE" },
  { id: "neovim_ai", name: "Neovim + AI Plugins", url: "https://github.com/neovim/neovim", category: "IDE" },
  { id: "emacs_ai", name: "Emacs + AI Plugins", url: "https://www.gnu.org/software/emacs", category: "IDE" },
  { id: "jupyterlab", name: "JupyterLab", url: "https://jupyter.org", category: "IDE" },
  { id: "jupyter_notebook", name: "Jupyter Notebook", url: "https://jupyter.org", category: "IDE" },
  { id: "lm_studio_notebook", name: "LM Studio Notebook", url: "https://lmstudio.ai", category: "IDE" },
  { id: "observablehq", name: "ObservableHQ", url: "https://observablehq.com", category: "IDE" },
  { id: "matlab_live", name: "MATLAB Live Scripts", url: "https://www.mathworks.com/products/matlab/live-editor.html", category: "IDE" },
  { id: "wolfram_notebook", name: "Wolfram Notebook", url: "https://www.wolfram.com/notebooks", category: "IDE" },
  { id: "github_codespaces", name: "GitHub Codespaces", url: "https://github.com/features/codespaces", category: "IDE" },

  // CLI (20)
  { id: "openai_cli", name: "OpenAI CLI", url: "https://platform.openai.com/docs/guides/cli", category: "CLI" },
  { id: "anthropic_cli", name: "Anthropic CLI", url: "https://docs.anthropic.com/claude/docs/anthropic-cli", category: "CLI" },
  { id: "google_ai_cli", name: "Google AI Studio CLI", url: "https://ai.google.dev/gemini-api/docs/quickstart?lang=node", category: "CLI" },
  { id: "hf_cli", name: "HuggingFace CLI", url: "https://huggingface.co/docs/huggingface_hub/quick-start", category: "CLI" },
  { id: "replicate_cli", name: "Replicate CLI", url: "https://replicate.com/docs/reference/cli", category: "CLI" },
  { id: "groq_cli", name: "Groq CLI", url: "https://console.groq.com/docs", category: "CLI" },
  { id: "llama_cpp_cli", name: "llama.cpp CLI", url: "https://github.com/ggerganov/llama.cpp", category: "CLI" },
  { id: "ollama_cli", name: "Ollama CLI", url: "https://github.com/ollama/ollama", category: "CLI" },
  { id: "mlflow_cli", name: "MLflow CLI", url: "https://mlflow.org/docs/latest/command-line-interface.html", category: "CLI" },
  { id: "copilot_cli", name: "GitHub Copilot CLI", url: "https://githubnext.com/projects/copilot-cli", category: "CLI" },
  { id: "aws_cli", name: "AWS CLI (AI/SageMaker)", url: "https://aws.amazon.com/cli", category: "CLI" },
  { id: "azure_cli", name: "Azure CLI for AI", url: "https://learn.microsoft.com/cli/azure", category: "CLI" },
  { id: "modal_cli", name: "Modal CLI", url: "https://modal.com/docs/reference/cli", category: "CLI" },
  { id: "runpod_cli", name: "RunPod CLI", url: "https://docs.runpod.io/docs/cli-overview", category: "CLI" },
  { id: "langchain_cli", name: "LangChain CLI", url: "https://python.langchain.com", category: "CLI" },
  { id: "weaviate_cli", name: "Weaviate CLI", url: "https://weaviate.io/developers/weaviate/cli", category: "CLI" },
  { id: "qdrant_cli", name: "Qdrant CLI", url: "https://qdrant.tech/documentation/tools/cli", category: "CLI" },
  { id: "pinecone_cli", name: "Pinecone CLI", url: "https://docs.pinecone.io", category: "CLI" },
  { id: "ray_cli", name: "Ray CLI", url: "https://docs.ray.io/en/latest/ray-overview/installation.html", category: "CLI" },
  { id: "dvc_cli", name: "DVC", url: "https://dvc.org/doc/command-reference", category: "CLI" },

  // Tools / Frameworks (20)
  { id: "langchain", name: "LangChain", url: "https://python.langchain.com", category: "Tools" },
  { id: "llamaindex", name: "LlamaIndex", url: "https://www.llamaindex.ai", category: "Tools" },
  { id: "hf_transformers", name: "HuggingFace Transformers", url: "https://huggingface.co/transformers", category: "Tools" },
  { id: "hf_diffusers", name: "HuggingFace Diffusers", url: "https://huggingface.co/docs/diffusers", category: "Tools" },
  { id: "pytorch", name: "PyTorch", url: "https://pytorch.org", category: "Tools" },
  { id: "tensorflow", name: "TensorFlow", url: "https://www.tensorflow.org", category: "Tools" },
  { id: "jax", name: "JAX", url: "https://github.com/google/jax", category: "Tools" },
  { id: "keras", name: "Keras", url: "https://keras.io", category: "Tools" },
  { id: "ray", name: "Ray", url: "https://www.ray.io", category: "Tools" },
  { id: "mlflow", name: "MLflow", url: "https://mlflow.org", category: "Tools" },
  { id: "wandb", name: "Weights & Biases", url: "https://wandb.ai", category: "Tools" },
  { id: "cometml", name: "CometML", url: "https://www.comet.com", category: "Tools" },
  { id: "dvc", name: "DVC (Data Version Control)", url: "https://dvc.org", category: "Tools" },
  { id: "kedro", name: "Kedro", url: "https://kedro.org", category: "Tools" },
  { id: "prefect", name: "Prefect", url: "https://www.prefect.io", category: "Tools" },
  { id: "airflow", name: "Apache Airflow", url: "https://airflow.apache.org", category: "Tools" },
  { id: "modal_tool", name: "Modal", url: "https://modal.com", category: "Tools" },
  { id: "autogpt", name: "AutoGPT", url: "https://github.com/Significant-Gravitas/AutoGPT", category: "Tools" },
  { id: "agentkit", name: "AgentKit", url: "https://github.com/openai", category: "Tools" },
  { id: "tensorrt_llm", name: "NVIDIA TensorRT-LLM", url: "https://developer.nvidia.com/tensorrt", category: "Tools" }
];

// Local ratings storage
let ratings = {};
let httpHistory = [];
try {
  httpHistory = JSON.parse(localStorage.getItem("aiDevHubHttpHistory") || "[]");
} catch {
  httpHistory = [];
}
try {
  ratings = JSON.parse(localStorage.getItem("aiDevHubRatings") || "{}");
} catch {
  ratings = {};
}

const searchInput = document.getElementById("search");
const categoryFilter = document.getElementById("categoryFilter");
const categoriesContainer = document.getElementById("categoriesContainer");

// Theme handling
const themeToggle = document.getElementById("themeToggle");
if (localStorage.getItem("theme") === "light") {
  document.body.classList.add("light");
}
themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("light");
  localStorage.setItem("theme", document.body.classList.contains("light") ? "light" : "dark");
});

// Export / import ratings
const exportBtn = document.getElementById("exportRatings");
const importBtn = document.getElementById("importRatingsBtn");
const importFile = document.getElementById("importRatingsFile");

exportBtn.addEventListener("click", () => {
  const dataStr = JSON.stringify(ratings, null, 2);
  const blob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ai-dev-hub-ratings.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

importBtn.addEventListener("click", () => {
  importFile.click();
});

importFile.addEventListener("change", () => {
  const file = importFile.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const parsed = JSON.parse(e.target.result);
      if (typeof parsed === "object" && parsed !== null) {
        ratings = parsed;
        localStorage.setItem("aiDevHubRatings", JSON.stringify(ratings));
        render();
      } else {
        alert("Nieprawidłowy format JSON.");
      }
    } catch (err) {
      alert("Błąd parsowania JSON: " + err);
    }
  };
  reader.readAsText(file);
});

// HTTP tester
const httpPreset = document.getElementById("httpPreset");
const httpMethod = document.getElementById("httpMethod");
const httpUrl = document.getElementById("httpUrl");
const httpHeaders = document.getElementById("httpHeaders");
const httpBody = document.getElementById("httpBody");
const httpSend = document.getElementById("httpSend");
const httpStatus = document.getElementById("httpStatus");
const httpResponseText = document.getElementById("httpResponseText");
const httpHistoryList = document.getElementById("httpHistoryList");
const httpTesterSection = document.querySelector(".http-tester");

httpPreset.addEventListener("change", () => {
  const v = httpPreset.value;
  if (!v) {
    httpMethod.value = "GET";
    httpUrl.value = "";
    httpHeaders.value = "";
    httpBody.value = "";
    return;
  }
  if (v === "openai_chat") {
    httpMethod.value = "POST";
    httpUrl.value = "https://api.openai.com/v1/chat/completions";
    httpHeaders.value = JSON.stringify({
      "Authorization": "Bearer sk-...TU_WSTAW_SWÓJ_KLUCZ...",
      "Content-Type": "application/json"
    }, null, 2);
    httpBody.value = JSON.stringify({
      "model": "gpt-4.1-mini",
      "messages": [
        {"role": "user", "content": "Hello from AI Developer Hub"}
      ]
    }, null, 2);
  } else if (v === "anthropic_messages") {
    httpMethod.value = "POST";
    httpUrl.value = "https://api.anthropic.com/v1/messages";
    httpHeaders.value = JSON.stringify({
      "x-api-key": "sk-ant-...TU_WSTAW_SWÓJ_KLUCZ...",
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json"
    }, null, 2);
    httpBody.value = JSON.stringify({
      "model": "claude-3-5-sonnet-20241022",
      "max_tokens": 64,
      "messages": [
        {"role": "user", "content": "Hello from AI Developer Hub"}
      ]
    }, null, 2);
  } else if (v === "gemini_generate") {
    httpMethod.value = "POST";
    httpUrl.value = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=API_KEY_TUTAJ";
    httpHeaders.value = JSON.stringify({
      "Content-Type": "application/json"
    }, null, 2);
    httpBody.value = JSON.stringify({
      "contents": [
        {
          "parts": [
            {"text": "Hello from AI Developer Hub"}
          ]
        }
      ]
    }, null, 2);
  } else if (v === "hf_inference") {
    httpMethod.value = "POST";
    httpUrl.value = "https://api-inference.huggingface.co/models/gpt2";
    httpHeaders.value = JSON.stringify({
      "Authorization": "Bearer hf_...TWÓJ_TOKEN...",
      "Content-Type": "application/json"
    }, null, 2);
    httpBody.value = JSON.stringify({
      "inputs": "Hello from AI Developer Hub",
      "parameters": {"max_new_tokens": 32}
    }, null, 2);
  }
});

httpSend.addEventListener("click", async () => {
  const url = httpUrl.value.trim();
  if (!url) {
    alert("Podaj URL.");
    return;
  }
  let headers = {};
  if (httpHeaders.value.trim()) {
    try {
      headers = JSON.parse(httpHeaders.value);
    } catch (e) {
      alert("Nagłówki muszą być poprawnym JSONem.");
      return;
    }
  }
  let body = undefined;
  if (httpMethod.value === "POST" && httpBody.value.trim()) {
    body = httpBody.value;
    if (!headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
  }
  httpStatus.textContent = "Wysyłanie...";
  httpResponseText.textContent = "";
  try {
    const resp = await fetch(url, {
      method: httpMethod.value,
      headers,
      body
    });
    httpStatus.textContent = `Status: ${resp.status} ${resp.statusText}`;
    const text = await resp.text();
    httpResponseText.textContent = text;

    // Zapis do historii
    try {
      httpHistory.unshift({
        method: httpMethod.value,
        url,
        headersText: httpHeaders.value,
        bodyText: httpBody.value,
        ts: Date.now()
      });
      httpHistory = httpHistory.slice(0, 10);
      localStorage.setItem("aiDevHubHttpHistory", JSON.stringify(httpHistory));
      renderHttpHistory();
    } catch (e) {
      console.warn("Nie udało się zapisać historii:", e);
    }
  } catch (err) {
    httpStatus.textContent = "Błąd żądania";
    httpResponseText.textContent = String(err);
  }
});

// Populate category filter
const allCategories = Array.from(new Set(tools.map(t => t.category)));
allCategories.forEach(cat => {
  const opt = document.createElement("option");
  opt.value = cat;
  opt.textContent = categoryConfig[cat]?.label || cat;
  categoryFilter.appendChild(opt);
});

// Core render
function render() {
  const q = searchInput.value.toLowerCase();
  const selectedCategory = categoryFilter.value;
  categoriesContainer.innerHTML = "";

  allCategories.forEach(category => {
    if (selectedCategory !== "all" && selectedCategory !== category) return;

    let catTools = tools.filter(t => t.category === category);

    if (q) {
      catTools = catTools.filter(t =>
        t.name.toLowerCase().includes(q) ||
        t.url.toLowerCase().includes(q)
      );
    }

    if (catTools.length === 0) return;

    const section = document.createElement("section");
    section.className = "category-section";

    const summary = document.createElement("div");
    summary.className = "category-summary";
    const iconSpan = document.createElement("span");
    iconSpan.className = "category-icon";
    iconSpan.innerHTML = categoryConfig[category]?.icon || "";
    const nameSpan = document.createElement("span");
    nameSpan.className = "category-name";
    nameSpan.textContent = categoryConfig[category]?.label || category;
    const countSpan = document.createElement("span");
    countSpan.className = "category-count";
    countSpan.textContent = catTools.length + " narz.";

    summary.appendChild(iconSpan);
    summary.appendChild(nameSpan);
    summary.appendChild(countSpan);

    const body = document.createElement("div");
    body.className = "category-body";
    body.style.display = "none";

    const grid = document.createElement("div");
    grid.className = "tools-grid";

    catTools.forEach(tool => {
      const card = document.createElement("div");
      card.className = "tool-card";

      const title = document.createElement("div");
      title.className = "tool-title";
      title.textContent = tool.name;

      const url = document.createElement("div");
      url.className = "tool-url";
      url.innerHTML = `<a href="${tool.url}" target="_blank" rel="noreferrer">${tool.url}</a>`;

      const meta = document.createElement("div");
      meta.className = "tool-meta";
      meta.textContent = category;

      const ratingBox = document.createElement("div");
      ratingBox.className = "rating-box";
      ratingBox.dataset.id = tool.id;

      const ratingData = ratings[tool.id];

      ratingBox.innerHTML = `
        <div class="rating-header">
          <span>Twoja ocena</span>
          <span class="rating-value">${ratingData ? ratingData.rating : "brak"}</span>
        </div>
        <div class="rating-slider-row">
          <input type="range" min="1" max="10" step="1" value="${ratingData ? ratingData.rating : 5}">
          <span>1–10</span>
        </div>
        <input type="text" class="rating-note" placeholder="Krótka notatka (opcjonalnie)">
        <div class="rating-actions">
          <button type="button" class="rating-save">Zapisz ocenę</button>
        </div>
      `;

      card.appendChild(title);
      card.appendChild(url);
      card.appendChild(meta);
      card.appendChild(ratingBox);

      // API → przycisk auto-uzupełniania testera
      if (category === "API") {
        const apiBtn = document.createElement("button");
        apiBtn.type = "button";
        apiBtn.className = "api-tester-btn";
        apiBtn.textContent = "Do testera API";
        apiBtn.addEventListener("click", () => {
          httpPreset.value = "";
          httpMethod.value = "GET";
          httpUrl.value = tool.url;
          httpHeaders.value = "";
          httpBody.value = "";
          if (httpTesterSection) {
            const top = httpTesterSection.getBoundingClientRect().top + window.scrollY - 8;
            window.scrollTo({ top, behavior: "smooth" });
          }
        });
        card.appendChild(apiBtn);
      }

      grid.appendChild(card);
    });

    body.appendChild(grid);
    section.appendChild(summary);
    section.appendChild(body);
    categoriesContainer.appendChild(section);

    summary.addEventListener("click", () => {
      body.style.display = body.style.display === "none" ? "block" : "none";
    });
  });

  applyRatingHandlers();
}

function applyRatingHandlers() {
  document.querySelectorAll(".rating-box").forEach(box => {
    const id = box.dataset.id;
    const slider = box.querySelector('input[type="range"]');
    const noteInput = box.querySelector(".rating-note");
    const valueSpan = box.querySelector(".rating-value");
    const button = box.querySelector(".rating-save");

    const existing = ratings[id];
    if (existing) {
      slider.value = existing.rating;
      valueSpan.textContent = existing.rating;
      noteInput.value = existing.note || "";
    }

    slider.addEventListener("input", () => {
      valueSpan.textContent = slider.value;
    });

    button.addEventListener("click", () => {
      const rating = parseInt(slider.value, 10);
      const note = noteInput.value.trim();
      ratings[id] = { rating, note };
      localStorage.setItem("aiDevHubRatings", JSON.stringify(ratings));
      valueSpan.textContent = rating;
    });
  });
}


function renderHttpHistory() {
  if (!httpHistoryList) return;
  httpHistoryList.innerHTML = "";
  if (!httpHistory || httpHistory.length === 0) {
    const span = document.createElement("span");
    span.style.fontSize = "0.75rem";
    span.style.opacity = "0.8";
    span.textContent = "Brak zapisanej historii.";
    httpHistoryList.appendChild(span);
    return;
  }
  httpHistory.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item-btn";
    const urlShort = item.url && item.url.length > 60 ? item.url.slice(0, 57) + "..." : (item.url || "");
    btn.textContent = `${item.method || "GET"} · ${urlShort}`;
    btn.addEventListener("click", () => {
      httpMethod.value = item.method || "GET";
      httpUrl.value = item.url || "";
      httpHeaders.value = item.headersText || "";
      httpBody.value = item.bodyText || "";
      const top = httpTesterSection.getBoundingClientRect().top + window.scrollY - 8;
      window.scrollTo({ top, behavior: "smooth" });
    });
    httpHistoryList.appendChild(btn);
  });
}

renderHttpHistory();

// Search & filter events
searchInput.addEventListener("input", render);
categoryFilter.addEventListener("change", render);

// Initial render
render();
