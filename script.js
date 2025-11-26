const defaultServices = [
  { name: "OpenAI Console", url: "https://platform.openai.com" },
  { name: "OpenAI API Docs", url: "https://platform.openai.com/docs/api-reference" },
  { name: "Anthropic Console", url: "https://console.anthropic.com" },
  { name: "Google AI Studio", url: "https://aistudio.google.com" },
  { name: "HuggingFace", url: "https://huggingface.co" },
  { name: "Supabase", url: "https://supabase.com" },
  { name: "Replicate", url: "https://replicate.com" }
];

const savedServices = JSON.parse(localStorage.getItem("services") || "[]");
let services = [...defaultServices, ...savedServices];

const container = document.getElementById("services");
const search = document.getElementById("search");

function render() {
  container.innerHTML = "";
  const q = search.value.toLowerCase();
  services
    .filter(s => s.name.toLowerCase().includes(q))
    .forEach(s => {
      const div = document.createElement("div");
      div.className = "card";
      div.innerHTML = `<strong>${s.name}</strong><br><a href="${s.url}" target="_blank">${s.url}</a>`;
      container.appendChild(div);
    });
}

search.addEventListener("input", render);

document.getElementById("addBtn").onclick = () => {
  const name = document.getElementById("newName").value.trim();
  const url = document.getElementById("newURL").value.trim();
  if (!name || !url) return;
  services.push({ name, url });
  localStorage.setItem("services", JSON.stringify(services.filter(s => !defaultServices.includes(s))));
  render();
};

document.getElementById("themeToggle").onclick = () => {
  document.body.classList.toggle("light");
  localStorage.setItem("theme", document.body.classList.contains("light") ? "light" : "dark");
};

if (localStorage.getItem("theme") === "light") {
  document.body.classList.add("light");
}

render();