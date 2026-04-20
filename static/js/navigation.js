// ================= GLOBAL NAVIGATION =================

document.addEventListener("click", function (e) {
  const link = e.target.closest("a");

  if (!link || link.target === "_blank" || link.hasAttribute("download")) return;

  const href = link.getAttribute("href");

  // 🚫 ignore external links (Pesapal fix)
  if (
    !href ||
    href.startsWith("#") ||
    href.startsWith("javascript") ||
    href.startsWith("http")
  ) return;

  e.preventDefault();

  const loader = document.getElementById("pageLoader");
  if (loader) loader.classList.remove("hidden");

  document.body.style.transform = "scale(0.995)";

  setTimeout(() => {
    window.location.href = href;
  }, 120);
});

// ================= LOADER RESET =================

function hideLoader() {
  const loader = document.getElementById("pageLoader");
  if (loader) loader.classList.add("hidden");
}

window.addEventListener("load", hideLoader);

window.addEventListener("pageshow", function (event) {
  if (event.persisted) hideLoader();
});