// ================================
// OptiGain Navigation Optimizer
// ================================

// Prefetch internal pages when users hover or touch links.
// This makes navigation feel much faster.

document.querySelectorAll("a[href]").forEach(link => {

    const href = link.getAttribute("href");

    // Ignore external links, anchors, javascript and downloads
    if (
        !href ||
        href.startsWith("#") ||
        href.startsWith("javascript:") ||
        href.startsWith("http") ||
        link.hasAttribute("download")
    ) {
        return;
    }

    const prefetch = () => {
        fetch(href, {
            credentials: "same-origin"
        }).catch(() => {});
    };

    link.addEventListener("mouseenter", prefetch, { once: true });
    link.addEventListener("touchstart", prefetch, { once: true });

});