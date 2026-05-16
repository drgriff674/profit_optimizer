if ("serviceWorker" in navigator) {

    window.addEventListener("load", async () => {

        try {

            const registration =
                await navigator.serviceWorker.register("/static/sw.js");

            console.log("✅ Service Worker Registered");

        } catch (error) {

            console.log("❌ SW registration failed:", error);

        }

    });

}