const CACHE_NAME = "optigain-cache-v1";

const urlsToCache = [
  "/",
  "/static/css/output.css",
  "/static/icon.png",
  "/static/favicon.ico",
  "/static/images/hero.jpg",
  "/static/images/payment.jpg",
  "/static/images/security.jpg"
];


// INSTALL SERVICE WORKER
self.addEventListener("install", (event) => {

  console.log("✅ OptiGain Service Worker Installed");

  event.waitUntil(

    caches.open(CACHE_NAME)
      .then((cache) => {

        console.log("✅ Files cached");

        return cache.addAll(urlsToCache);

      })

  );

});


// FETCH FILES
self.addEventListener("fetch", (event) => {

  event.respondWith(

    caches.match(event.request)
      .then((response) => {

        // Return cached version if available
        if (response) {
          return response;
        }

        // Otherwise fetch from internet
        return fetch(event.request);

      })

  );

});