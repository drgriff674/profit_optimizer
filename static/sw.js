const CACHE_NAME = "optigain-cache-v11";

self.addEventListener("install", (event) => {

  console.log("INSTALL STARTED");

  event.waitUntil(

    caches.open(CACHE_NAME)

      .then((cache) => {

        console.log("CACHE OPENED");

        return cache.addAll([
          "/",
          "/static/manifest.json",
          "/static/css/output.css",
          "/static/icon.png",
          "/static/favicon.ico",
          "/static/images/hero.jpg",
          "/static/images/payment.jpg",
          "/static/images/security.jpg"
        ]);

      })

      .then(() => {

        console.log("FILES CACHED SUCCESSFULLY");

      })

      .catch((error) => {

        console.error("CACHE FAILED:", error);

      })

  );

});

self.addEventListener("fetch", (event) => {

  event.respondWith(

    caches.match(event.request)

      .then((response) => {

        // RETURN CACHE IF FOUND
        if (response) {
          return response;
        }

        // OTHERWISE FETCH FROM NETWORK
        return fetch(event.request)

          .catch(() => {

            // FALLBACK TO HOMEPAGE OFFLINE
            return caches.match("/");

          });

      })

  );

});

self.addEventListener("activate", (event) => {

  console.log("SERVICE WORKER ACTIVATED");

  event.waitUntil(

    caches.keys().then((cacheNames) => {

      return Promise.all(

        cacheNames.map((cache) => {

          if (cache !== CACHE_NAME) {

            console.log("OLD CACHE REMOVED:", cache);

            return caches.delete(cache);

          }

        })

      );

    })

  );

});