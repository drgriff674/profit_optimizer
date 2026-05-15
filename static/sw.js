const CACHE_NAME = "optigain-cache-v12";

self.addEventListener("install", (event) => {

  console.log("INSTALL STARTED");

  event.waitUntil(

    caches.open(CACHE_NAME)

      .then((cache) => {

        console.log("CACHE OPENED");

        return cache.addAll([

          "/",
          "/dashboard",
          "/expense/entry",

          "/static/manifest.json",
          "/static/css/output.css",

          "/static/js/navigation.js",
          "/static/js/offline-db.js",

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

    fetch(event.request)

      .then((response) => {

        // CLONE RESPONSE
        const responseClone = response.clone();

        // UPDATE CACHE
        caches.open(CACHE_NAME)

          .then((cache) => {

            cache.put(event.request, responseClone);

          });

        return response;

      })

      .catch(() => {

        return caches.match(event.request)

          .then((cachedResponse) => {

            // RETURN CACHED VERSION
            if (cachedResponse) {
              return cachedResponse;
            }

            // FALLBACK
            return caches.match("/dashboard");

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