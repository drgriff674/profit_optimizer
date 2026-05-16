const CACHE_NAME = "optigain-cache-v14";

self.addEventListener("install", (event) => {

  console.log("SERVICE WORKER INSTALLED");

  event.waitUntil(

    caches.open(CACHE_NAME)

      .then((cache) => {

        return cache.addAll([

  // CORE PAGES
  "/",
  "/dashboard",
  "/expense/entry",
  "/sales",
  "/cash/revenue",
  "/inventory/setup",
  "/inventory/adjust",

  // STATIC FILES
  "/static/manifest.json",
  "/static/css/output.css",

  "/static/js/navigation.js",
  "/static/js/offline-db.js",

  "/static/icon.png",
  "/static/favicon.ico",

  // IMAGES
  "/static/images/hero.jpg",
  "/static/images/payment.jpg",
  "/static/images/security.jpg"

]);
      })

  );

});

// FETCH
self.addEventListener("fetch", (event) => {

  // ONLY CACHE GET REQUESTS
  if (event.request.method !== "GET") {
    return;
  }

  event.respondWith(

    fetch(event.request)

      .then((response) => {

        // CLONE RESPONSE
        const responseClone = response.clone();

        // SAVE VISITED PAGES AUTOMATICALLY
        caches.open(CACHE_NAME)

          .then((cache) => {

            cache.put(event.request, responseClone);

          });

        return response;

      })

      .catch(() => {

        return caches.match(event.request)

          .then((cachedResponse) => {

            // RETURN CACHED PAGE IF EXISTS
            if (cachedResponse) {
              return cachedResponse;
            }

            // FALLBACK OFFLINE PAGE
            return new Response(

              `
              <html>
              <head>
                <title>Offline</title>

                <style>
                  body{
                    font-family:sans-serif;
                    background:#0f172a;
                    color:white;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    height:100vh;
                    text-align:center;
                    padding:20px;
                  }

                  .box{
                    max-width:400px;
                  }

                  h1{
                    font-size:28px;
                    margin-bottom:10px;
                  }

                  p{
                    opacity:0.8;
                  }
                </style>

              </head>

              <body>

                <div class="box">

                  <h1>📡 Offline</h1>

                  <p>
                    This page is not cached yet.<br>
                    Reconnect once and visit it online first.
                  </p>

                </div>

              </body>
              </html>
              `,
              {
                headers: {
                  "Content-Type": "text/html"
                }
              }

            );

          });

      })

  );

});

// ACTIVATE
self.addEventListener("activate", (event) => {

  console.log("SERVICE WORKER ACTIVATED");

  event.waitUntil(

    caches.keys().then((cacheNames) => {

      return Promise.all(

        cacheNames.map((cache) => {

          if (cache !== CACHE_NAME) {

            return caches.delete(cache);

          }

        })

      );

    })

  );

});