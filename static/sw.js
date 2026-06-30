const CACHE_NAME = "optigain-cache-v16";

const STATIC_ASSETS = [

  "/",

  "/features",
  "/pricing",
  "/about",
  "/contact",
  "/how-it-works",
  "/help-center",
  "/documentation",
  "/faq",
  "/privacy",
  "/terms",
  "/login",
  "/register",

  "/static/manifest.json",
  "/static/css/output.css",

  "/static/js/navigation.js",
  "/static/js/offline-db.js",

  "/static/icon.png",
  "/static/favicon.ico",

  "/static/images/hero.jpg",
  "/static/images/payment.jpg",
  "/static/images/security.jpg"

];

// INSTALL
self.addEventListener("install", event => {

  console.log("✅ Service Worker Installed");

  event.waitUntil(

    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())

  );

});

// ACTIVATE
self.addEventListener("activate", event => {

  console.log("✅ Service Worker Activated");

  event.waitUntil(

    caches.keys().then(cacheNames =>

      Promise.all(

        cacheNames.map(cache => {

          if (cache !== CACHE_NAME) {

            return caches.delete(cache);

          }

        })

      )

    ).then(() => self.clients.claim())

  );

});

// FETCH
self.addEventListener("fetch", event => {

  if (event.request.method !== "GET") return;

  event.respondWith(

    caches.match(event.request).then(cachedResponse => {

      if (cachedResponse) {

        fetch(event.request)

          .then(networkResponse => {

            if (
              networkResponse &&
              networkResponse.status === 200
            ) {

              caches.open(CACHE_NAME).then(cache => {

                cache.put(
                  event.request,
                  networkResponse.clone()
                );

              });

            }

          })

          .catch(() => {});

        return cachedResponse;

      }

      return fetch(event.request)

        .then(networkResponse => {

          if (
            networkResponse &&
            networkResponse.status === 200
          ) {

            caches.open(CACHE_NAME).then(cache => {

              cache.put(
                event.request,
                networkResponse.clone()
              );

            });

          }

          return networkResponse;

        })

        .catch(() => {

          return new Response(`

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Offline</title>

<style>

body{

margin:0;

font-family:Arial,sans-serif;

background:#020617;

color:white;

display:flex;

justify-content:center;

align-items:center;

height:100vh;

text-align:center;

padding:30px;

}

.box{

max-width:420px;

}

h1{

font-size:32px;

margin-bottom:15px;

}

p{

opacity:.75;

line-height:1.7;

}

</style>

</head>

<body>

<div class="box">

<h1>📡 You're Offline</h1>

<p>

The page isn't available yet because it hasn't been cached.

Reconnect to the internet and open this page once.

After that it will be available offline.

</p>

</div>

</body>

</html>

          `, {

            headers: {

              "Content-Type":"text/html"

            }

          });

        });

    })

  );

});