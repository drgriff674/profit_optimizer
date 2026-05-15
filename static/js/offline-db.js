const DB_NAME = "optigain_offline_db";
const DB_VERSION = 1;

let db;

// OPEN DATABASE
function openDatabase() {

    return new Promise((resolve, reject) => {

        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => {
            console.error("IndexedDB failed");
            reject("DB failed");
        };

        request.onsuccess = () => {
            db = request.result;
            console.log("IndexedDB ready");
            resolve(db);
        };

        request.onupgradeneeded = (event) => {

            db = event.target.result;

            // EXPENSE STORE
            if (!db.objectStoreNames.contains("expenses")) {

                const store = db.createObjectStore("expenses", {
                    keyPath: "id",
                    autoIncrement: true
                });

                store.createIndex("synced", "synced", {
                    unique: false
                });

                console.log("Expenses store created");
            }
        };
    });
}

// SAVE OFFLINE EXPENSE
async function saveOfflineExpense(expenseData) {

    if (!db) {
        await openDatabase();
    }

    return new Promise((resolve, reject) => {

        const transaction = db.transaction(["expenses"], "readwrite");

        const store = transaction.objectStore("expenses");

        const request = store.add({
            ...expenseData,
            synced: false,
            created_offline_at: new Date().toISOString()
        });

        request.onsuccess = () => {
            console.log("Expense saved offline");
            resolve(true);
        };

        request.onerror = () => {
            console.error("Failed saving offline expense");
            reject(false);
        };
    });
}

// GET UNSYNCED EXPENSES
async function getUnsyncedExpenses() {

    if (!db) {
        await openDatabase();
    }

    return new Promise((resolve, reject) => {

        const transaction = db.transaction(["expenses"], "readonly");

        const store = transaction.objectStore("expenses");

        const request = store.getAll();

        request.onsuccess = () => {

            const unsynced = request.result.filter(
                item => item.synced === false
            );

            resolve(unsynced);
        };

        request.onerror = () => {
            reject([]);
        };
    });
}

// MARK EXPENSE AS SYNCED
async function markExpenseSynced(id) {

    if (!db) {
        await openDatabase();
    }

    return new Promise((resolve, reject) => {

        const transaction = db.transaction(["expenses"], "readwrite");

        const store = transaction.objectStore("expenses");

        const getRequest = store.get(id);

        getRequest.onsuccess = () => {

            const data = getRequest.result;

            if (!data) {
                resolve(false);
                return;
            }

            data.synced = true;

            const updateRequest = store.put(data);

            updateRequest.onsuccess = () => {
                resolve(true);
            };

            updateRequest.onerror = () => {
                reject(false);
            };
        };
    });
}

// INITIALIZE
openDatabase();

// ================= AUTO SYNC =================

async function syncOfflineExpenses() {

    if (!navigator.onLine) {
        return;
    }

    const expenses = await getUnsyncedExpenses();

    console.log("Unsynced expenses:", expenses);

    for (const expense of expenses) {

        try {

            const csrf =
                document.querySelector('[name="csrf_token"]').value;

            const response = await fetch("/expense/entry", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": csrf
                },
                body: JSON.stringify({
                    amount: expense.amount,
                    category: expense.category,
                    description: expense.description,
                    date: expense.date
                })
            });

            if (response.ok) {

                await markExpenseSynced(expense.id);

                console.log("✅ Expense synced:", expense.id);

            } else {

                console.log("❌ Sync failed");
            }

        } catch (err) {

            console.log("❌ Network sync error:", err);
        }
    }
}

// AUTO SYNC WHEN INTERNET RETURNS
window.addEventListener("online", () => {

    console.log("🌐 Internet restored");

    syncOfflineExpenses();
});

// ALSO TRY ON PAGE LOAD
window.addEventListener("load", () => {

    syncOfflineExpenses();
});