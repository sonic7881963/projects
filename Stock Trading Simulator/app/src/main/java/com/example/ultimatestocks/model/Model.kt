package com.example.ultimatestocks.model

import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import cafe.adriel.voyager.navigator.tab.Tab
import co.yml.charts.common.extensions.isNotNull
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.NewsEvent
import com.example.ultimatestocks.entities.Player
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.entities.Stock
import com.example.ultimatestocks.entities.StockDetails
import com.example.ultimatestocks.entities.StockNewPrice
import com.example.ultimatestocks.entities.FirebaseStock
import com.google.firebase.Firebase
import com.google.firebase.auth.auth
import com.google.firebase.firestore.firestore
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
//import io.ktor.client.*
//import io.ktor.client.call.body
//import io.ktor.client.engine.cio.*
//import io.ktor.client.request.get
//import io.ktor.client.statement.HttpResponse
import kotlinx.coroutines.*
import kotlinx.coroutines.launch
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.generationConfig
import com.google.firebase.firestore.EventListener
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow


// The model contains all of the application's internal state
// on the client side.
class Model : IPublisher() {
    val db = Firebase.firestore
    val auth = Firebase.auth
    val sandboxes = mutableListOf<Sandbox>()
    val activeComps = mutableListOf<Competition>()
    val stocks = mutableListOf<FirebaseStock>()
//    val client = HttpClient(CIO)

//    val competitions = mutableListOf<Competition>()

    // The data of a logged in user
    var userUID = ""
    var userEmail = ""
    var userName = ""
    var isAdmin = false


    // Some Addition made to the model
    val previousTabs = mutableListOf<Tab>()
    var SandCompState = ""
    var BuySellState = ""
    var Currency = "USD ($)"
    var NewsList = mutableListOf<NewsEvent>()
    var selectedNews = NewsEvent("", "", listOf())

    init {
        getUserData()
//        addTestCompetition()
    }


    private val systemPrompt = "Respond with JSON object representing a news event, with the only the following properties: title (string), body (string), newPrices: (array of objects of format {ticker: string, newPrice: float, changeInPrice: float})." +
            "Given an input of a list of stocks (including ticker symbol and company description), your role is to generate a truly random but realistic event that is equally likely to have a positive or negative impact on the stocks and is not directly related to the stocks, the government, or earthquakes, and new prices for each of the stocks as a result of the news event. \n Input:"
//
    val geminiModel = GenerativeModel(
        "gemini-1.5-flash",
        "AIzaSyDyouk2xVnqZ2hq1mt7PVaXglncxnqUgnQ",
        generationConfig = generationConfig {
            temperature = 2f
            topK = 40
            topP = 0.95f
            maxOutputTokens = 8192
            responseMimeType = "application/json"
        },
    )
//


    // main function for calling the Firebase Function that then calls the ChatGPT API
    suspend fun generateNewsEvent(stocks: List<StockDetails>): NewsEvent {

        var newStr = "["
        for (stock in stocks) {
            newStr += "{ ticker: " + stock.ticker + ", description: " + stock.description + ", currentPrice: " + stock.historicPrice.last() + "}, "
        }
        newStr += "]"

        val response = geminiModel.generateContent(systemPrompt + newStr)
        println(response.text)

        val newsEvent = Json.decodeFromString<NewsEvent>(response.text as String)
        return newsEvent
//        GlobalScope.launch {
//            val response = geminiModel.generateContent(systemPrompt + newStr)
//            val newsEvent = Json.decodeFromString<NewsEvent>(response.text as String)
//
////            response = client.get("https://catfact.ninja/fact")
////            println("Response received from API")
////            val responseBody: String = response.body()
////            println(responseBody)
//        }

    }

    fun saveSandbox(sandboxID: String, sandbox: Sandbox) {
        if (auth.currentUser != null) { // if signed in
            val userID = auth.currentUser!!.uid
            val newSandboxString = Json.encodeToString(sandbox)
            val sandboxObject = hashMapOf(
                "ownerID" to userID,
                "sandboxString" to newSandboxString
            )

            db.collection("Sandboxes").document(sandbox.id).set(sandboxObject)
                .addOnSuccessListener { println("DocumentSnapshot successfully written!") }
                .addOnFailureListener { e -> println("Error writing document: " + e) }

            notifySubscribers()
        }
    }

    // Read all the user's data from firebase and store it in model variables.
    fun getUserData() {
        if (auth.currentUser != null) {
            db.collection("Users").document(auth.currentUser!!.uid).get()
            .addOnSuccessListener { document ->
                if (document != null) {
                    userUID = document.id
                    userName = (if (document.data?.get("userName") == null) "Temp name" else document.data?.get("userName").toString())
                    userEmail = (if (document.data?.get("userEmail") == null) "Temp email" else document.data?.get("userEmail").toString())
                    isAdmin = (if (document.data?.get("isAdmin") == null) false else document.data?.get("isAdmin") as Boolean)
                }
                getSandboxes() // fetch sandboxes and store globally

                // Notify the subscribers since model variables have changed
                notifySubscribers()
            }.addOnFailureListener { exception ->
                println("get failed with " + exception)
            }
        }

    }
    fun getSandboxes() {
        if (auth.currentUser != null) { // if signed in
            // Create a reference to the cities collection
            val sandboxesRef = db.collection("Sandboxes")

            // Create a query against the collection.
            val query = sandboxesRef.whereEqualTo("ownerID", auth.currentUser!!.uid)
            query.get()
                .addOnSuccessListener { documents ->
                    sandboxes.clear()
                    for (document in documents) {
                        // document.data.sandboxString stores the encoded JSON string
                        val newSandbox = Json.decodeFromString<Sandbox>(document.data.get("sandboxString") as String)
                        sandboxes.add(newSandbox)
                    }
                    notifySubscribers()
                }
                .addOnFailureListener { e -> println("Error reading sandbox documents: " + e) }
        }
    }

    // get Competitions from database, update model variables, notify subscribers
    fun getCompetitions() {
        activeComps.clear()
        db.collection("Competitions").whereEqualTo("isActive", true).get()
            .addOnSuccessListener { documents ->

            for (document in documents) {
                // document.data.sandboxString stores the encoded JSON string
                val newComp = Json.decodeFromString<Competition>(document.data.get("compString") as String)
                activeComps.add(newComp)
            }
            notifySubscribers()
        }
            .addOnFailureListener { e -> println("Error reading competition documents: " + e) }
    }




    // Add a testing competition to firebase.
//    fun addTestCompetition() {
//        if (auth.currentUser != null) {
//        var stocks = mutableListOf<Stock>(Stock("CMP", "Company", "desc", 50.0f, "Sector"))
//        var players = mutableListOf<Player>(Player(auth.currentUser!!.uid, 1f, mutableListOf<Stock>(Stock("OWN", "Owned Stock", "desc", 50.0f, "Sector"))))
//        var newsEvents = mutableListOf<NewsEvent>(NewsEvent("News Title", "News Body", listOf(
//            StockNewPrice("TCK", 5f, 10f)
//        )))
//
//        db.collection("Competitions").add(hashMapOf(
//            "title" to "Test",
//            "availableStocks" to Json.encodeToString(stocks),
//            "initialCash" to 500,
//            "players" to Json.encodeToString(players),
//            "maxNewsEvents" to 5,
//            "newsEvents" to Json.encodeToString(newsEvents)
//        ))
//            }
//    }

    fun add(sandbox: Sandbox) {
        if (auth.currentUser != null) { // if signed in
            val userID = auth.currentUser!!.uid
//            val newSandbox = hashMapOf(
//                "ownerId" to userID,
//                "name" to sandbox.name,
//                "timeDuration" to sandbox.timeDuration,
//                "newsEventsGenerated" to sandbox.newsEventsGenerated,
//                "stocks" to sandbox.stocks
//            )
            val newSandboxString = Json.encodeToString(sandbox)
            val sandboxObject = hashMapOf(
                "ownerID" to userID,
                "sandboxString" to newSandboxString
            )

            db.collection("Sandboxes").document(sandbox.id).set(sandboxObject)
                .addOnSuccessListener { println("DocumentSnapshot successfully written!") }
                .addOnFailureListener { e -> println("Error writing document: " + e) }
            sandboxes.add(sandbox)
            notifySubscribers()
        }
    }


    fun attemptSignUp(
        email: String, password: String, confirmPassword: String, username: String,
        onSuccess: () -> Unit, onError: (String) -> Unit
    ) {
        if (password != confirmPassword) {
            onError("Passwords do not match.")
            return
        }

        auth.createUserWithEmailAndPassword(email, password).addOnCompleteListener { task ->
            if (task.isSuccessful) {
                println("createUserWithEmailAndPassword:success")
                db.collection("Users").document(auth.currentUser!!.uid)
                    .set(hashMapOf("userName" to username, "userEmail" to email, "isAdmin" to false))
                    .addOnSuccessListener { getUserData() }
                    .addOnFailureListener { e -> println("Error writing document: $e") }
                notifySubscribers()
                onSuccess()
            } else {
                val errorMessage = task.exception?.localizedMessage ?: "Sign-up failed. Please try again."
                println("createUserWithEmailAndPassword:failed - $errorMessage")
                onError(errorMessage)
            }
        }
    }
    fun getAllStocks() {
        if (auth.currentUser != null) { // if signed in
            // Create a reference to the cities collection
            val stocksRef = db.collection("Stocks")


            stocksRef.get()
                .addOnSuccessListener { documents ->

                    for (document in documents) {
                        // document.data.sandboxString stores the encoded JSON string
                        stocks.add(FirebaseStock(
                            document.data.get("companyName").toString(),
                            document.data.get("description").toString(),
                            document.data.get("sector").toString(),
                            document.data.get("startingPrice") as Double,
                            document.data.get("ticker").toString()
                        ))

                    }
                    notifySubscribers()
                }
                .addOnFailureListener { e -> println("Error reading sandbox documents: " + e) }
        }
    }

    fun refreshCompetition(comp: Competition, onSuccess: (returnedComp: Competition) -> Unit) {

        db.collection("Competitions").document(comp.id).get()
            .addOnSuccessListener { document ->
                if (document.isNotNull()) {
                    // document.data.sandboxString stores the encoded JSON string
                    val newComp = Json.decodeFromString<Competition>(document.data?.get("compString") as String)
                    onSuccess(newComp)
                }

                notifySubscribers()
            }
            .addOnFailureListener { e -> println("Error reading competition documents: " + e) }

    }

    fun attemptSignIn(
        email: String,
        password: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        // Input validation
        if (email.isEmpty()) {
            onError("Email cannot be empty.")
            return
        }

        if (!android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
            onError("Please enter a valid email address.")
            return
        }

        if (password.isEmpty()) {
            onError("Password cannot be empty.")
            return
        }

        if (password.length < 6) {
            onError("Password must be at least 6 characters long.")
            return
        }

        // Proceed with Firebase authentication
        auth.signInWithEmailAndPassword(email, password).addOnCompleteListener { task ->
            if (task.isSuccessful) {
                println("signInWithEmail:success")
                onSuccess()
                getUserData()
                notifySubscribers()
            } else {
                val errorMessage = task.exception?.localizedMessage ?: "Sign-in failed. Please check your credentials."
                println("signInWithEmail:failure - $errorMessage")
                onError(errorMessage)
            }
        }
    }


    fun attemptLogout() {
        auth.signOut()

        // Important: reset the model variables for the user data
        userUID = ""
        userName = ""
        userEmail = ""
        notifySubscribers()
    }

    fun createCompetition(comp: Competition) {
        if (auth.currentUser != null) { // if signed in
            val userID = auth.currentUser!!.uid
            val newCompString = Json.encodeToString(comp)
            val compObject = hashMapOf(
                "isActive" to true,
                "compString" to newCompString
            )

            db.collection("Competitions").document(comp.id).set(compObject)
                .addOnSuccessListener { println("DocumentSnapshot successfully written!") }
                .addOnFailureListener { e -> println("Error writing document: " + e) }
            activeComps.add(comp)
            notifySubscribers()
        }
    }

    fun saveCompetition(comp: Competition) {
        if (auth.currentUser != null) { // if signed in
            val userID = auth.currentUser!!.uid
            val newCompString = Json.encodeToString(comp)
            val compObject = hashMapOf(
                "isActive" to true,
                "compString" to newCompString
            )

            db.collection("Competitions").document(comp.id).set(compObject)
                .addOnSuccessListener { println("DocumentSnapshot successfully written!") }
                .addOnFailureListener { e -> println("Error writing document: " + e) }

            notifySubscribers()
        }
    }

    fun del(sandbox: Sandbox) {
        db.collection("Sandboxes").document(sandbox.id).delete()
            .addOnSuccessListener { println("Sandbox deleted") }
            .addOnFailureListener { e -> println("Error writing document$e") }

        sandboxes.remove(sandbox)
        notifySubscribers()
    }

    fun setBuySell(bsState: String, scState: String){
        BuySellState = bsState
        SandCompState = scState
        notifySubscribers()
    }

    fun getCurrency(currency: String){
        Currency = currency
        notifySubscribers()
    }

    fun setNews(news: NewsEvent){
        selectedNews = news
        notifySubscribers()
    }

//    fun get
}