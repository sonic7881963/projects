package com.example.ultimatestocks.sharedPages

import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.compete.OngoingCompetitionPage
import com.example.ultimatestocks.competeAdmin.ManageCompetitionPage
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.NewsEvent
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.llightBlue
import com.example.ultimatestocks.pprimaryBlack
import com.example.ultimatestocks.ppureWhite
import com.example.ultimatestocks.secondaryBlackk
import com.google.ai.client.generativeai.type.ServerException
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking

// List of news events
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewsPage(sandbox: Sandbox, onBack: () -> Unit) {
    var newsList by remember { mutableStateOf<List<NewsEvent>>(sandbox.newsEvents) }
    var selectedNews by remember { mutableStateOf<NewsEvent?>(null) }
    var showError by remember { mutableStateOf(false) }

    if (selectedNews != null) {
        NewsDetailPage(newsItem = selectedNews!!, onBack = { selectedNews = null }, sandbox)
    } else {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("News", color = deeepBlue, fontWeight = FontWeight.Bold) },
                    navigationIcon = {
                        IconButton(onClick = onBack) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                                tint = deeepBlue
                            )
                        }
                    },
                    actions = {
                        Button(
                            onClick = {
                                try {
                                    runBlocking {
                                        val newEvent = async {
                                            MainActivity.model.generateNewsEvent(sandbox.allStocks)
                                        }
                                        newsList += newEvent.await()
                                        sandbox.newsEvents = newsList as MutableList<NewsEvent>
                                        sandbox.latestNewsNotRevealed = newEvent.await().title
                                        MainActivity.model.saveSandbox(sandbox.id, sandbox)
                                    }
                                } catch (_: ServerException) {
                                    showError = true
                                }
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                            shape = RoundedCornerShape(15.dp),
                            elevation = ButtonDefaults.buttonElevation(4.dp),
                            enabled = sandbox.latestNewsNotRevealed.length == 0
                        ) {
                            Text(
                                text = "New Event",
                                color = ppureWhite,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
                )
            }
        ) { innerPadding ->

            if (showError) {
                AlertDialog(
                    title = {
                        Text(text = "Failed to generate event")
                    },
                    text = {
                        Text(text = "Uh Oh! It seems like we failed to generate the news event. This can happen if the model is overloaded. Please try again later.")
                    },
                    onDismissRequest = {
                        showError = false
                    },
                    confirmButton = {
                        TextButton(
                            onClick = {
                                showError = false
                            }
                        ) {
                            Text("Close")
                        }
                    }
                )
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .background(ppureWhite)
                    .padding(innerPadding)
                    .padding(16.dp)


            ) {
                items(newsList.reversed()) { newsItem ->
                    NewsListItem(newsItem = newsItem, onClick = { selectedNews = newsItem })
                }
            }
        }
    }
}


// [Competition Flow] Details of a particular news event
class NewsListPage(val comp: Competition) : Screen {
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        var newsList by remember { mutableStateOf<List<NewsEvent>>(comp.newsEvents) }
        var selectedNews by remember { mutableStateOf<NewsEvent?>(null) }
        var showError by remember { mutableStateOf(false) }
        var navigator = LocalNavigator.currentOrThrow

        if (selectedNews != null) {
            NewsDetailPage(newsItem = selectedNews!!, onBack = { selectedNews = null }, comp)
        } else {
            Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text("News", color = deeepBlue, fontWeight = FontWeight.Bold) },
                        navigationIcon = {
                            IconButton(
                                onClick = {
                                    if (MainActivity.model.isAdmin) {
                                        navigator.push(ManageCompetitionPage(comp))
                                    } else {
                                        navigator.push(OngoingCompetitionPage(comp))
                                    }
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                    contentDescription = "Back",
                                    tint = deeepBlue
                                )
                            }
                        },
                        actions = {
                            if (MainActivity.model.isAdmin) {
                                Button(
                                    onClick = {
                                        try {
                                            runBlocking {
                                                val newEvent = async { MainActivity.model.generateNewsEvent(comp.availableStocks) }
                                                newsList += newEvent.await()
                                                comp.newsEvents = newsList as MutableList<NewsEvent>
                                                comp.latestNewsNotRevealed = newEvent.await().title
                                                MainActivity.model.saveCompetition(comp)
                                            }
                                        } catch (_: ServerException) {
                                            showError = true
                                        }
                                    },
                                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                                    shape = RoundedCornerShape(15.dp),
                                    elevation = ButtonDefaults.buttonElevation(4.dp),
                                    enabled = comp.latestNewsNotRevealed.length == 0 && comp.hasStarted
                                ) {
                                    Text(
                                        text = "New Event (Admin Only)",
                                        color = ppureWhite,
                                        fontWeight = FontWeight.Bold
                                    )
                                }

                            }
                        },
                        colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
                    )
                }
            ) { innerPadding ->
                if (showError) {
                    AlertDialog(
                        title = {
                            Text(text = "Failed to generate event")
                        },
                        text = {
                            Text(text = "Uh Oh! It seems like we failed to generate the news event. This can happen if the model is overloaded. Please try again later.")
                        },
                        onDismissRequest = {
                            showError = false
                        },
                        confirmButton = {
                            TextButton(
                                onClick = {
                                    showError = false
                                }
                            ) {
                                Text("Close")
                            }
                        }
                    )
                }
                LazyColumn(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(ppureWhite)
                        .padding(innerPadding)
                        .padding(16.dp)


                ) {
                    items(newsList.reversed()) { newsItem ->
                        NewsListItem(newsItem = newsItem, onClick = { selectedNews = newsItem })
                    }
                }
            }
        }
    }
}


// [Competition Flow] Details of a particular news event
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewsDetailPage(newsItem: NewsEvent, onBack: () -> Unit, comp: Competition) {

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "News Event",
                        color = deeepBlue,
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = deeepBlue
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(ppureWhite)
                .padding(innerPadding)
                .padding(16.dp)
                .padding(bottom = 100.dp)
                .verticalScroll(ScrollState(0), true)

        ) {
            Text(
                text = newsItem.title,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = pprimaryBlack
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = newsItem.body,
                fontSize = 16.sp,
                color = secondaryBlackk,
                lineHeight = 24.sp
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "New Stock Prices:",
                fontSize = 16.sp,
                color = secondaryBlackk,
                lineHeight = 24.sp
            )

            if (!MainActivity.model.isAdmin && comp.latestNewsNotRevealed == newsItem.title) {
                Text("Price updates have not yet been revealed by the Competition moderator. Check back soon!")
            } else {
                newsItem.newPrices.forEach { newPrice ->
                    Text(
                        text = newPrice.ticker + ": " + newPrice.newPrice.toString() + "( " + (if (newPrice.changeInPrice >= 0) "+" else "") + newPrice.changeInPrice.toString() + ")",
                        fontSize = 16.sp,
                        color = secondaryBlackk,
                        lineHeight = 24.sp
                    )
                }
            }


//            Card(
//                modifier = Modifier.fillMaxWidth(),
//                colors = CardDefaults.cardColors(containerColor = llightBlue),
//                shape = RoundedCornerShape(20.dp),
//                elevation = CardDefaults.cardElevation(4.dp)
//            ) {
//                Column(modifier = Modifier.padding(16.dp)) {
//                    Text(
//                        text = newsItem.title,
//                        fontSize = 24.sp,
//                        fontWeight = FontWeight.Bold,
//                        color = pprimaryBlack
//                    )
//                    Spacer(modifier = Modifier.height(16.dp))
//                    Text(
//                        text = newsItem.body,
//                        fontSize = 16.sp,
//                        color = secondaryBlackk,
//                        lineHeight = 24.sp
//                    )
//                    Spacer(modifier = Modifier.height(16.dp))
//                    Text(
//                        text = "New Stock Prices:",
//                        fontSize = 16.sp,
//                        color = secondaryBlackk,
//                        lineHeight = 24.sp
//                    )
//                    newsItem.newPrices.forEach { newPrice ->
//                        Text(
//                            text = newPrice.ticker + ": " + newPrice.newPrice.toString(),
//                            fontSize = 16.sp,
//                            color = secondaryBlackk,
//                            lineHeight = 24.sp
//                        )
//                    }
//                }
//            }
        }
    }
}


// [Sandboxes Flow] Details of a particular news event
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewsDetailPage(newsItem: NewsEvent, onBack: () -> Unit, sandbox: Sandbox) {
    var latestNewsNotRevealed by remember { mutableStateOf(sandbox.latestNewsNotRevealed) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "News Event",
                        color = deeepBlue,
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = deeepBlue
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(ppureWhite)
                .padding(innerPadding)
                .padding(16.dp)
                .padding(bottom = 100.dp)
                .verticalScroll(ScrollState(0), true)

        ) {
            Text(
                text = newsItem.title,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = pprimaryBlack
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = newsItem.body,
                fontSize = 16.sp,
                color = secondaryBlackk,
                lineHeight = 24.sp
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "New Stock Prices:",
                fontSize = 16.sp,
                color = secondaryBlackk,
                lineHeight = 24.sp
            )
            if (latestNewsNotRevealed == newsItem.title) {
                Text(
                    text = "The market's about to close! Make your trades before the impact of the news above reaches the market (stock prices are updated). Click the button below when you're done.",
                    fontSize = 16.sp,
                    color = secondaryBlackk,
                    lineHeight = 24.sp
                )
                Button(
                    onClick = {
                        sandbox.allStocks.forEachIndexed { index, element ->
                            for (newPrice in newsItem.newPrices) {
                                if (element.ticker == newPrice.ticker) sandbox.allStocks[index].historicPrice.add(newPrice.newPrice)
                            }
                        }
                        latestNewsNotRevealed = ""
                        sandbox.latestNewsNotRevealed = ""
                        MainActivity.model.saveSandbox(sandbox.id, sandbox)
                    }
                ) {
                    Text("Reveal Prices")
                }
            } else {
                newsItem.newPrices.forEach { newPrice ->
                    Text(
                        text = newPrice.ticker + ": " + newPrice.newPrice.toString() + "( " + (if (newPrice.changeInPrice >= 0) "+" else "") + newPrice.changeInPrice.toString() + ")",
                        fontSize = 16.sp,
                        color = secondaryBlackk,
                        lineHeight = 24.sp
                    )
                }
            }
        }
    }
}



@Composable
fun NewsListItem(newsItem: NewsEvent, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .clickable { onClick() },
        colors = CardDefaults.cardColors(containerColor = llightBlue),
        shape = RoundedCornerShape(15.dp),
        elevation = CardDefaults.cardElevation(2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = newsItem.title,
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp,
                color = pprimaryBlack
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = newsItem.body,
                fontSize = 14.sp,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                color = secondaryBlackk
            )
        }
    }
}

