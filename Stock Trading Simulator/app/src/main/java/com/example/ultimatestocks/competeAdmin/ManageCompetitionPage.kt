package com.example.ultimatestocks.competeAdmin

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.sharedPages.BuyPage
import com.example.ultimatestocks.compete.CompeteHomePage
import com.example.ultimatestocks.compete.CompetitionResultsPage
import com.example.ultimatestocks.compete.JoinCompetitionPage
import com.example.ultimatestocks.sharedPages.NewsListPage
import com.example.ultimatestocks.compete.OngoingCompetitionPage
import com.example.ultimatestocks.sharedPages.SellPage
import com.example.ultimatestocks.compete.accentBlue
import com.example.ultimatestocks.compete.deepBlue
import com.example.ultimatestocks.compete.defaultCompetition
import com.example.ultimatestocks.compete.lightBlue
import com.example.ultimatestocks.compete.primaryBlack
import com.example.ultimatestocks.compete.pureWhite
import com.example.ultimatestocks.compete.rememberSlowScrollBehavior
import com.example.ultimatestocks.compete.secondaryBlack
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.ppureWhite

class ManageCompetitionPage(val comp: Competition) : Screen {
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow
        var latestNewsNotRevealed by remember { mutableStateOf(comp.latestNewsNotRevealed) }
        var compStarted by remember { mutableStateOf(comp.hasStarted) }
        var compEnded by remember { mutableStateOf(comp.hasEnded) }


        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(comp.title, color = deeepBlue, fontWeight = FontWeight.Bold) },
                    navigationIcon = {
                        IconButton(onClick = { navigator.push(CompeteAdminHomePage()) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                                tint = deeepBlue
                            )
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = ppureWhite
                    )
                )
            }
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(pureWhite)
                    .padding(16.dp)
                    .padding(innerPadding)
                    .padding(bottom = 80.dp)
                    .verticalScroll(rememberScrollState()),

                ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Manage Competition (Admin)",
                        fontSize = 14.sp,
                        color = secondaryBlack,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Button(
                        onClick = { navigator.push(NewsListPage(comp)) },
                        colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                        elevation = ButtonDefaults.buttonElevation(8.dp),
                        shape = RoundedCornerShape(15.dp)
                    ) {
                        Text(
                            text = "News",
                            color = ppureWhite,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Text(
                    text = comp.title,
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                    modifier = Modifier.padding(bottom = 24.dp)
                )
                Text(
                    text = "Competition Has " + (if (compStarted) "" else "Not Yet ") + "Started",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                )
                if (!compStarted) {
                    Button(
                        onClick = {
                            comp.hasStarted = true
                            compStarted = true
                            MainActivity.model.saveCompetition(comp)
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                        elevation = ButtonDefaults.buttonElevation(8.dp),
                        shape = RoundedCornerShape(15.dp)
                    ) {
                        Text(
                            text = "Start Competition",
                            color = ppureWhite,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Text(
                    text = "News Events Remaining: " + (comp.maxNewsEvents - comp.newsEvents.size + 1).toString(),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                )

                Text(
                    text = "Latest Price Updates Have" + (if (latestNewsNotRevealed.isEmpty()) "" else " Not") + " Been Revealed",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                )
                if (latestNewsNotRevealed.isNotEmpty()) {
                    Text("Price updates for \"" + comp.latestNewsNotRevealed + "\" have not been revealed")
                }
                if (latestNewsNotRevealed.isNotEmpty()) {
                    Button(
                        onClick = {
                            comp.availableStocks.forEachIndexed { index, element ->
                                for (newPrice in comp.newsEvents.last().newPrices) {
                                    if (element.ticker == newPrice.ticker) comp.availableStocks[index].historicPrice.add(newPrice.newPrice)
                                }
                            }

                            comp.latestNewsNotRevealed = ""
                            latestNewsNotRevealed = ""
                            if (comp.newsEvents.size == comp.maxNewsEvents + 1) {
                                comp.hasEnded = true
                                compEnded = true
                            }
                            MainActivity.model.saveCompetition(comp)
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                        elevation = ButtonDefaults.buttonElevation(8.dp),
                        shape = RoundedCornerShape(15.dp)
                    ) {
                        Text(
                            text = "Reveal Prices",
                            color = ppureWhite,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                Text(
                    text = "No. of Participants: " + comp.players.size.toString(),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                )



                Text(
                    text = "Available Stocks",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                    modifier = Modifier.padding(vertical = 16.dp)
                )


                for (stock in comp.availableStocks) {
                    StockInfoRow(
                        stock.ticker,
                        stock.description,
                        stock.historicPrice.last().toString(),
                    )
                }

//            StockInfoRow("TCK", "Tricake Industries", "23.05", "20")
//            StockInfoRow("ABC", "Abracadabra Co.", "3.05", "120")
//            StockInfoRow("TCK", "Tricake Industries", "23.05", "20")
            }
        }
    }
}


@Composable
private fun StockInfoRow(ticker: String, desc: String, price: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column {
            Text(
                text = ticker,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = primaryBlack
            )
            Text(
                text = desc,
                fontSize = 14.sp,
                modifier = Modifier.width(200.dp),
                color = secondaryBlack
            )
        }
        Column(
            horizontalAlignment = Alignment.End
        ) {
            Text(
                text = "$$price",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = primaryBlack
            )
            Text(
                text = "/share",
                fontSize = 14.sp,
                color = primaryBlack
            )
        }

    }
}