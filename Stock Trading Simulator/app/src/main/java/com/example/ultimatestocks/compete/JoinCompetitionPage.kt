package com.example.ultimatestocks.compete

import android.view.Gravity
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.FlingBehavior
import androidx.compose.foundation.gestures.ScrollScope
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.annotation.ExperimentalVoyagerApi
import cafe.adriel.voyager.core.lifecycle.LifecycleEffectOnce
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.Player

private val softBlue = Color(0xFFBBDEFB)

class CustomScrollBehavior(
    private val slowdownFactor: Float = 0.5f
) : FlingBehavior {
    override suspend fun ScrollScope.performFling(initialVelocity: Float): Float {
        val adjustedVelocity = initialVelocity * slowdownFactor
        return adjustedVelocity
    }
}

@Composable
fun rememberCustomScrollBehavior(slowdownFactor: Float = 0.5f): FlingBehavior {
    return remember {
        CustomScrollBehavior(slowdownFactor)
    }
}

class JoinCompetitionPage(val comp: Competition) : Screen {
    @OptIn(ExperimentalVoyagerApi::class)
    @Composable
    override fun Content() {
//        LifecycleEffectOnce {
//            MainActivity.model.getCompetitions()
//        }
        val viewModel by remember { mutableStateOf(MainActivity.viewModel) }
        val navigator = LocalNavigator.currentOrThrow
        val userId = MainActivity.model.userUID

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(pureWhite)
        ) {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 100.dp),
                contentPadding = PaddingValues(vertical = 16.dp),
                flingBehavior = rememberCustomScrollBehavior(slowdownFactor = 0.5f)
            ) {
                item {
                    Text(
                        text = "USX",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = deepBlue,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )
                }

                item {
                    Text(
                        text = "New Competition",
                        fontSize = 20.sp,
                        color = secondaryBlack,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                }

                item {
                    Text(
                        text = comp.title,
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = primaryBlack,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )
                }


                item {
                    Text(
                        text = "Registration is open!",
                        fontSize = 16.sp,
                        fontStyle = FontStyle.Italic,
                        color = primaryBlack,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )
                }

                item {
                    Button(
                        onClick = {
                            comp.players[userId] = Player(
                                userId,
                                cash = comp.initialCash,
                                mutableListOf()
                            )
                            MainActivity.model.saveCompetition(comp)
                            navigator.push(OngoingCompetitionPage(comp))
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                        shape = RoundedCornerShape(15.dp),
                        elevation = ButtonDefaults.buttonElevation(8.dp)
                    ) {
                        Text(
                            text = "Join Competition",
                            color = pureWhite,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(8.dp)
                        )
                    }
                }

                item {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 24.dp),
                        colors = CardDefaults.cardColors(containerColor = lightBlue),
                        shape = RoundedCornerShape(20.dp),
                        elevation = CardDefaults.cardElevation(4.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text(
                                text = "Competition Info",
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = primaryBlack,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                            CompetitionInfoRow("News Events:", comp.maxNewsEvents.toString())
                            CompetitionInfoRow("Participants:", comp.players.count().toString())
                        }
                    }
                }

                item {
                    Text(
                        text = "Available Stocks",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = primaryBlack,
                        modifier = Modifier.padding(top = 24.dp, bottom = 16.dp)
                    )
                    for (stock in comp.availableStocks) {
                        Text(stock.ticker + ": " + stock.description + ", starting at $" + stock.historicPrice.last().toString() + "/share")
                    }
                }


            }
        }
    }
}

@Composable
private fun CompetitionInfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Text(
            text = label,
            color = secondaryBlack,
            fontSize = 16.sp
        )
        Text(
            text = " $value",
            color = primaryBlack,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun StockInfoRow(ticker: String, company: String, price: String, shares: String) {
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
                text = company,
                fontSize = 14.sp,
                color = secondaryBlack
            )
        }
        Column {
            Text(
                text = "Share Price: $$price",
                fontSize = 14.sp,
                color = primaryBlack
            )
            Text(
                text = "No. of Shares: $shares",
                fontSize = 14.sp,
                color = primaryBlack
            )
        }
    }
}