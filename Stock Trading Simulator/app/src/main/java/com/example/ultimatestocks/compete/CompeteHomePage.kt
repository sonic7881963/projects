package com.example.ultimatestocks.compete

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.FlingBehavior
import androidx.compose.foundation.gestures.ScrollScope
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.StockDetails

private val softBlue = Color(0xFFBBDEFB)

class SlowScrollBehavior(
    private val slowdownFactor: Float = 0.5f
) : FlingBehavior {
    override suspend fun ScrollScope.performFling(initialVelocity: Float): Float {
        val adjustedVelocity = initialVelocity * slowdownFactor
        return adjustedVelocity
    }
}

@Composable
fun rememberSlowScrollBehavior(slowdownFactor: Float = 0.5f): FlingBehavior {
    return remember {
        SlowScrollBehavior(slowdownFactor)
    }
}
val dummyStocks = listOf( // this dummy data is generated using ChatGPT
    StockDetails("AAPL", "Apple Inc. designs and manufactures consumer electronics and software.", mutableListOf(125.50f, 127.30f, 126.40f, 128.90f, 127.80f)),
    StockDetails("GOOGL", "Alphabet Inc. is the parent company of Google, specializing in internet services and products.", mutableListOf(2345.70f, 2360.80f, 2350.20f, 2372.10f, 2380.60f)),
    StockDetails("MSFT", "Microsoft Corporation develops software, services, and solutions for businesses and individuals.", mutableListOf(275.40f, 276.80f, 274.90f, 278.30f, 277.00f)),
    StockDetails("AMZN", "Amazon.com, Inc. is an e-commerce and cloud computing giant.", mutableListOf(3350.10f, 3360.40f, 3345.90f, 3380.20f, 3375.10f)),
    StockDetails("META", "Meta Platforms, Inc. focuses on social media and virtual reality.", mutableListOf(300.20f, 302.40f, 301.30f, 303.50f, 302.80f)),
    StockDetails("TSLA", "Tesla, Inc. designs and manufactures electric vehicles and energy storage solutions.", mutableListOf(680.30f, 685.20f, 678.40f, 690.10f, 687.30f)),
    StockDetails("NFLX", "Netflix, Inc. is a streaming entertainment service provider.", mutableListOf(540.20f, 542.50f, 538.90f, 544.60f, 543.30f)),
    StockDetails("NVDA", "NVIDIA Corporation designs graphics processing units and AI solutions.", mutableListOf(750.40f, 752.80f, 748.20f, 755.30f, 753.90f)),
    StockDetails("ADBE", "Adobe Inc. provides software solutions for creatives and businesses.", mutableListOf(560.30f, 562.70f, 559.20f, 563.80f, 561.50f)),
    StockDetails("INTC", "Intel Corporation designs and manufactures microprocessors and other semiconductor products.", mutableListOf(52.20f, 51.80f, 52.70f, 53.10f, 52.50f))
)
val defaultCompetition = Competition(
    title = "Default Competition",
    availableStocks = dummyStocks,
    initialCash = 10000.0f,
    players = mutableMapOf(),
    maxNewsEvents = 10,
    newsEvents = mutableListOf(),
    hasStarted = false,
    hasEnded = false,
    latestNewsNotRevealed = "",

)

class CompeteHomePage : Screen {
    @Composable
    override fun Content() {
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
                flingBehavior = rememberSlowScrollBehavior(slowdownFactor = 0.5f)
            ) {
                item {
                    Text(
                        text = "USX",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = deepBlue,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    Text(
                        text = "Compete",
                        fontSize = 40.sp,
                        fontWeight = FontWeight.Bold,
                        color = primaryBlack,
                        modifier = Modifier.padding(bottom = 24.dp)
                    )

                }

                item {
                    for (comp in MainActivity.model.activeComps) {
                        val isUserInComp = comp.players.containsKey(userId)
                        if (!comp.hasStarted || comp.players.containsKey(MainActivity.model.userUID)) {
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(bottom = 16.dp),
                                colors = CardDefaults.cardColors(containerColor = lightBlue),
                                shape = RoundedCornerShape(20.dp),
                                elevation = CardDefaults.cardElevation(4.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp)
                                ) {
                                    Text(
                                        text = (if (isUserInComp) "" else "New competition: ") + comp.title,
                                        fontSize = 24.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = primaryBlack
                                    )
                                    Text(
                                        text = if (isUserInComp) "You've already joined this competition. Continue trading now!" else "A new competition is about to start, join now to secure your spot!",
                                        color = secondaryBlack,
                                        modifier = Modifier.padding(vertical = 8.dp)
                                    )
                                    Button(
                                        onClick = {
                                            if (!isUserInComp) navigator.push(JoinCompetitionPage(comp)) else if (comp.hasEnded) navigator.push(CompetitionResultsPage(comp)) else navigator.push(OngoingCompetitionPage(comp))
//                                            if (isUserInComp) navigator.push(OngoingCompetitionPage(comp)) else navigator.push(JoinCompetitionPage(comp))
                                                  },
                                        modifier = Modifier.fillMaxWidth(),
                                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                                        shape = RoundedCornerShape(15.dp),
                                        elevation = ButtonDefaults.buttonElevation(8.dp)
                                    ) {
                                        Text(
                                            text = if (!isUserInComp) "Join Competition" else if (comp.hasEnded) "See Results" else "Continue Competition",
                                            color = pureWhite,
                                            fontSize = 16.sp,
                                            fontWeight = FontWeight.Bold,
                                            modifier = Modifier.padding(8.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }

                    if (MainActivity.model.activeComps.isEmpty())
                        Text("No active Competitions at the moment. Check back later!")

                }


            }
        }
    }
}

@Composable
private fun LeaderboardItem(name: String, rank: String, isUser: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .background(
                color = if (isUser) softBlue else Color.Transparent,
                shape = RoundedCornerShape(10.dp)
            )
            .padding(8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(CircleShape)
                    .background(accentBlue)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = name,
                fontSize = 16.sp,
                color = primaryBlack
            )
        }
        Text(
            text = "Rank $rank",
            fontSize = 16.sp,
            color = primaryBlack
        )
    }
}