package com.example.ultimatestocks.compete

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.entities.Competition

val lightBlue = Color(0xFFE3F2FD)
val accentBlue = Color(0xFF2196F3)
val deepBlue = Color(0xFF1976D2)
private val softBlue = Color(0xFFBBDEFB)
val primaryBlack = Color(0xFF111111)
val secondaryBlack = Color(0xFF2D2D2D)
val pureWhite = Color(0xFFFFFFFF)

class CompetitionResultsPage(val comp: Competition) : Screen {
    val leaderboard = comp.getLeaderboard()
    val userId = MainActivity.model.userUID
    val leaderboardPos = comp.getPositionInLeaderboard(leaderboard, userId) + 1
    @Composable
    override fun Content() {

        var showLeaderboard by remember { mutableStateOf(false) }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(pureWhite)
                .padding(16.dp)
                .padding(bottom = 80.dp)
                .verticalScroll(rememberScrollState()),

        ) {
            Text(
                text = "USX",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = deepBlue,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            Text(
                text = "Competition Ended",
                fontSize = 20.sp,
                color = secondaryBlack,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            Text(
                text = comp.title,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = primaryBlack,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp),
                colors = CardDefaults.cardColors(containerColor = lightBlue),
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "You ranked " + leaderboardPos.toString() + " out of " + comp.players.size.toString() + " players!",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = primaryBlack,
                        modifier = Modifier.padding(bottom = 10.dp)
                    )
                    Text(
                        text = if (leaderboardPos < 3) "That's top 3! Amazing!" else if (leaderboardPos < 10) "That's top 10! Great work!" else "Great Effort!",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = primaryBlack
                    )

                    Button(
                        onClick = {
                            showLeaderboard = !showLeaderboard
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 16.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                        shape = RoundedCornerShape(15.dp),
                        elevation = ButtonDefaults.buttonElevation(8.dp)
                    ) {
                        Text(
                            text = if (showLeaderboard) "Hide Leaderboard" else "Show Leaderboard",
                            color = pureWhite,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(8.dp)
                        )
                    }

//
//                    Text(
//                        text = "You placed in the top [x] percentile, and performed particularly well when it came to tech-related stocks.",
//                        fontSize = 16.sp,
//                        color = secondaryBlack,
//                        modifier = Modifier.padding(vertical = 16.dp)
//                    )
//
//                    Text(
//                        text = "Proficiency Score Increased!",
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.Bold,
//                        fontStyle = FontStyle.Italic,
//                        color = accentBlue
//                    )
                }
            }


            if (showLeaderboard) {
                Card(
                    modifier = Modifier.fillMaxWidth().padding(bottom = 15.dp),
                    colors = CardDefaults.cardColors(containerColor = lightBlue),
                    shape = RoundedCornerShape(20.dp),
                    elevation = CardDefaults.cardElevation(4.dp)

                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "Leaderboard",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = primaryBlack,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                        Text(
                            text = "See how you measured up to other competitors",
                            color = secondaryBlack,
                            modifier = Modifier.padding(bottom = 16.dp)
                        )
                        leaderboard.forEachIndexed { index, player ->
                            LeaderboardItem(
                                name = player.userID,
                                rank = (index + 1).toString(),
                                isUser = player.userID == userId
                            )
                        }

//                    repeat(50) { index ->
//                        LeaderboardItem(
//                            name = when (index) {
//                                0 -> "John"
//                                1 -> "Jane"
//                                2 -> "Jason (You)"
//                                else -> "Player ${index + 1}"
//                            },
//                            rank = "${346 + index}",
//                            isUser = index == 2
//                        )
//                    }


                    }
                }
            }


            Text(
                text = "Final Portfolio",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = primaryBlack,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                        text = "CASH",
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = primaryBlack
                    )
                    Text(
                        text = "Your remaining cash value",
                        fontSize = 14.sp,
                        color = secondaryBlack
                    )
                }
                Column(
                    horizontalAlignment = androidx.compose.ui.Alignment.End
                ) {
                    Text(
                        text = "$${comp.players.get(userId)?.cash?.toInt().toString()}",
                        fontSize = 14.sp,
                        color = primaryBlack
                    )

                }
            }

            for (stock in comp.players.get(userId)!!.portfolio) {
                StockInfoRow(stock.ticker, "", comp.getStockValue(stock.ticker).toString(), stock.quantityOwned.toString())
            }
//            StockInfoRow("ABC", "Abracadabra Co.", "3.05", "120")
//            StockInfoRow("TCK", "Tricake Industries", "23.05", "20")
//            StockInfoRow("TCK", "Tricake Industries", "23.05", "20")
        }
    }
}

@Composable
private fun StockInfoRow(ticker: String, company: String, price: String, shares: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
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
        Column(
            horizontalAlignment = androidx.compose.ui.Alignment.End
        ) {
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
//        verticalAlignment = Alignment.CenterVertically
    ) {
//        Row(
//            verticalAlignment = Alignment.CenterVertically
//        ) {
//            Box(
//                modifier = Modifier
//                    .size(32.dp)
//                    .clip(CircleShape)
//                    .background(accentBlue)
//            )
//            Spacer(modifier = Modifier.width(12.dp))
//            Text(
//                text = name,
//                fontSize = 16.sp,
//                color = primaryBlack
//            )
//        }
        Text(
            text = name.take(15) + (if (name.length > 15) "..." else "") + if (isUser) "(You)" else "",
            fontSize = 16.sp,
            color = primaryBlack
        )
        Text(
            text = "Rank $rank",
            fontSize = 16.sp,
            color = primaryBlack
        )
    }
}