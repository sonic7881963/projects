package com.example.ultimatestocks

import androidx.compose.foundation.ScrollState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.navigator.tab.Tab
import cafe.adriel.voyager.navigator.tab.TabOptions
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import com.example.ultimatestocks.compete.CompeteHomePage
import com.example.ultimatestocks.compete.OngoingCompetitionPage
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults.buttonColors
import androidx.compose.material3.ButtonDefaults.buttonElevation
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults.cardColors
import androidx.compose.material3.CardDefaults.cardElevation
import androidx.compose.material3.Icon
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.currentOrThrow
import cafe.adriel.voyager.navigator.tab.LocalTabNavigator
import com.example.ultimatestocks.sandboxes.LearnTab

object DashboardScreen : Screen {
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow
        val tabNavigator = LocalTabNavigator.current

        Dashboard(
            userName = if (MainActivity.viewModel.userName.value == "") "" else MainActivity.viewModel.userName.value,
            streakDays = 25,
            competitionTime = "3hr 15min",
            prize = "Cookies",
            tabNavigator = tabNavigator
        )
    }
}

@Composable
fun Dashboard(userName: String, streakDays: Int, competitionTime: String, modifier: Modifier = Modifier, prize: String, tabNavigator: cafe.adriel.voyager.navigator.tab.TabNavigator) {
    val lightBlue = Color(0xFFE3F2FD)
    val accentBlue = Color(0xFF2196F3)
    val deepBlue = Color(0xFF1976D2)
    val softBlue = Color(0xFFBBDEFB)
    val primaryBlack = Color(0xFF111111)
    val secondaryBlack = Color(0xFF2D2D2D)
    val pureWhite = Color(0xFFFFFFFF)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(pureWhite)
            .padding(bottom = 100.dp)
            .verticalScroll(ScrollState(0), true),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(modifier = Modifier.weight(1f))

            Card(
                modifier = Modifier
                    .fillMaxWidth(0.95f)
                    .height(180.dp),
                colors = cardColors(containerColor = lightBlue),
                shape = RoundedCornerShape(24.dp),
                elevation = cardElevation(4.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(24.dp)
                ) {
                    Column {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "USX",
                                fontSize = 40.sp,
                                fontWeight = FontWeight.ExtraBold,
                                color = deepBlue
                            )
                            Image(
                                painter = painterResource(id = R.drawable.ct2),
                                contentDescription = "Logo",
                                modifier = Modifier.size(48.dp),
                                contentScale = ContentScale.Fit
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Welcome back, $userName",
                            fontSize = 28.sp,
                            fontWeight = FontWeight.Bold,
                            color = primaryBlack
                        )
                    }
                }
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth(0.95f),
                colors = cardColors(containerColor = accentBlue),
                shape = RoundedCornerShape(24.dp),
                elevation = cardElevation(6.dp)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (MainActivity.viewModel.sandboxes.isEmpty()) "Start Learning" else "Continue Learning",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = pureWhite
                        )
                        Icon(
                            imageVector = Icons.Default.Star,
                            contentDescription = "Learn",
                            tint = pureWhite,
                            modifier = Modifier.size(32.dp)
                        )
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = if (MainActivity.viewModel.sandboxes.isEmpty())
                            "Create your first sandbox to start learning!"
                        else
                            "You have ${MainActivity.viewModel.sandboxes.size} saved sandboxes",
                        fontSize = 16.sp,
                        color = pureWhite.copy(alpha = 0.9f)
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Button(
                        onClick = { tabNavigator.current = LearnTab },
                        colors = buttonColors(containerColor = pureWhite),
                        elevation = buttonElevation(8.dp),
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            if (MainActivity.viewModel.sandboxes.isEmpty()) "Create a Sandbox" else "Go to Your Sandbox",
                            color = accentBlue,
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    }
                }
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth(0.95f),
                colors = cardColors(containerColor = deepBlue),
                shape = RoundedCornerShape(24.dp),
                elevation = cardElevation(6.dp)
            ) {
                Column(
                    modifier = Modifier.padding(24.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (!MainActivity.model.activeComps.isEmpty()) "Active Competition" else "Competitions",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = pureWhite
                        )
                        Icon(
                            imageVector = Icons.Default.Star,
                            contentDescription = "Competition",
                            tint = pureWhite,
                            modifier = Modifier.size(32.dp)
                        )
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = if (!MainActivity.model.activeComps.isEmpty())
                            "You have active competitions to participate in!"
                        else
                            "Check out available competitions",
                        fontSize = 16.sp,
                        color = pureWhite.copy(alpha = 0.9f)
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Button(
                        onClick = { tabNavigator.current = CompeteTab },
                        colors = buttonColors(containerColor = pureWhite),
                        elevation = buttonElevation(8.dp),
                        shape = RoundedCornerShape(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            if (!MainActivity.model.activeComps.isEmpty()) "Active Competition" else "View Competitions",
                            color = deepBlue,
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.weight(1f))
        }
    }
}