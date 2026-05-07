package com.example.ultimatestocks.competeAdmin

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.compete.JoinCompetitionPage
import com.example.ultimatestocks.compete.OngoingCompetitionPage
import com.example.ultimatestocks.compete.accentBlue
import com.example.ultimatestocks.compete.deepBlue
import com.example.ultimatestocks.compete.defaultCompetition
import com.example.ultimatestocks.compete.lightBlue
import com.example.ultimatestocks.compete.primaryBlack
import com.example.ultimatestocks.compete.pureWhite
import com.example.ultimatestocks.compete.rememberSlowScrollBehavior
import com.example.ultimatestocks.compete.secondaryBlack

class CompeteAdminHomePage : Screen {
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
                        modifier = Modifier.padding(vertical = 16.dp)
                    )
                }
                item {
                    Text(
                        text = "Compete (Admin)",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = primaryBlack,
                        modifier = Modifier.padding(bottom = 24.dp)
                    )
                    Button(
                        modifier = Modifier.fillMaxWidth().padding(bottom = 10.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                        shape = RoundedCornerShape(15.dp),
                        elevation = ButtonDefaults.buttonElevation(8.dp),
                        onClick = { navigator.push(CreateCompetitionPage()) }
                    ) {
                        Text("Create Competition")
                    }

                }

                item {
                    for (comp in MainActivity.model.activeComps) {
                        val isUserInComp = comp.players.containsKey(userId)

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
                                        text = comp.title,
                                        fontSize = 24.sp,
                                        fontWeight = FontWeight.Bold,
                                        color = primaryBlack
                                    )
                                    Text(
                                        text = "Status: " + (if (comp.hasStarted) "Started" else "Not Started"),
                                        color = secondaryBlack,
                                        modifier = Modifier.padding(vertical = 8.dp)
                                    )
                                    Button(
                                        onClick = { navigator.push(ManageCompetitionPage(comp))},
                                        modifier = Modifier.fillMaxWidth(),
                                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                                        shape = RoundedCornerShape(15.dp),
                                        elevation = ButtonDefaults.buttonElevation(8.dp)
                                    ) {
                                        Text(
                                            text = "Manage Competition",
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

            }
        }
    }
}
