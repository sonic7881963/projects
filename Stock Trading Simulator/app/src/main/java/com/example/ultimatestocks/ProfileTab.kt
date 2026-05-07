package com.example.ultimatestocks

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.currentOrThrow
import cafe.adriel.voyager.navigator.tab.LocalTabNavigator
import cafe.adriel.voyager.navigator.tab.Tab
import cafe.adriel.voyager.navigator.tab.TabOptions
import com.example.ultimatestocks.view.ViewModel

val lightBlue = Color(0xFFE3F2FD)
val accentBlue = Color(0xFF2196F3)
val deepBlue = Color(0xFF1976D2)
val softBlue = Color(0xFFBBDEFB)
val primaryBlack = Color(0xFF111111)
val secondaryBlack = Color(0xFF2D2D2D)
val pureWhite = Color(0xFFFFFFFF)

object ProfileTab : Tab {
    override val options: TabOptions
        @Composable get() {
            val icon = rememberVectorPainter(Icons.Default.Person)
            return remember {
                TabOptions(
                    index = 3u, title = "Profile", icon = icon
                )
            }
        }

    @Composable
    override fun Content() {
        Navigator(ProfileScreen())
    }
}

class ProfileScreen : Screen {
    @Composable
    override fun Content() {
        Profile(MainActivity.viewModel)
    }
}

enum class Page {
    Profile, AccountInfo
}

@Composable
fun Profile(vm: ViewModel, modifier: Modifier = Modifier) {
    val viewModel by remember { mutableStateOf(vm) }
    val navigator = LocalNavigator.currentOrThrow
    val tabNavigator = LocalTabNavigator
    var currentPage by remember { mutableStateOf(Page.Profile) }

    when (currentPage) {
        Page.Profile -> {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(pureWhite)
                    .verticalScroll(rememberScrollState())
            ) {
                // Profile Header
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .background(deepBlue)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(80.dp)
                                .clip(CircleShape)
                                .background(accentBlue),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = "Profile",
                                tint = pureWhite,
                                modifier = Modifier.size(40.dp)
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = viewModel.userName.value,
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = pureWhite
                        )
                        Text(
                            text = viewModel.userEmail.value,
                            fontSize = 16.sp,
                            color = pureWhite.copy(alpha = 0.8f)
                        )
                    }
                }

                // Quick Stats
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    StatCard(value = "15m", label = "Today")
                    StatCard(value = "4", label = "Wins")
                }

                // Settings Cards
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                ) {
                    SettingsCard(
                        title = "Account Information",
                        onClick = { currentPage = Page.AccountInfo }
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Logout Button
                Button(
                    onClick = {
                        viewModel.model.attemptLogout()
                        navigator.parent?.popUntilRoot()
                        navigator.parent?.parent?.popUntilRoot()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = deepBlue),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp)
                ) {
                    Text(
                        "Log Out",
                        color = pureWhite,
                        fontWeight = FontWeight.Bold
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))
            }
        }
        Page.AccountInfo -> {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(pureWhite)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    // Add extra space at the top
                    Spacer(modifier = Modifier.height(100.dp))

                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = lightBlue),
                        elevation = CardDefaults.cardElevation(4.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp)
                        ) {
                            Text(
                                text = "Account Details",
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Bold,
                                color = deepBlue
                            )
                            Spacer(modifier = Modifier.height(24.dp))
                            InfoRow("User ID", viewModel.userUID.value)
                            InfoRow("Username", viewModel.userName.value)
                            InfoRow("Email", viewModel.userEmail.value)
                            InfoRow("Status", if (MainActivity.model.isAdmin) "Administrator" else "User")
                        }
                    }

                    Spacer(modifier = Modifier.height(24.dp))

                    Button(
                        onClick = { currentPage = Page.Profile },
                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            "Back to Profile",
                            color = pureWhite,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StatCard(
    value: String,
    label: String
) {
    Card(
        modifier = Modifier.width(100.dp),
        colors = CardDefaults.cardColors(containerColor = lightBlue),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = primaryBlack
            )
            Text(
                text = label,
                fontSize = 14.sp,
                color = secondaryBlack
            )
        }
    }
}

@Composable
private fun SettingsCard(
    title: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = lightBlue),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = null,
                tint = deepBlue,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Text(
                text = title,
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium,
                color = primaryBlack,
                modifier = Modifier.weight(1f)
            )
            Icon(
                imageVector = Icons.Default.ArrowBack,
                contentDescription = null,
                tint = accentBlue
            )
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Column(modifier = Modifier.padding(vertical = 8.dp)) {
        Text(
            text = label,
            fontSize = 14.sp,
            color = secondaryBlack
        )
        Text(
            text = value,
            fontSize = 16.sp,
            fontWeight = FontWeight.Medium,
            color = primaryBlack
        )
    }
}