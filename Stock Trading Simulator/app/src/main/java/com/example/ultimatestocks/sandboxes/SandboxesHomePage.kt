package com.example.ultimatestocks.sandboxes

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.currentOrThrow
import cafe.adriel.voyager.navigator.tab.Tab
import cafe.adriel.voyager.navigator.tab.TabOptions
import com.example.ultimatestocks.sharedPages.BuyPage
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.llightBlue
import com.example.ultimatestocks.pprimaryBlack
import com.example.ultimatestocks.ppureWhite
import com.example.ultimatestocks.secondaryBlackk
import com.example.ultimatestocks.view.ViewModel

object LearnTab : Tab {
    private fun readResolve(): Any = LearnTab
    override val options: TabOptions
        @Composable get() {
            val icon = rememberVectorPainter(Icons.Default.Add)
            return remember {
                TabOptions(
                    index = 2u, title = "Learn", icon = icon
                )
            }
        }

    @Composable
    override fun Content() {
        Navigator(SandboxesHomePage())
    }
}

class SandboxesHomePage: Screen {

    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow
        val viewModel = MainActivity.viewModel

        Box(
            modifier = Modifier
                .background(color = ppureWhite)
                .fillMaxSize()
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(bottom = 80.dp)
                    .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(32.dp))

                Row(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "Learn",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = deeepBlue,
                        modifier = Modifier.padding(start = 16.dp)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = { navigator.push(CreateSandboxPage()) },
                    shape = RoundedCornerShape(15.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                    elevation = ButtonDefaults.buttonElevation(8.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                ) {
                    Text(
                        text = "Create New Sandbox",
                        color = ppureWhite,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(8.dp)
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "Saved Sandboxes",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = deeepBlue,
                    modifier = Modifier
                        .align(Alignment.Start)
                        .padding(start = 16.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                for (item in viewModel.sandboxes) {
                    SandboxCard(
                        name = item.name,
                        portfolioValue = item.portfolioVal.last().toString(),
                        newsEventsGenerated = item.newsEvents.count().toString(),
                        onDelete = { viewModel.model.del(item) },
                        onOpen = { navigator.push(SandboxDetailPage(item)) }
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

@Composable
fun SandboxCard(
    name: String,
    portfolioValue: String,
    newsEventsGenerated: String,
    onDelete: () -> Unit,
    onOpen: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = llightBlue),
        elevation = CardDefaults.cardElevation(4.dp),
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp, horizontal = 16.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = name,
                fontWeight = FontWeight.Bold,
                fontSize = 24.sp,
                color = pprimaryBlack
            )

            Text(
                text = "Portfolio value: $$portfolioValue",
                fontSize = 16.sp,
                color = secondaryBlackk
            )

            Text(
                text = "News Events Generated: $newsEventsGenerated",
                fontSize = 16.sp,
                color = secondaryBlackk
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Button(
                    onClick = onOpen,
                    shape = RoundedCornerShape(15.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                    elevation = ButtonDefaults.buttonElevation(8.dp),
                    modifier = Modifier.weight(2f)
                ) {
                    Text(
                        text = "Open Sandbox",
                        color = ppureWhite,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Button(
                    onClick = onDelete,
                    shape = RoundedCornerShape(15.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = deeepBlue),
                    elevation = ButtonDefaults.buttonElevation(8.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = "Delete",
                        color = ppureWhite,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}