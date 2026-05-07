package com.example.ultimatestocks.competeAdmin

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
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
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.currentOrThrow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.FirebaseStock
import com.example.ultimatestocks.entities.NewsEvent
import com.example.ultimatestocks.entities.StockDetails
import com.example.ultimatestocks.ppureWhite
import com.example.ultimatestocks.secondaryBlackk
import com.example.ultimatestocks.ssoftBlue
import com.example.ultimatestocks.view.ViewModel

class CreateCompetitionPage: Screen {
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow

        var compTitle by remember { mutableStateOf("") }
        var compCash by remember { mutableStateOf(0) }
        var maxNewsEvents by remember { mutableStateOf(0)}
        var selectedStocks by remember { mutableStateOf<List<FirebaseStock>>(listOf()) }

        var expanded by remember { mutableStateOf(false) }

        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("Create New Competition", color = deeepBlue, fontWeight = FontWeight.Bold) },
                    navigationIcon = {
                        IconButton(onClick = { navigator.push(CompeteAdminHomePage()) }) {
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
                    .background(color = ppureWhite)
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(innerPadding)
                    .padding(bottom = 80.dp)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(32.dp))

                Row(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = "New Competition",
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Bold,
                        color = deeepBlue,
                        modifier = Modifier.padding(start = 16.dp)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = compTitle,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number,
                        imeAction = ImeAction.Done
                    ),
                    onValueChange = { input: String ->
                        compTitle = input
                    },
                    label = { Text("Competition Title", color = secondaryBlackk) },
                    shape = RoundedCornerShape(15.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = aaccentBlue,
                        unfocusedBorderColor = ssoftBlue
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp)
                )
                OutlinedTextField(
                    value = compCash.toString(),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number,
                        imeAction = ImeAction.Done
                    ),
                    onValueChange = { input: String ->
                        try {
                            val intInput = input.toInt()
                            compCash = intInput
                        } catch (_: NumberFormatException) {}
                    },
                    label = { Text("Starting Cash", color = secondaryBlackk) },
                    shape = RoundedCornerShape(15.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = aaccentBlue,
                        unfocusedBorderColor = ssoftBlue
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp)
                )

                OutlinedTextField(
                    value = maxNewsEvents.toString(),
                    keyboardOptions = KeyboardOptions(
                        imeAction = ImeAction.Done
                    ),
                    onValueChange = { input: String ->
                        try {
                            val intInput = input.toInt()
                            maxNewsEvents = intInput
                        } catch (_: NumberFormatException) {}
                    },
                    label = { Text("Max News Events", color = secondaryBlackk) },
                    shape = RoundedCornerShape(15.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = aaccentBlue,
                        unfocusedBorderColor = ssoftBlue
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp)
                )

                for (stock in selectedStocks) {
                    Text(stock.ticker + " at " + stock.startingPrice.toString())
                }

                OutlinedButton(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 10.dp),
                    onClick = { expanded = true },
                    colors = ButtonDefaults.outlinedButtonColors(
                        containerColor = ppureWhite,
                        contentColor = aaccentBlue
                    ),
                    border = BorderStroke(1.dp, aaccentBlue),
                    shape = RoundedCornerShape(15.dp)
                ) {
                    Text(
                        "Add Stock",
                        fontWeight = FontWeight.Medium
                    )
                }

                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    for (stock in MainActivity.model.stocks) {
                        DropdownMenuItem(
                            text = { Text(stock.companyName + ": " + stock.description) },
                            onClick = {
                                if (!selectedStocks.contains(stock)) {
                                    selectedStocks = selectedStocks + stock
                                }
                                expanded = false
                            }
                        )
                    }
                }

                Button(

                    onClick = {
                        if (compTitle.isNotEmpty() && compCash > 0 && selectedStocks.isNotEmpty()) {
                            MainActivity.model.createCompetition(
                                Competition(
                                    title = compTitle,
                                    newsEvents = mutableListOf(
                                        NewsEvent("Welcome to the Competition", "This is a sample news event. " +
                                                "News events are simulations of real-life events that may or may not impact the stocks you're trading in this competition, " +
                                                "so read each news event carefully and make the best trades to maximise profits!", listOf())
                                    ),
                                    latestNewsNotRevealed = "",
                                    availableStocks = selectedStocks.map {stock -> StockDetails(stock.ticker, stock.description, mutableListOf(stock.startingPrice.toFloat())) },
                                    players = mutableMapOf(),
                                    initialCash = compCash.toFloat(),
                                    hasStarted = false,
                                    maxNewsEvents = maxNewsEvents,
                                    hasEnded = false
                                )
                            )
                            navigator.push(CompeteAdminHomePage())
                        }
                    },
                    shape = RoundedCornerShape(15.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                    elevation = ButtonDefaults.buttonElevation(8.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                ) {
                    Text(
                        text = "Create New Competition",
                        color = ppureWhite,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(8.dp)
                    )
                }


            }
        }
    }
}
