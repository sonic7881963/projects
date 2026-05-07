package com.example.ultimatestocks.sharedPages

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import com.example.ultimatestocks.FinancialInfoRow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.llightBlue
import com.example.ultimatestocks.ppureWhite
import com.example.ultimatestocks.secondaryBlackk
import com.example.ultimatestocks.ssoftBlue

// Sell Page for Competition Flow
class SellPage(val comp: Competition, val onBack: () -> Unit) : Screen {

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        val userId = MainActivity.model.userUID
        val playerInfo = comp.players.getValue(userId)
        var tickerSymbol by remember { mutableStateOf("") }
        var numberOfShares by remember { mutableStateOf<Int>(0) }
        var valuePerShare by remember { mutableFloatStateOf(0.0f) }
        var sharesAvailable by remember { mutableStateOf(0) }

        val availableCash = playerInfo.cash
        val totalRevenue:Float = numberOfShares * valuePerShare
        val resultingCash = availableCash + totalRevenue


        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("Sell", color = deeepBlue, fontWeight = FontWeight.Bold) },
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
            ) {
                Text(
                    text = "Buy",
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    color = deeepBlue,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                var expanded by remember { mutableStateOf(false) }
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
                        if (tickerSymbol == "") "Select Stock"
                        else "Selected Stock: $tickerSymbol",
                        fontWeight = FontWeight.Medium
                    )
                }

                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    for (stock in playerInfo.portfolio) {
                        DropdownMenuItem(
                            text = { Text(stock.ticker) },
                            onClick = {
                                tickerSymbol = stock.ticker
                                expanded = false
                                valuePerShare = comp.getStockValue(stock.ticker)
                                sharesAvailable = stock.quantityOwned
                            }
                        )
                    }
                }

                OutlinedTextField(
                    value = numberOfShares.toString(),
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number,
                        imeAction = ImeAction.Done
                    ),
                    onValueChange = { input: String ->
                        try {
                            val intInput = input.toInt()
                            numberOfShares = intInput
                        } catch (_: NumberFormatException) {}
                    },
                    label = { Text("No. of Shares (${sharesAvailable} shares available)", color = secondaryBlackk) },
                    shape = RoundedCornerShape(15.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = aaccentBlue,
                        unfocusedBorderColor = ssoftBlue
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp)
                )

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp),
                    colors = CardDefaults.cardColors(containerColor = llightBlue),
                    shape = RoundedCornerShape(20.dp),
                    elevation = CardDefaults.cardElevation(4.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "USD ($)",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = deeepBlue,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )

                        FinancialInfoRow("Value per share", "$$valuePerShare")
                        FinancialInfoRow("Available Cash", "$$availableCash")
                        FinancialInfoRow("Total Revenue", "$$totalRevenue")
                        HorizontalDivider(
                            modifier = Modifier.padding(vertical = 8.dp),
                            color = secondaryBlackk.copy(alpha = 0.2f)
                        )
                        FinancialInfoRow("Resulting Cash", "$$resultingCash")
                    }
                }

//            Spacer(modifier = Modifier.weight(1f))

                Button(
                    onClick = {
                        if (numberOfShares > 0 && numberOfShares <= sharesAvailable && tickerSymbol !== "") {
                            comp.sell(tickerSymbol, numberOfShares, userId)
                            MainActivity.model.saveCompetition(comp)
                            onBack()
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
                        text = "Confirm Sale",
                        color = ppureWhite,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                }
            }
        }
    }
}

// Sell Page for Sandboxes Flow
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SellPage(sandbox: Sandbox, onBack: () -> Unit) {
    var tickerSymbol by remember { mutableStateOf("") }
    var numberOfShares by remember { mutableStateOf<Int>(0) }
    var valuePerShare by remember { mutableFloatStateOf(0.0f) }
    var sharesAvailable by remember { mutableStateOf(0) }

    val availableCash = sandbox.cash
    val totalRevenue:Float = numberOfShares * valuePerShare
    val resultingCash = availableCash + totalRevenue

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Sell", color = deeepBlue, fontWeight = FontWeight.Bold) },
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
        ) {
            Text(
                text = "Sell",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = deeepBlue,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            var expanded by remember { mutableStateOf(false) }
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
                    if (tickerSymbol == "") "Select Stock"
                    else "Selected Stock: $tickerSymbol",
                    fontWeight = FontWeight.Medium
                )
            }

            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                for (stock in sandbox.ownedStocks) {
                    DropdownMenuItem(
                        text = { Text(stock.ticker) },
                        onClick = {
                            tickerSymbol = stock.ticker
                            expanded = false
                            valuePerShare = sandbox.getStockValue(stock.ticker)
                            sharesAvailable = stock.quantityOwned
                        }
                    )
                }
            }

            OutlinedTextField(
                value = numberOfShares.toString(),
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Number,
                    imeAction = ImeAction.Done
                ),
                onValueChange = { input: String ->
                    try {
                        val intInput = input.toInt()
                        numberOfShares = intInput
                    } catch (_: NumberFormatException) {}
                },
                label = { Text("No. of Shares (${sharesAvailable} shares available)", color = secondaryBlackk) },
                shape = RoundedCornerShape(15.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = aaccentBlue,
                    unfocusedBorderColor = ssoftBlue
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp)
            )

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp),
                colors = CardDefaults.cardColors(containerColor = llightBlue),
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(4.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "USD ($)",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = deeepBlue,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )

                    FinancialInfoRow("Value per share", "$$valuePerShare")
                    FinancialInfoRow("Available Cash", "$$availableCash")
                    FinancialInfoRow("Total Revenue", "$$totalRevenue")
                    Divider(
                        color = secondaryBlackk.copy(alpha = 0.2f),
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                    FinancialInfoRow("Resulting Cash", "$$resultingCash")
                }
            }

//            Spacer(modifier = Modifier.weight(1f))

            Button(
                onClick = {
                    if (numberOfShares > 0 && numberOfShares <= sharesAvailable && tickerSymbol !== "") {
                        sandbox.sell(tickerSymbol, numberOfShares)
                        MainActivity.model.saveSandbox(sandbox.id, sandbox)
                        onBack()
                    }
                },
                shape = RoundedCornerShape(15.dp),
                colors = ButtonDefaults.buttonColors(containerColor = deeepBlue),
                elevation = ButtonDefaults.buttonElevation(8.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            ) {
                Text(
                    text = "Confirm Sale",
                    color = ppureWhite,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
        }
    }
}