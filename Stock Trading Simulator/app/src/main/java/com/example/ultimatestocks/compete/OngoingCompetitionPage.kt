package com.example.ultimatestocks.compete

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.currentOrThrow
import co.yml.charts.axis.AxisData
import co.yml.charts.common.model.Point
import co.yml.charts.ui.linechart.LineChart
import co.yml.charts.ui.linechart.model.GridLines
import co.yml.charts.ui.linechart.model.IntersectionPoint
import co.yml.charts.ui.linechart.model.Line
import co.yml.charts.ui.linechart.model.LineChartData
import co.yml.charts.ui.linechart.model.LinePlotData
import co.yml.charts.ui.linechart.model.LineStyle
import co.yml.charts.ui.linechart.model.SelectionHighlightPoint
import co.yml.charts.ui.linechart.model.SelectionHighlightPopUp
import co.yml.charts.ui.linechart.model.ShadowUnderLine
import com.example.ultimatestocks.FinancialInfoRow
import com.example.ultimatestocks.MainActivity
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.competeAdmin.ManageCompetitionPage
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.NewsEvent
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.llightBlue
import com.example.ultimatestocks.pprimaryBlack
import com.example.ultimatestocks.ppureWhite
import com.example.ultimatestocks.secondaryBlackk
import com.example.ultimatestocks.sharedPages.BuyPage
import com.example.ultimatestocks.sharedPages.NewsListPage
import com.example.ultimatestocks.sharedPages.SellPage


private val softBlue = Color(0xFFBBDEFB)

//fun nav(state: String, nav: Navigator) {
//    if (state == "Buy" || state == "Sell") {
//        MainActivity.model.setBuySell(state, "Competition")
//        nav.push(BuySellScreen)
//    } else if(state == "News") {
//        nav.push(NewsListScreen)
//    }
//}

class OngoingCompetitionPage(var competition: Competition) : Screen {
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow
        var comp by remember { mutableStateOf(competition) }

        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(comp.title, color = deeepBlue, fontWeight = FontWeight.Bold) },
                    navigationIcon = {
                        IconButton(onClick = { navigator.push(CompeteHomePage()) }) {
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
                    .padding(bottom = 100.dp)
                    .verticalScroll(rememberScrollState()),

                ) {
                Text(
                    text = "Ongoing Competition",
                    fontSize = 20.sp,
                    color = secondaryBlack,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {

                    Button(
                        onClick = { MainActivity.model.refreshCompetition(comp) {newComp -> comp = newComp} },
                        colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                        elevation = ButtonDefaults.buttonElevation(8.dp),
                        shape = RoundedCornerShape(15.dp)
                    ) {
                        Icon(imageVector = Icons.Default.Refresh, modifier = Modifier, contentDescription = "icon to refresh competition state")
//                        Text(
//                            text = "Refresh",
//                            color = ppureWhite,
//                            fontWeight = FontWeight.Bold
//                        )
                    }
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

                if (!comp.hasStarted) {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(60.dp)
                            .padding(bottom = 10.dp),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFFEB4D42)),
                        shape = RoundedCornerShape(10.dp),
                        elevation = CardDefaults.cardElevation(4.dp)
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "This competition has not started yet.",
                                color = ppureWhite,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    }
                }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp),
                    colors = CardDefaults.cardColors(containerColor = llightBlue),
                    shape = RoundedCornerShape(20.dp),
                    elevation = CardDefaults.cardElevation(4.dp)
                ) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        val allStocks = comp.availableStocks

                        // Calculate the maximum number of data points across all stocks
                        val maxDataPoints = allStocks.maxOfOrNull { it.historicPrice.size } ?: 0

                        // Prepare line data with distinct colors for each stock
                        val lines = allStocks.mapIndexed { index, stock ->
                            Line(
                                dataPoints = stock.historicPrice.mapIndexed { priceIndex, price ->
                                    Point(priceIndex.toFloat(), price)
                                },
                                lineStyle = LineStyle(
                                    color = Color.hsv(
                                        hue = (index * 360f / allStocks.size) % 360f,
                                        saturation = 0.8f,
                                        value = 1f
                                    ),
                                    width = 2f
                                ),
                                intersectionPoint = IntersectionPoint(
                                    color = Color.White,
                                    radius = 4.dp
                                ),
                                selectionHighlightPoint = SelectionHighlightPoint(
                                    color = Color.Red,
                                    radius = 6.dp
                                ),
                                shadowUnderLine = ShadowUnderLine(
                                    color = Color.hsv(
                                        hue = (index * 360f / allStocks.size) % 360f,
                                        saturation = 0.8f,
                                        value = 0.5f
                                    ),
                                    alpha = 0.2f
                                ),
                                selectionHighlightPopUp = SelectionHighlightPopUp(
                                    backgroundColor = Color.White,
                                    labelColor = Color.Black,
                                    labelSize = 12.sp
                                )
                            )
                        }

                        // Calculate Y-axis range based on the data
                        val allPrices = allStocks.flatMap { it.historicPrice }
                        val yMin = allPrices.minOrNull() ?: 0f
                        val yMax = allPrices.maxOrNull() ?: 100f
                        val ySteps = 10
                        val yStepValue = (yMax - yMin) / ySteps

                        val desiredLabelCount = 6
                        val labelStep = if (maxDataPoints / desiredLabelCount > 0) maxDataPoints / desiredLabelCount else 1

                        val xAxisData = AxisData.Builder()
                            .axisStepSize(20.dp)
                            .steps(maxDataPoints - 1)
                            .labelData { i ->
                                if (i % labelStep == 0 || i == maxDataPoints - 1) {
                                    "Day ${i + 1}"
                                } else {
                                    ""
                                }
                            }
                            .labelAndAxisLinePadding(16.dp)
                            .axisLabelAngle(-45f) // Rotated labels for better readability
                            .axisLabelFontSize(10.sp) // Reduced font size to prevent overlap
                            .axisLineColor(Color.Gray)
                            .axisLabelColor(Color.Black)
                            .build()

                        val yAxisData = AxisData.Builder()
                            .steps(ySteps)
                            .labelData { i ->
                                val value = yMin + i * yStepValue
                                String.format("%.2f", value)
                            }
                            .labelAndAxisLinePadding(8.dp)
                            .axisLabelFontSize(12.sp) // Consistent font size with X-axis
                            .axisLineColor(Color.Gray)
                            .axisLabelColor(Color.Black)
                            .build()

                        val lineChartData = LineChartData(
                            linePlotData = LinePlotData(lines = lines),
                            xAxisData = xAxisData,
                            yAxisData = yAxisData,
                            gridLines = GridLines(
                                color = Color.LightGray,
                                lineWidth = 1.dp
                            ),
                            backgroundColor = Color.White
                        )

                        LineChart(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(300.dp),
                            lineChartData = lineChartData
                        )
                    }

                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Button(
                        onClick = {
                            navigator.push(BuyPage(comp) {
                                navigator.push(
                                    OngoingCompetitionPage(comp)
                                )
                            })
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                        shape = RoundedCornerShape(15.dp),
                        enabled = comp.hasStarted,
                        elevation = ButtonDefaults.buttonElevation(8.dp)
                    ) {
                        Text(
                            text = "buy",
                            color = pureWhite,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    }

                    Button(
                        onClick = {
                            navigator.push(SellPage(comp) {
                                navigator.push(
                                    OngoingCompetitionPage(comp)
                                )
                            })
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = accentBlue),
                        shape = RoundedCornerShape(15.dp),
                        elevation = ButtonDefaults.buttonElevation(8.dp),
                        enabled = comp.hasStarted,
                    ) {
                        Text(
                            text = "sell",
                            color = pureWhite,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    }
                }

//                Button(
//                    onClick = { navigator.push(CompetitionResultsPage()) },
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(bottom = 16.dp),
//                    colors = ButtonDefaults.buttonColors(containerColor = deepBlue),
//                    shape = RoundedCornerShape(15.dp),
//                    elevation = ButtonDefaults.buttonElevation(8.dp)
//                ) {
//                    Text(
//                        text = "View Results",
//                        color = pureWhite,
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.Bold,
//                        modifier = Modifier.padding(vertical = 8.dp)
//                    )
//                }

                Text(
                    text = "Current Portfolio",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = primaryBlack,
                    modifier = Modifier.padding(vertical = 16.dp)
                )
                if (!comp.players[MainActivity.model.userUID]?.portfolio.isNullOrEmpty()) {

                    for (stock in comp.players.getValue(MainActivity.model.userUID).portfolio) {
                        StockInfoRow(
                            stock.ticker,
                            "",
                            comp.availableStocks.find { it.ticker == stock.ticker }?.historicPrice?.last()
                                .toString(),
                            stock.quantityOwned.toString()
                        )
                    }
                }
//            StockInfoRow("TCK", "Tricake Industries", "23.05", "20")
//            StockInfoRow("ABC", "Abracadabra Co.", "3.05", "120")
//            StockInfoRow("TCK", "Tricake Industries", "23.05", "20")
            }
        }
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
        Column(
            horizontalAlignment = Alignment.End
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

//class BuyPage(val comp: Competition,val onBack: () -> Unit) : Screen {
//
//    @OptIn(ExperimentalMaterial3Api::class)
//    @Composable
//    override fun Content() {
//        val userId = MainActivity.model.userUID
//        val playerInfo = comp.players.getValue(userId)
//        var tickerSymbol by remember { mutableStateOf("") }
//        var numberOfShares by remember { mutableStateOf<Int>(0) }
//        var valuePerShare by remember { mutableFloatStateOf(0.0f) }
//
//
//        val availableCash = playerInfo.cash
//        val totalCost:Float = numberOfShares * valuePerShare
//        val resultingCash = availableCash - totalCost
//
//        Scaffold(
//            topBar = {
//                TopAppBar(
//                    title = { Text("Buy", color = deeepBlue, fontWeight = FontWeight.Bold) },
//                    navigationIcon = {
//                        IconButton(onClick = onBack) {
//                            Icon(
//                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
//                                contentDescription = "Back",
//                                tint = deeepBlue
//                            )
//                        }
//                    },
//                    colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
//                )
//            }
//        ) { innerPadding ->
//            Column(
//                modifier = Modifier
//                    .fillMaxSize()
//                    .background(ppureWhite)
//                    .padding(innerPadding)
//                    .padding(16.dp)
//            ) {
//                Text(
//                    text = "Buy",
//                    fontSize = 32.sp,
//                    fontWeight = FontWeight.Bold,
//                    color = deeepBlue,
//                    modifier = Modifier.padding(bottom = 16.dp)
//                )
//
//                var expanded by remember { mutableStateOf(false) }
//                OutlinedButton(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(bottom = 10.dp),
//                    onClick = { expanded = true },
//                    colors = ButtonDefaults.outlinedButtonColors(
//                        containerColor = ppureWhite,
//                        contentColor = aaccentBlue
//                    ),
//                    border = BorderStroke(1.dp, aaccentBlue),
//                    shape = RoundedCornerShape(15.dp)
//                ) {
//                    Text(
//                        if (tickerSymbol == "") "Select Stock"
//                        else "Selected Stock: $tickerSymbol",
//                        fontWeight = FontWeight.Medium
//                    )
//                }
//
//                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
//                    for (stock in comp.availableStocks) {
//                        DropdownMenuItem(
//                            text = { Text(stock.ticker) },
//                            onClick = {
//                                tickerSymbol = stock.ticker
//                                expanded = false
//                                valuePerShare = stock.historicPrice.last()
//                            }
//                        )
//                    }
//                }
//
//                OutlinedTextField(
//                    value = numberOfShares.toString(),
//                    keyboardOptions = KeyboardOptions(
//                        keyboardType = KeyboardType.Number,
//                        imeAction = ImeAction.Done
//                    ),
//                    onValueChange = { input: String ->
//                        numberOfShares = if (input == "") 0 else input.toInt()
//                    },
//                    label = { Text("No. of Shares", color = secondaryBlackk) },
//                    shape = RoundedCornerShape(15.dp),
//                    colors = OutlinedTextFieldDefaults.colors(
//                        focusedBorderColor = aaccentBlue,
//                        unfocusedBorderColor = ssoftBlue
//                    ),
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(vertical = 16.dp)
//                )
//
//                Card(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(vertical = 16.dp),
//                    colors = CardDefaults.cardColors(containerColor = llightBlue),
//                    shape = RoundedCornerShape(20.dp),
//                    elevation = CardDefaults.cardElevation(4.dp)
//                ) {
//                    Column(modifier = Modifier.padding(16.dp)) {
//                        Text(
//                            text = "USD ($)",
//                            fontSize = 16.sp,
//                            fontWeight = FontWeight.SemiBold,
//                            color = deeepBlue,
//                            modifier = Modifier.padding(bottom = 8.dp)
//                        )
//
//                        FinancialInfoRow("Value per share", "$$valuePerShare")
//                        FinancialInfoRow("Available Cash", "$$availableCash")
//                        FinancialInfoRow("Total Cost", "$$totalCost")
//                        HorizontalDivider(
//                            modifier = Modifier.padding(vertical = 8.dp),
//                            color = secondaryBlackk.copy(alpha = 0.2f)
//                        )
//                        FinancialInfoRow("Resulting Cash", "$$resultingCash")
//                    }
//                }
//
////            Spacer(modifier = Modifier.weight(1f))
//
//                Button(
//                    onClick = {
//                        if (numberOfShares > 0 && resultingCash >= 0 && tickerSymbol !== "") {
//                            comp.buy(tickerSymbol, numberOfShares, userId)
//                            MainActivity.model.saveCompetition(comp)
//                            onBack()
//                        }
//                    },
//                    shape = RoundedCornerShape(15.dp),
//                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
//                    elevation = ButtonDefaults.buttonElevation(8.dp),
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(vertical = 8.dp)
//                ) {
//                    Text(
//                        text = "Confirm Purchase",
//                        color = ppureWhite,
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.Bold,
//                        modifier = Modifier.padding(vertical = 8.dp)
//                    )
//                }
//            }
//        }
//    }
//}

//class SellPage(val comp: Competition,val onBack: () -> Unit) : Screen {
//
//    @OptIn(ExperimentalMaterial3Api::class)
//    @Composable
//    override fun Content() {
//        val userId = MainActivity.model.userUID
//        val playerInfo = comp.players.getValue(userId)
//        var tickerSymbol by remember { mutableStateOf("") }
//        var numberOfShares by remember { mutableStateOf<Int>(0) }
//        var valuePerShare by remember { mutableFloatStateOf(0.0f) }
//        var sharesAvailable by remember { mutableStateOf(0) }
//
//        val availableCash = playerInfo.cash
//        val totalRevenue:Float = numberOfShares * valuePerShare
//        val resultingCash = availableCash + totalRevenue
//
//
//        Scaffold(
//            topBar = {
//                TopAppBar(
//                    title = { Text("Sell", color = deeepBlue, fontWeight = FontWeight.Bold) },
//                    navigationIcon = {
//                        IconButton(onClick = onBack) {
//                            Icon(
//                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
//                                contentDescription = "Back",
//                                tint = deeepBlue
//                            )
//                        }
//                    },
//                    colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
//                )
//            }
//        ) { innerPadding ->
//            Column(
//                modifier = Modifier
//                    .fillMaxSize()
//                    .background(ppureWhite)
//                    .padding(innerPadding)
//                    .padding(16.dp)
//            ) {
//                Text(
//                    text = "Buy",
//                    fontSize = 32.sp,
//                    fontWeight = FontWeight.Bold,
//                    color = deeepBlue,
//                    modifier = Modifier.padding(bottom = 16.dp)
//                )
//
//                var expanded by remember { mutableStateOf(false) }
//                OutlinedButton(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(bottom = 10.dp),
//                    onClick = { expanded = true },
//                    colors = ButtonDefaults.outlinedButtonColors(
//                        containerColor = ppureWhite,
//                        contentColor = aaccentBlue
//                    ),
//                    border = BorderStroke(1.dp, aaccentBlue),
//                    shape = RoundedCornerShape(15.dp)
//                ) {
//                    Text(
//                        if (tickerSymbol == "") "Select Stock"
//                        else "Selected Stock: $tickerSymbol",
//                        fontWeight = FontWeight.Medium
//                    )
//                }
//
//                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
//                    for (stock in playerInfo.portfolio) {
//                        DropdownMenuItem(
//                            text = { Text(stock.ticker) },
//                            onClick = {
//                                tickerSymbol = stock.ticker
//                                expanded = false
//                                valuePerShare = comp.getStockValue(stock.ticker)
//                                sharesAvailable = stock.quantityOwned
//                            }
//                        )
//                    }
//                }
//
//                OutlinedTextField(
//                    value = numberOfShares.toString(),
//                    keyboardOptions = KeyboardOptions(
//                        keyboardType = KeyboardType.Number,
//                        imeAction = ImeAction.Done
//                    ),
//                    onValueChange = { input: String ->
//                        numberOfShares = if (input == "") 0 else input.toInt()
//                    },
//                    label = { Text("No. of Shares (${sharesAvailable} shares available)", color = secondaryBlackk) },
//                    shape = RoundedCornerShape(15.dp),
//                    colors = OutlinedTextFieldDefaults.colors(
//                        focusedBorderColor = aaccentBlue,
//                        unfocusedBorderColor = ssoftBlue
//                    ),
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(vertical = 16.dp)
//                )
//
//                Card(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(vertical = 16.dp),
//                    colors = CardDefaults.cardColors(containerColor = llightBlue),
//                    shape = RoundedCornerShape(20.dp),
//                    elevation = CardDefaults.cardElevation(4.dp)
//                ) {
//                    Column(modifier = Modifier.padding(16.dp)) {
//                        Text(
//                            text = "USD ($)",
//                            fontSize = 16.sp,
//                            fontWeight = FontWeight.SemiBold,
//                            color = deeepBlue,
//                            modifier = Modifier.padding(bottom = 8.dp)
//                        )
//
//                        FinancialInfoRow("Value per share", "$$valuePerShare")
//                        FinancialInfoRow("Available Cash", "$$availableCash")
//                        FinancialInfoRow("Total Revenue", "$$totalRevenue")
//                        HorizontalDivider(
//                            modifier = Modifier.padding(vertical = 8.dp),
//                            color = secondaryBlackk.copy(alpha = 0.2f)
//                        )
//                        FinancialInfoRow("Resulting Cash", "$$resultingCash")
//                    }
//                }
//
////            Spacer(modifier = Modifier.weight(1f))
//
//                Button(
//                    onClick = {
//                        if (numberOfShares > 0 && numberOfShares <= sharesAvailable && tickerSymbol !== "") {
//                            comp.sell(tickerSymbol, numberOfShares, userId)
//                            MainActivity.model.saveCompetition(comp)
//                            onBack()
//                        }
//                    },
//                    shape = RoundedCornerShape(15.dp),
//                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
//                    elevation = ButtonDefaults.buttonElevation(8.dp),
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .padding(vertical = 8.dp)
//                ) {
//                    Text(
//                        text = "Confirm Sale",
//                        color = ppureWhite,
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.Bold,
//                        modifier = Modifier.padding(vertical = 8.dp)
//                    )
//                }
//            }
//        }
//    }
//}

//class NewsListPage(val comp: Competition) : Screen {
//    @OptIn(ExperimentalMaterial3Api::class)
//    @Composable
//    override fun Content() {
//        var newsList by remember { mutableStateOf<List<NewsEvent>>(comp.newsEvents) }
//        var selectedNews by remember { mutableStateOf<NewsEvent?>(null) }
//
//        var navigator = LocalNavigator.currentOrThrow
//
//        if (selectedNews != null) {
//            NewsDetailPage(newsItem = selectedNews!!, onBack = { selectedNews = null }, comp)
//        } else {
//            Scaffold(
//                topBar = {
//                    TopAppBar(
//                        title = { Text("News", color = deeepBlue, fontWeight = FontWeight.Bold) },
//                        navigationIcon = {
//                            IconButton(
//                                onClick = {
//                                    if (MainActivity.model.isAdmin) {
//                                        navigator.push(ManageCompetitionPage(comp))
//                                    } else {
//                                        navigator.push(OngoingCompetitionPage(comp))
//                                    }
//                                }
//                            ) {
//                                Icon(
//                                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
//                                    contentDescription = "Back",
//                                    tint = deeepBlue
//                                )
//                            }
//                        },
//                        actions = {
//                            if (MainActivity.model.isAdmin) {
//                                Button(
//                                    onClick = { runBlocking {
//                                        val newEvent = async { MainActivity.model.generateNewsEvent(comp.availableStocks) }
//                                        newsList += newEvent.await()
//                                        comp.newsEvents = newsList as MutableList<NewsEvent>
//                                        comp.latestNewsNotRevealed = newEvent.await().title
//                                        MainActivity.model.saveCompetition(comp)
//                                    }
//                                    },
//                                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
//                                    shape = RoundedCornerShape(15.dp),
//                                    elevation = ButtonDefaults.buttonElevation(4.dp),
//                                    enabled = comp.latestNewsNotRevealed.length == 0 && comp.hasStarted
//                                ) {
//                                    Text(
//                                        text = "New Event (Admin Only)",
//                                        color = ppureWhite,
//                                        fontWeight = FontWeight.Bold
//                                    )
//                                }
//
//                            }
//                        },
//                        colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
//                    )
//                }
//            ) { innerPadding ->
//
//                LazyColumn(
//                    modifier = Modifier
//                        .fillMaxSize()
//                        .background(ppureWhite)
//                        .padding(innerPadding)
//                        .padding(16.dp)
//
//
//                ) {
//                    items(newsList.reversed()) { newsItem ->
//                        NewsListItem(newsItem = newsItem, onClick = { selectedNews = newsItem })
//                    }
//                }
//            }
//        }
//    }
//}
//
//@Composable
//fun NewsListItem(newsItem: NewsEvent, onClick: () -> Unit) {
//    Card(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(vertical = 8.dp)
//            .clickable { onClick() },
//        colors = CardDefaults.cardColors(containerColor = llightBlue),
//        shape = RoundedCornerShape(15.dp),
//        elevation = CardDefaults.cardElevation(2.dp)
//    ) {
//        Column(
//            modifier = Modifier.padding(16.dp)
//        ) {
//            Text(
//                text = newsItem.title,
//                fontWeight = FontWeight.Bold,
//                fontSize = 20.sp,
//                color = pprimaryBlack
//            )
//            Spacer(modifier = Modifier.height(8.dp))
//            Text(
//                text = newsItem.body,
//                fontSize = 14.sp,
//                maxLines = 2,
//                overflow = TextOverflow.Ellipsis,
//                color = secondaryBlackk
//            )
//        }
//    }
//}

//@OptIn(ExperimentalMaterial3Api::class)
//@Composable
//fun NewsDetailPage(newsItem: NewsEvent, onBack: () -> Unit, comp: Competition) {
//
//    Scaffold(
//        topBar = {
//            TopAppBar(
//                title = {
//                    Text(
//                        "News Event",
//                        color = deeepBlue,
//                        fontWeight = FontWeight.Bold
//                    )
//                },
//                navigationIcon = {
//                    IconButton(onClick = onBack) {
//                        Icon(
//                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
//                            contentDescription = "Back",
//                            tint = deeepBlue
//                        )
//                    }
//                },
//                colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
//            )
//        }
//    ) { innerPadding ->
//        Column(
//            modifier = Modifier
//                .fillMaxSize()
//                .background(ppureWhite)
//                .padding(innerPadding)
//                .padding(16.dp)
//                .padding(bottom = 100.dp)
//                .verticalScroll(ScrollState(0), true)
//
//        ) {
//            Text(
//                text = newsItem.title,
//                fontSize = 24.sp,
//                fontWeight = FontWeight.Bold,
//                color = pprimaryBlack
//            )
//            Spacer(modifier = Modifier.height(16.dp))
//            Text(
//                text = newsItem.body,
//                fontSize = 16.sp,
//                color = secondaryBlackk,
//                lineHeight = 24.sp
//            )
//            Spacer(modifier = Modifier.height(16.dp))
//            Text(
//                text = "New Stock Prices:",
//                fontSize = 16.sp,
//                color = secondaryBlackk,
//                lineHeight = 24.sp
//            )
//
//            if (!MainActivity.model.isAdmin && comp.latestNewsNotRevealed == newsItem.title) {
//                Text("Price updates have not yet been revealed by the Competition moderator. Check back soon!")
//            } else {
//                newsItem.newPrices.forEach { newPrice ->
//                    Text(
//                        text = newPrice.ticker + ": " + newPrice.newPrice.toString() + "( " + (if (newPrice.changeInPrice >= 0) "+" else "") + newPrice.changeInPrice.toString() + ")",
//                        fontSize = 16.sp,
//                        color = secondaryBlackk,
//                        lineHeight = 24.sp
//                    )
//                }
//            }
//
//
////            Card(
////                modifier = Modifier.fillMaxWidth(),
////                colors = CardDefaults.cardColors(containerColor = llightBlue),
////                shape = RoundedCornerShape(20.dp),
////                elevation = CardDefaults.cardElevation(4.dp)
////            ) {
////                Column(modifier = Modifier.padding(16.dp)) {
////                    Text(
////                        text = newsItem.title,
////                        fontSize = 24.sp,
////                        fontWeight = FontWeight.Bold,
////                        color = pprimaryBlack
////                    )
////                    Spacer(modifier = Modifier.height(16.dp))
////                    Text(
////                        text = newsItem.body,
////                        fontSize = 16.sp,
////                        color = secondaryBlackk,
////                        lineHeight = 24.sp
////                    )
////                    Spacer(modifier = Modifier.height(16.dp))
////                    Text(
////                        text = "New Stock Prices:",
////                        fontSize = 16.sp,
////                        color = secondaryBlackk,
////                        lineHeight = 24.sp
////                    )
////                    newsItem.newPrices.forEach { newPrice ->
////                        Text(
////                            text = newPrice.ticker + ": " + newPrice.newPrice.toString(),
////                            fontSize = 16.sp,
////                            color = secondaryBlackk,
////                            lineHeight = 24.sp
////                        )
////                    }
////                }
////            }
//        }
//    }
//}