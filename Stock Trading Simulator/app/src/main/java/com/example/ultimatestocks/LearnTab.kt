//package com.example.ultimatestocks
//import androidx.compose.ui.graphics.vector.rememberVectorPainter
//import androidx.compose.material.icons.automirrored.filled.ArrowBack
//import androidx.compose.ui.text.input.VisualTransformation
//import androidx.compose.foundation.BorderStroke
//import androidx.compose.foundation.ScrollState
//import androidx.compose.foundation.lazy.LazyColumn
//import androidx.compose.foundation.lazy.items
//import androidx.compose.foundation.clickable
//import androidx.compose.material.icons.filled.ArrowBack
//import androidx.compose.ui.text.style.TextOverflow
//import androidx.compose.foundation.background
//import androidx.compose.foundation.gestures.FlingBehavior
//import androidx.compose.foundation.gestures.ScrollableDefaults
//import androidx.compose.foundation.layout.*
//import androidx.compose.foundation.rememberScrollState
//import androidx.compose.foundation.shape.RoundedCornerShape
//import androidx.compose.foundation.text.KeyboardOptions
//import androidx.compose.foundation.verticalScroll
//import androidx.compose.material.icons.Icons
//import androidx.compose.material.icons.filled.Add
//import androidx.compose.material.icons.filled.MoreVert
//import androidx.compose.material.icons.outlined.Edit
//import androidx.compose.material.icons.outlined.Email
//import androidx.compose.material.icons.outlined.Settings
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Alignment
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Color
//import androidx.compose.ui.text.font.FontWeight
//import androidx.compose.ui.text.input.ImeAction
//import androidx.compose.ui.text.input.KeyboardType
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//import cafe.adriel.voyager.navigator.tab.Tab
//import cafe.adriel.voyager.navigator.tab.TabOptions
//import co.yml.charts.axis.AxisData
//import co.yml.charts.common.model.Point
//import co.yml.charts.ui.linechart.LineChart
//import co.yml.charts.ui.linechart.model.*
//import co.yml.charts.ui.linechart.model.Line
//import co.yml.charts.ui.linechart.model.LineChartData
//import co.yml.charts.ui.linechart.model.LinePlotData
//import co.yml.charts.ui.linechart.model.LineStyle
//import com.example.ultimatestocks.entities.FirebaseStock
//import com.example.ultimatestocks.entities.NewsEvent
//import com.example.ultimatestocks.entities.Sandbox
//import com.example.ultimatestocks.entities.StockDetails
//import com.example.ultimatestocks.entities.StockInventory
//import com.example.ultimatestocks.view.ViewModel
//import kotlinx.coroutines.async
//import kotlinx.coroutines.channels.ticker
//import kotlinx.coroutines.runBlocking
//import kotlin.time.Duration.Companion.hours
//import androidx.compose.foundation.layout.*
//import androidx.compose.ui.text.font.Typeface
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//import cafe.adriel.voyager.navigator.Navigator
//import co.yml.charts.common.extensions.isNotNull
//import com.example.ultimatestocks.sandboxes.SandboxesHomePage
//import kotlin.math.floor
//import kotlin.math.log10
//import kotlin.math.pow
//
//internal val llightBlue = Color(0xFFE3F2FD)
//internal val aaccentBlue = Color(0xFF2196F3)
//internal val deeepBlue = Color(0xFF1976D2)
//internal val ssoftBlue = Color(0xFFBBDEFB)
//val pprimaryBlack = Color(0xFF111111)
//internal val secondaryBlackk = Color(0xFF2D2D2D)
//internal val ppureWhite = Color(0xFFFFFFFF)
//
//
//
//@Composable
//fun LearnScreen(vm: ViewModel, modifier: Modifier = Modifier) {
//    var selectedSandbox by remember { mutableStateOf<Sandbox?>(null) }
//    var showBuyPage by remember { mutableStateOf(false) }
//    var showSellPage by remember { mutableStateOf(false) }
//    var showCreateSandboxPage by remember { mutableStateOf(false)}
//
//    when {
//        showBuyPage -> {
//            BuyPage(
//                sandbox = selectedSandbox!!,
//                onBack = { showBuyPage = false }  // Go back to SandboxDetailPage
//            )
//        }
//        showSellPage -> {
//            SellPage(
//                sandbox = selectedSandbox!!,
//                onBack = { showSellPage = false }  // Go back to SandboxDetailPage
//            )
//        }
//        showCreateSandboxPage -> {
//            CreateSandboxPage(
//                vm = vm,
//                onOpenSandbox = { sandbox ->
//                    selectedSandbox = sandbox
//                },
//                onBack = {
//                    showCreateSandboxPage = false
//                }
//            )
//        }
//        selectedSandbox == null -> {
//            LearnPage(
//                vm = vm,
//                onOpenSandbox = { sandbox ->
//                    selectedSandbox = sandbox  // Open selected sandbox in SandboxDetailPage
//                },
//                onCreateNewSandbox = {
//                    showCreateSandboxPage = true
//                }
//            )
//        }
//        else -> {
//            SandboxDetailPage(
//                selectedSandbox!!,
//                onBack = { selectedSandbox = null },
//                onBuyClick = { showBuyPage = true },
//                onSellClick = { showSellPage = true }
//            )
//        }
//    }
//}
//val dummyStocks = listOf( // this dummy data is generated using ChatGPT
//StockDetails("AAPL", "Apple Inc. designs and manufactures consumer electronics and software.", mutableListOf(125.50f, 127.30f, 126.40f, 128.90f, 127.80f)),
//StockDetails("GOOGL", "Alphabet Inc. is the parent company of Google, specializing in internet services and products.", mutableListOf(2345.70f, 2360.80f, 2350.20f, 2372.10f, 2380.60f)),
//StockDetails("MSFT", "Microsoft Corporation develops software, services, and solutions for businesses and individuals.", mutableListOf(275.40f, 276.80f, 274.90f, 278.30f, 277.00f)),
//StockDetails("AMZN", "Amazon.com, Inc. is an e-commerce and cloud computing giant.", mutableListOf(3350.10f, 3360.40f, 3345.90f, 3380.20f, 3375.10f)),
//StockDetails("META", "Meta Platforms, Inc. focuses on social media and virtual reality.", mutableListOf(300.20f, 302.40f, 301.30f, 303.50f, 302.80f)),
//StockDetails("TSLA", "Tesla, Inc. designs and manufactures electric vehicles and energy storage solutions.", mutableListOf(680.30f, 685.20f, 678.40f, 690.10f, 687.30f)),
//StockDetails("NFLX", "Netflix, Inc. is a streaming entertainment service provider.", mutableListOf(540.20f, 542.50f, 538.90f, 544.60f, 543.30f)),
//StockDetails("NVDA", "NVIDIA Corporation designs graphics processing units and AI solutions.", mutableListOf(750.40f, 752.80f, 748.20f, 755.30f, 753.90f)),
//StockDetails("ADBE", "Adobe Inc. provides software solutions for creatives and businesses.", mutableListOf(560.30f, 562.70f, 559.20f, 563.80f, 561.50f)),
//StockDetails("INTC", "Intel Corporation designs and manufactures microprocessors and other semiconductor products.", mutableListOf(52.20f, 51.80f, 52.70f, 53.10f, 52.50f))
//)
//val defaultSandbox = Sandbox(
//    name = "New Sandbox",
//    newsEvents = mutableListOf(NewsEvent("Welcome to the Sandbox", "This is a sample news event. " +
//            "News events allow you to simulate real-life events and observe how they impact different stocks, " +
//            "so you can explore with unlimited scenarios at no risk to you.", listOf())),
//    allStocks = listOf( // this dummy data is generated using ChatGPT
//        StockDetails("AAPL", "Apple Inc. designs and manufactures consumer electronics and software.", mutableListOf(125.50f, 127.30f, 126.40f, 128.90f, 127.80f)),
////        StockDetails("GOOGL", "Alphabet Inc. is the parent company of Google, specializing in internet services and products.", listOf(2345.70f, 2360.80f, 2350.20f, 2372.10f, 2380.60f)),
//        StockDetails("MSFT", "Microsoft Corporation develops software, services, and solutions for businesses and individuals.", mutableListOf(275.40f, 276.80f, 274.90f, 278.30f, 277.00f)),
////        StockDetails("AMZN", "Amazon.com, Inc. is an e-commerce and cloud computing giant.", listOf(3350.10f, 3360.40f, 3345.90f, 3380.20f, 3375.10f)),
//        StockDetails("META", "Meta Platforms, Inc. focuses on social media and virtual reality.", mutableListOf(300.20f, 302.40f, 301.30f, 303.50f, 302.80f)),
////        StockDetails("TSLA", "Tesla, Inc. designs and manufactures electric vehicles and energy storage solutions.", listOf(680.30f, 685.20f, 678.40f, 690.10f, 687.30f)),
////        StockDetails("NFLX", "Netflix, Inc. is a streaming entertainment service provider.", listOf(540.20f, 542.50f, 538.90f, 544.60f, 543.30f)),
////        StockDetails("NVDA", "NVIDIA Corporation designs graphics processing units and AI solutions.", listOf(750.40f, 752.80f, 748.20f, 755.30f, 753.90f)),
////        StockDetails("ADBE", "Adobe Inc. provides software solutions for creatives and businesses.", listOf(560.30f, 562.70f, 559.20f, 563.80f, 561.50f)),
//        StockDetails("INTC", "Intel Corporation designs and manufactures microprocessors and other semiconductor products.", mutableListOf(52.20f, 51.80f, 52.70f, 53.10f, 52.50f))
//    ),
//    ownedStocks = mutableListOf(),
//    cash = 500.00f,
//    portfolioVal = mutableListOf(500.00f),
//    latestNewsNotRevealed = ""
//)
//
//// LearnPage with callback for "Open Sandbox"
//@Composable
//fun LearnPage(
//    vm: ViewModel,
//    onOpenSandbox: (Sandbox) -> Unit,
//    modifier: Modifier = Modifier,
//    onCreateNewSandbox: () -> Unit,
//) {
//    val viewModel by remember { mutableStateOf(vm) }
//    Box(
//        modifier = modifier
//            .background(color = ppureWhite)
//            .fillMaxSize()
//    ) {
//        Column(
//            modifier = Modifier
//                .fillMaxSize()
//                .padding(bottom = 80.dp)
//                .verticalScroll(rememberScrollState()),
//            horizontalAlignment = Alignment.CenterHorizontally
//        ) {
//            Spacer(modifier = Modifier.height(32.dp))
//
//            Row(modifier = Modifier.fillMaxWidth()) {
//                Text(
//                    text = "Learn",
//                    fontSize = 32.sp,
//                    fontWeight = FontWeight.Bold,
//                    color = deeepBlue,
//                    modifier = Modifier.padding(start = 16.dp)
//                )
//            }
//
//            Spacer(modifier = Modifier.height(8.dp))
//
//            Button(
//                onClick = { onCreateNewSandbox() },
//                shape = RoundedCornerShape(15.dp),
//                colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
//                elevation = ButtonDefaults.buttonElevation(8.dp),
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(horizontal = 16.dp, vertical = 8.dp)
//            ) {
//                Text(
//                    text = "Create New Sandbox",
//                    color = ppureWhite,
//                    fontSize = 16.sp,
//                    fontWeight = FontWeight.Bold,
//                    modifier = Modifier.padding(8.dp)
//                )
//            }
//
//            Spacer(modifier = Modifier.height(16.dp))
//
//            Text(
//                text = "Saved Sandboxes",
//                fontSize = 24.sp,
//                fontWeight = FontWeight.Bold,
//                color = deeepBlue,
//                modifier = Modifier
//                    .align(Alignment.Start)
//                    .padding(start = 16.dp)
//            )
//
//            Spacer(modifier = Modifier.height(16.dp))
//
//            for (item in viewModel.sandboxes) {
//                SandboxCard(
//                    name = item.name,
//                    portfolioValue = item.portfolioVal.last().toString(),
//                    newsEventsGenerated = item.newsEvents.count().toString(),
//                    onDelete = { viewModel.model.del(item) },
//                    onOpen = { onOpenSandbox(item) }
//                )
//            }
//
//            Spacer(modifier = Modifier.height(16.dp))
//        }
//    }
//}
//
//@OptIn(ExperimentalMaterial3Api::class)
//@Composable
//fun CreateSandboxPage(
//    vm: ViewModel,
//    onOpenSandbox: (Sandbox) -> Unit,
//    onBack: () -> Unit,
//    modifier: Modifier = Modifier
//) {
//    val viewModel by remember { mutableStateOf(vm) }
//    var sandboxName by remember { mutableStateOf("") }
//    var sandboxCash by remember { mutableStateOf(0) }
//    var selectedStocks by remember { mutableStateOf<List<FirebaseStock>>(listOf())}
//    var expanded by remember { mutableStateOf(false) }
//
//    Scaffold(
//        topBar = {
//            TopAppBar(
//                title = { Text("Create New Sandbox", color = deeepBlue, fontWeight = FontWeight.Bold) },
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
//            modifier = modifier
//                .background(color = ppureWhite)
//                .fillMaxSize()
//                .verticalScroll(rememberScrollState())
//                .padding(innerPadding)
//                .padding(10.dp),
//            horizontalAlignment = Alignment.CenterHorizontally
//        ) {
//            Spacer(modifier = Modifier.height(32.dp))
//
//            Row(modifier = Modifier.fillMaxWidth()) {
//                Text(
//                    text = "New Sandbox",
//                    fontSize = 32.sp,
//                    fontWeight = FontWeight.Bold,
//                    color = deeepBlue,
//                    modifier = Modifier.padding(start = 16.dp)
//                )
//            }
//
//            Spacer(modifier = Modifier.height(8.dp))
//
//            OutlinedTextField(
//                value = sandboxName,
//                keyboardOptions = KeyboardOptions(
//                    keyboardType = KeyboardType.Number,
//                    imeAction = ImeAction.Done
//                ),
//                onValueChange = { input: String ->
//                    sandboxName = input
//                },
//                label = { Text("Sandbox Title", color = secondaryBlackk) },
//                shape = RoundedCornerShape(15.dp),
//                colors = OutlinedTextFieldDefaults.colors(
//                    focusedBorderColor = aaccentBlue,
//                    unfocusedBorderColor = ssoftBlue
//                ),
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(vertical = 16.dp)
//            )
//            OutlinedTextField(
//                value = sandboxCash.toString(),
//                keyboardOptions = KeyboardOptions(
//                    keyboardType = KeyboardType.Number,
//                    imeAction = ImeAction.Done
//                ),
//                onValueChange = { input: String ->
//                    sandboxCash = if (input == "") 0 else input.toInt()
//                },
//                label = { Text("Starting Cash", color = secondaryBlackk) },
//                shape = RoundedCornerShape(15.dp),
//                colors = OutlinedTextFieldDefaults.colors(
//                    focusedBorderColor = aaccentBlue,
//                    unfocusedBorderColor = ssoftBlue
//                ),
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(vertical = 16.dp)
//            )
//
//            for (stock in selectedStocks) {
//                Text(stock.ticker + " at " + stock.startingPrice.toString())
//            }
//
//            OutlinedButton(
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(bottom = 10.dp),
//                onClick = { expanded = true },
//                colors = ButtonDefaults.outlinedButtonColors(
//                    containerColor = ppureWhite,
//                    contentColor = aaccentBlue
//                ),
//                border = BorderStroke(1.dp, aaccentBlue),
//                shape = RoundedCornerShape(15.dp)
//            ) {
//                Text(
//                    "Add Stock",
//                    fontWeight = FontWeight.Medium
//                )
//            }
//
//            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
//                for (stock in MainActivity.model.stocks) {
//                    DropdownMenuItem(
//                        text = { Text(stock.companyName + ": " + stock.description) },
//                        onClick = {
//                            if (!selectedStocks.contains(stock)) {
//                                selectedStocks = selectedStocks + stock
//                            }
//                            expanded = false
//                        }
//                    )
//                }
//            }
//
//            Button(
////                onClick = { viewModel.model.add(defaultSandbox) },
//                onClick = {
//                    if (sandboxName.length > 0 && sandboxCash > 0 && !selectedStocks.isEmpty()) {
//                        viewModel.model.add(Sandbox(
//                            name = sandboxName,
//                            newsEvents = mutableListOf(NewsEvent("Welcome to the Sandbox", "This is a sample news event. " +
//                                    "News events allow you to simulate real-life events and observe how they impact different stocks, " +
//                                    "so you can explore with unlimited scenarios at no risk to you.", listOf())),
//                            latestNewsNotRevealed = "",
//                            allStocks = selectedStocks.map {stock -> StockDetails(stock.ticker, stock.description, mutableListOf(stock.startingPrice.toFloat())) },
//                            ownedStocks = mutableListOf(),
//                            portfolioVal = mutableListOf(sandboxCash.toFloat()),
//                            cash = sandboxCash.toFloat()
//                        ))
//                        onBack()
//                    }
//                },
//                shape = RoundedCornerShape(15.dp),
//                colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
//                elevation = ButtonDefaults.buttonElevation(8.dp),
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(vertical = 8.dp)
//            ) {
//                Text(
//                    text = "Create New Sandbox",
//                    color = ppureWhite,
//                    fontSize = 16.sp,
//                    fontWeight = FontWeight.Bold,
//                    modifier = Modifier.padding(8.dp)
//                )
//            }
//
//
//        }
//    }
//}
//
//@Composable
//fun SandboxCard(
//    name: String,
//    portfolioValue: String,
//    newsEventsGenerated: String,
//    onDelete: () -> Unit,
//    onOpen: () -> Unit,
//    modifier: Modifier = Modifier
//) {
//    Card(
//        shape = RoundedCornerShape(20.dp),
//        colors = CardDefaults.cardColors(containerColor = llightBlue),
//        elevation = CardDefaults.cardElevation(4.dp),
//        modifier = modifier
//            .fillMaxWidth()
//            .padding(vertical = 8.dp, horizontal = 16.dp)
//    ) {
//        Column(
//            modifier = Modifier.padding(16.dp),
//            verticalArrangement = Arrangement.spacedBy(8.dp)
//        ) {
//            Text(
//                text = name,
//                fontWeight = FontWeight.Bold,
//                fontSize = 24.sp,
//                color = pprimaryBlack
//            )
//
//            Text(
//                text = "Portfolio value: $$portfolioValue",
//                fontSize = 16.sp,
//                color = secondaryBlackk
//            )
//
//            Text(
//                text = "News Events Generated: $newsEventsGenerated",
//                fontSize = 16.sp,
//                color = secondaryBlackk
//            )
//
//            Spacer(modifier = Modifier.height(8.dp))
//
//            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
//                Button(
//                    onClick = onOpen,
//                    shape = RoundedCornerShape(15.dp),
//                    colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
//                    elevation = ButtonDefaults.buttonElevation(8.dp),
//                    modifier = Modifier.weight(2f)
//                ) {
//                    Text(
//                        text = "Open Sandbox",
//                        color = ppureWhite,
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.Bold
//                    )
//                }
//
//                Button(
//                    onClick = onDelete,
//                    shape = RoundedCornerShape(15.dp),
//                    colors = ButtonDefaults.buttonColors(containerColor = deeepBlue),
//                    elevation = ButtonDefaults.buttonElevation(8.dp),
//                    modifier = Modifier.weight(1f)
//                ) {
//                    Text(
//                        text = "Delete",
//                        color = ppureWhite,
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.Bold
//                    )
//                }
//            }
//        }
//    }
//}
//
////@OptIn(ExperimentalMaterial3Api::class)
////@Composable
////fun SandboxDetailPage(
////    originalSandbox: Sandbox,
////    onBack: () -> Unit,
////    onBuyClick: () -> Unit,
////    onSellClick: () -> Unit
////) {
////    var sandbox by remember { mutableStateOf(originalSandbox) }
////    var searchQuery by remember { mutableStateOf("") }
////    var showBuyPage by remember { mutableStateOf(false) }
////    var showSellPage by remember { mutableStateOf(false) }
////    var showNewsPage by remember { mutableStateOf(false) }
////
////    when {
////        showBuyPage -> BuyPage(sandbox) { showBuyPage = false }
////        showSellPage -> SellPage(sandbox) { showSellPage = false }
////        showNewsPage -> NewsPage(sandbox) { showNewsPage = false }
////        else -> Scaffold(
////            topBar = {
////                TopAppBar(
////                    title = { Text(sandbox.name, color = deeepBlue, fontWeight = FontWeight.Bold) },
////                    navigationIcon = {
////                        IconButton(onClick = onBack) {
////                            Icon(
////                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
////                                contentDescription = "Back",
////                                tint = deeepBlue
////                            )
////                        }
////                    },
////                    colors = TopAppBarDefaults.topAppBarColors(
////                        containerColor = ppureWhite
////                    )
////                )
////            }
////        ) { innerPadding ->
////            Column(
////                modifier = Modifier
////                    .fillMaxSize()
////                    .background(color = ppureWhite)
////                    .padding(innerPadding)
////                    .padding(16.dp)
////            ) {
////                Row(
////                    modifier = Modifier.fillMaxWidth(),
////                    horizontalArrangement = Arrangement.SpaceBetween,
////                    verticalAlignment = Alignment.CenterVertically
////                ) {
////                    Text(
////                        text = sandbox.name,
////                        fontSize = 24.sp,
////                        fontWeight = FontWeight.Bold,
////                        color = deeepBlue
////                    )
////                    Button(
////                        onClick = { showNewsPage = true },
////                        colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
////                        elevation = ButtonDefaults.buttonElevation(8.dp),
////                        shape = RoundedCornerShape(15.dp)
////                    ) {
////                        Text(
////                            text = "News",
////                            color = ppureWhite,
////                            fontWeight = FontWeight.Bold
////                        )
////                    }
////                }
////
////                Spacer(modifier = Modifier.height(16.dp))
////
////                Card(
////                    modifier = Modifier
////                        .fillMaxWidth()
////                        .height(200.dp),
////                    colors = CardDefaults.cardColors(containerColor = llightBlue),
////                    shape = RoundedCornerShape(20.dp),
////                    elevation = CardDefaults.cardElevation(4.dp)
////                ) {
////                    Box(
////                        modifier = Modifier.fillMaxSize(),
////                        contentAlignment = Alignment.Center
////                    ) {
////                        val allStocks = sandbox.allStocks
////
////                        // Calculate the maximum number of data points across all stocks
////                        val maxDataPoints = allStocks.maxOfOrNull { it.historicPrice.size } ?: 0
////
////                        // Prepare line data with distinct colors for each stock
////                        val lines = allStocks.mapIndexed { index, stock ->
////                            Line(
////                                dataPoints = stock.historicPrice.mapIndexed { priceIndex, price ->
////                                    Point(priceIndex.toFloat(), price)
////                                },
////                                lineStyle = LineStyle(
////                                    color = Color.hsv(
////                                        hue = (index * 360f / allStocks.size) % 360f,
////                                        saturation = 0.8f,
////                                        value = 1f
////                                    ),
////                                    width = 2f
////                                ),
////                                intersectionPoint = IntersectionPoint(
////                                    color = Color.Gray,
////                                    radius = 4.dp
////                                ),
////                                selectionHighlightPoint = SelectionHighlightPoint(
////                                    color = Color.Red,
////                                    radius = 6.dp
////                                ),
////                                shadowUnderLine = ShadowUnderLine(
////                                    color = Color.hsv(
////                                        hue = (index * 360f / allStocks.size) % 360f,
////                                        saturation = 0.8f,
////                                        value = 0.5f
////                                    ),
////                                    alpha = 0.2f
////                                ),
////                                selectionHighlightPopUp = SelectionHighlightPopUp(
////                                    backgroundColor = Color.White,
////                                    labelColor = Color.Black,
////                                    labelSize = 12.sp
////                                )
////                            )
////                        }
////
////                        // Calculate Y-axis range based on the data
////                        val allPrices = allStocks.flatMap { it.historicPrice }
////                        val yMin = allPrices.minOrNull() ?: 0f
////                        val yMax = allPrices.maxOrNull() ?: 100f
////                        val ySteps = 10
////                        val yStepValue = (yMax - yMin) / ySteps
////
////                        val desiredLabelCount = 6
////                        val labelStep = if (maxDataPoints / desiredLabelCount > 0) maxDataPoints / desiredLabelCount else 1
////
////                        val xAxisData = AxisData.Builder()
////                            .axisStepSize(20.dp)
////                            .steps(maxDataPoints - 1)
////                            .labelData { i ->
////                                if (i % labelStep == 0 || i == maxDataPoints - 1) {
////                                    "Day ${i + 1}"
////                                } else {
////                                    ""
////                                }
////                            }
////                            .labelAndAxisLinePadding(16.dp)
////                            .axisLabelAngle(-45f) // Rotated labels for better readability
////                            .axisLabelFontSize(10.sp) // Reduced font size to prevent overlap
////                            .axisLineColor(Color.Gray)
////                            .axisLabelColor(Color.Black)
////                            .build()
////
////                        val yAxisData = AxisData.Builder()
////                            .steps(ySteps)
////                            .labelData { i ->
////                                val value = yMin + i * yStepValue
////                                String.format("%.2f", value)
////                            }
////                            .labelAndAxisLinePadding(8.dp)
////                            .axisLabelFontSize(12.sp) // Consistent font size with X-axis
////                            .axisLineColor(Color.Gray)
////                            .axisLabelColor(Color.Black)
////                            .build()
////
////                        val lineChartData = LineChartData(
////                            linePlotData = LinePlotData(lines = lines),
////                            xAxisData = xAxisData,
////                            yAxisData = yAxisData,
////                            gridLines = GridLines(
////                                color = Color.LightGray,
////                                lineWidth = 1.dp
////                            ),
////                            backgroundColor = Color.White
////                        )
////
////                        LineChart(
////                            modifier = Modifier
////                                .fillMaxWidth()
////                                .height(300.dp),
////                            lineChartData = lineChartData
////                        )
////                    }
////
////                }
////
////                Spacer(modifier = Modifier.height(16.dp))
////
////                Row(
////                    modifier = Modifier.fillMaxWidth(),
////                    horizontalArrangement = Arrangement.spacedBy(16.dp)
////                ) {
////                    Button(
////                        onClick = { showBuyPage = true },
////                        shape = RoundedCornerShape(15.dp),
////                        colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
////                        elevation = ButtonDefaults.buttonElevation(8.dp),
////                        modifier = Modifier.weight(1f)
////                    ) {
////                        Text(
////                            text = "Buy",
////                            color = ppureWhite,
////                            fontSize = 16.sp,
////                            fontWeight = FontWeight.Bold
////                        )
////                    }
////
////                    Button(
////                        onClick = { showSellPage = true },
////                        shape = RoundedCornerShape(15.dp),
////                        colors = ButtonDefaults.buttonColors(containerColor = deeepBlue),
////                        elevation = ButtonDefaults.buttonElevation(8.dp),
////                        modifier = Modifier.weight(1f)
////                    ) {
////                        Text(
////                            text = "Sell",
////                            color = ppureWhite,
////                            fontSize = 16.sp,
////                            fontWeight = FontWeight.Bold
////                        )
////                    }
////                }
////
////                MyCustomTextField(searchQuery) { query ->
////                    searchQuery = query
////                }
////
////                val filteredStocks = filterStocks(searchQuery, sandbox.ownedStocks)
////
////                Text(
////                    text = "Current Portfolio",
////                    fontSize = 24.sp,
////                    fontWeight = FontWeight.Bold,
////                    color = deeepBlue,
////                    modifier = Modifier.padding(vertical = 16.dp)
////                )
////
////                StockList(stocks = filteredStocks)
////            }
////        }
////    }
////}
//
//fun calculateYStepValue(yRange: Float): Float {
//    val approxStep = yRange / 5 // Aim for around 5 steps
//    val magnitude = 10.0.pow(floor(log10(approxStep.toDouble()))).toFloat()
//    val residual = approxStep / magnitude
//    val stepSize = when {
//        residual < 1f -> 1f
//        residual < 2f -> 2f
//        residual < 5f -> 5f
//        else -> 10f
//    }
//    return stepSize * magnitude
//}
//
//@Composable
//fun MyCustomTextField(searchQuery: String, onQueryChanged: (String) -> Unit) {
//    OutlinedTextField(
//        value = searchQuery,
//        onValueChange = { onQueryChanged(it) },
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(vertical = 8.dp),
//        label = { Text("Search", color = secondaryBlackk) },
//        colors = OutlinedTextFieldDefaults.colors(
//            focusedBorderColor = aaccentBlue,
//            unfocusedBorderColor = ssoftBlue
//        ),
//        shape = RoundedCornerShape(15.dp)
//    )
//}
//
//@Composable
//fun StockCard(stock: StockInventory) {
//    Card(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(vertical = 4.dp, horizontal = 8.dp),
//        elevation = CardDefaults.cardElevation(4.dp),
//        colors = CardDefaults.cardColors(containerColor = llightBlue),
//        shape = RoundedCornerShape(15.dp)
//    ) {
//        Row(
//            modifier = Modifier
//                .fillMaxWidth()
//                .padding(16.dp),
//            horizontalArrangement = Arrangement.SpaceBetween,
//            verticalAlignment = Alignment.CenterVertically
//        ) {
//            Column {
//                Text(
//                    text = stock.ticker,
//                    fontWeight = FontWeight.Bold,
//                    fontSize = 18.sp,
//                    color = pprimaryBlack
//                )
//            }
//            Text(
//                text = stock.quantityOwned.toString(),
//                fontWeight = FontWeight.Bold,
//                fontSize = 18.sp,
//                color = pprimaryBlack
//            )
//        }
//    }
//}
//
//@Composable
//fun StockList(stocks: List<StockInventory>) {
//    LazyColumn(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(8.dp)
//    ) {
//        items(stocks) { stock ->
//            StockCard(stock)
//            Spacer(modifier = Modifier.height(8.dp))
//        }
//    }
//}
//
//fun filterStocks(query: String, allStocks: List<StockInventory>): List<StockInventory> {
//    if (query.isEmpty()) {
//        return allStocks
//    }
//    return allStocks.filter {
//        it.ticker.contains(query, ignoreCase = true)
//    }
//}
//
////@OptIn(ExperimentalMaterial3Api::class)
////@Composable
////fun BuyPage(sandbox: Sandbox, onBack: () -> Unit) {
////    var tickerSymbol by remember { mutableStateOf("") }
////    var numberOfShares by remember { mutableStateOf<Int>(0) }
////    var valuePerShare by remember { mutableFloatStateOf(0.0f) }
////
////    val availableCash = sandbox.cash
////    val totalCost:Float = numberOfShares * valuePerShare
////    val resultingCash = availableCash - totalCost
////
////    Scaffold(
////        topBar = {
////            TopAppBar(
////                title = { Text("Buy", color = deeepBlue, fontWeight = FontWeight.Bold) },
////                navigationIcon = {
////                    IconButton(onClick = onBack) {
////                        Icon(
////                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
////                            contentDescription = "Back",
////                            tint = deeepBlue
////                        )
////                    }
////                },
////                colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
////            )
////        }
////    ) { innerPadding ->
////        Column(
////            modifier = Modifier
////                .fillMaxSize()
////                .background(ppureWhite)
////                .padding(innerPadding)
////                .padding(16.dp)
////        ) {
////            Text(
////                text = "Buy",
////                fontSize = 32.sp,
////                fontWeight = FontWeight.Bold,
////                color = deeepBlue,
////                modifier = Modifier.padding(bottom = 16.dp)
////            )
////
////            var expanded by remember { mutableStateOf(false) }
////            OutlinedButton(
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(bottom = 10.dp),
////                onClick = { expanded = true },
////                colors = ButtonDefaults.outlinedButtonColors(
////                    containerColor = ppureWhite,
////                    contentColor = aaccentBlue
////                ),
////                border = BorderStroke(1.dp, aaccentBlue),
////                shape = RoundedCornerShape(15.dp)
////            ) {
////                Text(
////                    if (tickerSymbol == "") "Select Stock"
////                    else "Selected Stock: $tickerSymbol",
////                    fontWeight = FontWeight.Medium
////                )
////            }
////
////            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
////                for (stock in sandbox.allStocks) {
////                    DropdownMenuItem(
////                        text = { Text(stock.ticker) },
////                        onClick = {
////                            tickerSymbol = stock.ticker
////                            expanded = false
////                            valuePerShare = stock.historicPrice.last()
////                        }
////                    )
////                }
////            }
////
////            OutlinedTextField(
////                value = numberOfShares.toString(),
////                keyboardOptions = KeyboardOptions(
////                    keyboardType = KeyboardType.Number,
////                    imeAction = ImeAction.Done
////                ),
////                onValueChange = { input: String ->
////                    numberOfShares = if (input == "") 0 else input.toInt()
////                },
////                label = { Text("No. of Shares", color = secondaryBlackk) },
////                shape = RoundedCornerShape(15.dp),
////                colors = OutlinedTextFieldDefaults.colors(
////                    focusedBorderColor = aaccentBlue,
////                    unfocusedBorderColor = ssoftBlue
////                ),
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(vertical = 16.dp)
////            )
////
////            Card(
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(vertical = 16.dp),
////                colors = CardDefaults.cardColors(containerColor = llightBlue),
////                shape = RoundedCornerShape(20.dp),
////                elevation = CardDefaults.cardElevation(4.dp)
////            ) {
////                Column(modifier = Modifier.padding(16.dp)) {
////                    Text(
////                        text = "USD ($)",
////                        fontSize = 16.sp,
////                        fontWeight = FontWeight.SemiBold,
////                        color = deeepBlue,
////                        modifier = Modifier.padding(bottom = 8.dp)
////                    )
////
////                    FinancialInfoRow("Value per share", "$$valuePerShare")
////                    FinancialInfoRow("Available Cash", "$$availableCash")
////                    FinancialInfoRow("Total Cost", "$$totalCost")
////                    Divider(
////                        color = secondaryBlackk.copy(alpha = 0.2f),
////                        modifier = Modifier.padding(vertical = 8.dp)
////                    )
////                    FinancialInfoRow("Resulting Cash", "$$resultingCash")
////                }
////            }
////
//////            Spacer(modifier = Modifier.weight(1f))
////
////            Button(
////                onClick = {
////                    if (numberOfShares > 0 && resultingCash >= 0 && tickerSymbol !== "") {
////                        sandbox.buy(tickerSymbol, numberOfShares)
////                        MainActivity.model.saveSandbox(sandbox.id, sandbox)
////                        onBack()
////                    }
////                },
////                shape = RoundedCornerShape(15.dp),
////                colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
////                elevation = ButtonDefaults.buttonElevation(8.dp),
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(vertical = 8.dp)
////            ) {
////                Text(
////                    text = "Confirm Purchase",
////                    color = ppureWhite,
////                    fontSize = 16.sp,
////                    fontWeight = FontWeight.Bold,
////                    modifier = Modifier.padding(vertical = 8.dp)
////                )
////            }
////        }
////    }
////}
//
//@OptIn(ExperimentalMaterial3Api::class)
//@Composable
//fun SellPage(sandbox: Sandbox, onBack: () -> Unit) {
//    var tickerSymbol by remember { mutableStateOf("") }
//    var numberOfShares by remember { mutableStateOf<Int>(0) }
//    var valuePerShare by remember { mutableFloatStateOf(0.0f) }
//    var sharesAvailable by remember { mutableStateOf(0) }
//
//    val availableCash = sandbox.cash
//    val totalRevenue:Float = numberOfShares * valuePerShare
//    val resultingCash = availableCash + totalRevenue
//
//    Scaffold(
//        topBar = {
//            TopAppBar(
//                title = { Text("Sell", color = deeepBlue, fontWeight = FontWeight.Bold) },
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
//        ) {
//            Text(
//                text = "Sell",
//                fontSize = 32.sp,
//                fontWeight = FontWeight.Bold,
//                color = deeepBlue,
//                modifier = Modifier.padding(bottom = 16.dp)
//            )
//
//            var expanded by remember { mutableStateOf(false) }
//            OutlinedButton(
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(bottom = 10.dp),
//                onClick = { expanded = true },
//                colors = ButtonDefaults.outlinedButtonColors(
//                    containerColor = ppureWhite,
//                    contentColor = aaccentBlue
//                ),
//                border = BorderStroke(1.dp, aaccentBlue),
//                shape = RoundedCornerShape(15.dp)
//            ) {
//                Text(
//                    if (tickerSymbol == "") "Select Stock"
//                    else "Selected Stock: $tickerSymbol",
//                    fontWeight = FontWeight.Medium
//                )
//            }
//
//            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
//                for (stock in sandbox.ownedStocks) {
//                    DropdownMenuItem(
//                        text = { Text(stock.ticker) },
//                        onClick = {
//                            tickerSymbol = stock.ticker
//                            expanded = false
//                            valuePerShare = sandbox.getStockValue(stock.ticker)
//                            sharesAvailable = stock.quantityOwned
//                        }
//                    )
//                }
//            }
//
//            OutlinedTextField(
//                value = numberOfShares.toString(),
//                keyboardOptions = KeyboardOptions(
//                    keyboardType = KeyboardType.Number,
//                    imeAction = ImeAction.Done
//                ),
//                onValueChange = { input: String ->
//                    numberOfShares = if (input == "") 0 else input.toInt()
//                },
//                label = { Text("No. of Shares (${sharesAvailable} shares available)", color = secondaryBlackk) },
//                shape = RoundedCornerShape(15.dp),
//                colors = OutlinedTextFieldDefaults.colors(
//                    focusedBorderColor = aaccentBlue,
//                    unfocusedBorderColor = ssoftBlue
//                ),
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(vertical = 16.dp)
//            )
//
//            Card(
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(vertical = 16.dp),
//                colors = CardDefaults.cardColors(containerColor = llightBlue),
//                shape = RoundedCornerShape(20.dp),
//                elevation = CardDefaults.cardElevation(4.dp)
//            ) {
//                Column(modifier = Modifier.padding(16.dp)) {
//                    Text(
//                        text = "USD ($)",
//                        fontSize = 16.sp,
//                        fontWeight = FontWeight.SemiBold,
//                        color = deeepBlue,
//                        modifier = Modifier.padding(bottom = 8.dp)
//                    )
//
//                    FinancialInfoRow("Value per share", "$$valuePerShare")
//                    FinancialInfoRow("Available Cash", "$$availableCash")
//                    FinancialInfoRow("Total Revenue", "$$totalRevenue")
//                    Divider(
//                        color = secondaryBlackk.copy(alpha = 0.2f),
//                        modifier = Modifier.padding(vertical = 8.dp)
//                    )
//                    FinancialInfoRow("Resulting Cash", "$$resultingCash")
//                }
//            }
//
////            Spacer(modifier = Modifier.weight(1f))
//
//            Button(
//                onClick = {
//                    if (numberOfShares > 0 && numberOfShares <= sharesAvailable && tickerSymbol !== "") {
//                        sandbox.sell(tickerSymbol, numberOfShares)
//                        MainActivity.model.saveSandbox(sandbox.id, sandbox)
//                        onBack()
//                    }
//                },
//                shape = RoundedCornerShape(15.dp),
//                colors = ButtonDefaults.buttonColors(containerColor = deeepBlue),
//                elevation = ButtonDefaults.buttonElevation(8.dp),
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(vertical = 8.dp)
//            ) {
//                Text(
//                    text = "Confirm Sale",
//                    color = ppureWhite,
//                    fontSize = 16.sp,
//                    fontWeight = FontWeight.Bold,
//                    modifier = Modifier.padding(vertical = 8.dp)
//                )
//            }
//        }
//    }
//}
//
////@Composable
////fun FinancialInfoRow(label: String, value: String) {
////    Row(
////        modifier = Modifier
////            .fillMaxWidth()
////            .padding(vertical = 4.dp),
////        horizontalArrangement = Arrangement.SpaceBetween,
////        verticalAlignment = Alignment.CenterVertically
////    ) {
////        Text(
////            text = label,
////            fontSize = 16.sp,
////            color = secondaryBlackk
////        )
////        Text(
////            text = value,
////            fontSize = 16.sp,
////            color = pprimaryBlack,
////            fontWeight = FontWeight.Bold
////        )
////    }
////}
//
//fun handleNewEventClick() = runBlocking {
//
//}
//
////@OptIn(ExperimentalMaterial3Api::class)
////@Composable
////fun NewsPage(sandbox: Sandbox, onBack: () -> Unit) {
////    var newsList by remember { mutableStateOf<List<NewsEvent>>(sandbox.newsEvents) }
////    var selectedNews by remember { mutableStateOf<NewsEvent?>(null) }
////
////    if (selectedNews != null) {
////        NewsDetailPage(newsItem = selectedNews!!, onBack = { selectedNews = null }, sandbox)
////    } else {
////        Scaffold(
////            topBar = {
////                TopAppBar(
////                    title = { Text("News", color = deeepBlue, fontWeight = FontWeight.Bold) },
////                    navigationIcon = {
////                        IconButton(onClick = onBack) {
////                            Icon(
////                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
////                                contentDescription = "Back",
////                                tint = deeepBlue
////                            )
////                        }
////                    },
////                    actions = {
////                        Button(
////                            onClick = {
////                                runBlocking {
////                                    val newEvent = async {
////                                        MainActivity.model.generateNewsEvent(sandbox.allStocks)
////                                    }
////                                    newsList += newEvent.await()
////                                    sandbox.newsEvents = newsList as MutableList<NewsEvent>
////                                    sandbox.latestNewsNotRevealed = newEvent.await().title
////                                    MainActivity.model.saveSandbox(sandbox.id, sandbox)
////                                }},
////                            colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
////                            shape = RoundedCornerShape(15.dp),
////                            elevation = ButtonDefaults.buttonElevation(4.dp),
////                            enabled = sandbox.latestNewsNotRevealed.length == 0
////                        ) {
////                            Text(
////                                text = "New Event",
////                                color = ppureWhite,
////                                fontWeight = FontWeight.Bold
////                            )
////                        }
////                    },
////                    colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
////                )
////            }
////        ) { innerPadding ->
////
////            LazyColumn(
////                modifier = Modifier
////                    .fillMaxSize()
////                    .background(ppureWhite)
////                    .padding(innerPadding)
////                    .padding(16.dp)
////
////
////            ) {
////                items(newsList.reversed()) { newsItem ->
////                    NewsListItem(newsItem = newsItem, onClick = { selectedNews = newsItem })
////                }
////            }
////        }
////    }
////}
//
////@Composable
////fun NewsListItem(newsItem: NewsEvent, onClick: () -> Unit) {
////    Card(
////        modifier = Modifier
////            .fillMaxWidth()
////            .padding(vertical = 8.dp)
////            .clickable { onClick() },
////        colors = CardDefaults.cardColors(containerColor = llightBlue),
////        shape = RoundedCornerShape(15.dp),
////        elevation = CardDefaults.cardElevation(2.dp)
////    ) {
////        Column(
////            modifier = Modifier.padding(16.dp)
////        ) {
////            Text(
////                text = newsItem.title,
////                fontWeight = FontWeight.Bold,
////                fontSize = 20.sp,
////                color = pprimaryBlack
////            )
////            Spacer(modifier = Modifier.height(8.dp))
////            Text(
////                text = newsItem.body,
////                fontSize = 14.sp,
////                maxLines = 2,
////                overflow = TextOverflow.Ellipsis,
////                color = secondaryBlackk
////            )
////        }
////    }
////}
////
////@OptIn(ExperimentalMaterial3Api::class)
////@Composable
////fun NewsDetailPage(newsItem: NewsEvent, onBack: () -> Unit, sandbox: Sandbox) {
////    var latestNewsNotRevealed by remember { mutableStateOf(sandbox.latestNewsNotRevealed)}
////    Scaffold(
////        topBar = {
////            TopAppBar(
////                title = {
////                    Text(
////                        "News Event",
////                        color = deeepBlue,
////                        fontWeight = FontWeight.Bold
////                    )
////                },
////                navigationIcon = {
////                    IconButton(onClick = onBack) {
////                        Icon(
////                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
////                            contentDescription = "Back",
////                            tint = deeepBlue
////                        )
////                    }
////                },
////                colors = TopAppBarDefaults.topAppBarColors(containerColor = ppureWhite)
////            )
////        }
////    ) { innerPadding ->
////        Column(
////            modifier = Modifier
////                .fillMaxSize()
////                .background(ppureWhite)
////                .padding(innerPadding)
////                .padding(16.dp)
////                .padding(bottom = 100.dp)
////                .verticalScroll(ScrollState(0), true)
////
////        ) {
////            Text(
////                text = newsItem.title,
////                fontSize = 24.sp,
////                fontWeight = FontWeight.Bold,
////                color = pprimaryBlack
////            )
////            Spacer(modifier = Modifier.height(16.dp))
////            Text(
////                text = newsItem.body,
////                fontSize = 16.sp,
////                color = secondaryBlackk,
////                lineHeight = 24.sp
////            )
////            Spacer(modifier = Modifier.height(16.dp))
////            Text(
////                text = "New Stock Prices:",
////                fontSize = 16.sp,
////                color = secondaryBlackk,
////                lineHeight = 24.sp
////            )
////            if (latestNewsNotRevealed == newsItem.title) {
////                Text(
////                    text = "The market's about to close! Make your trades before the impact of the news above reaches the market (stock prices are updated). Click the button below when you're done.",
////                    fontSize = 16.sp,
////                    color = secondaryBlackk,
////                    lineHeight = 24.sp
////                )
////                Button(
////                    onClick = {
////                        sandbox.allStocks.forEachIndexed { index, element ->
////                            for (newPrice in newsItem.newPrices) {
////                                if (element.ticker == newPrice.ticker) sandbox.allStocks[index].historicPrice.add(newPrice.newPrice)
////                            }
////                        }
////                        latestNewsNotRevealed = ""
////                        sandbox.latestNewsNotRevealed = ""
////                        MainActivity.model.saveSandbox(sandbox.id, sandbox)
////                    }
////                ) {
////                    Text("Reveal Prices")
////                }
////            } else {
////                newsItem.newPrices.forEach { newPrice ->
////                    Text(
////                        text = newPrice.ticker + ": " + newPrice.newPrice.toString() + "( " + (if (newPrice.changeInPrice >= 0) "+" else "") + newPrice.changeInPrice.toString() + ")",
////                        fontSize = 16.sp,
////                        color = secondaryBlackk,
////                        lineHeight = 24.sp
////                    )
////                }
////            }
//////            Card(
//////                modifier = Modifier.fillMaxWidth(),
//////                colors = CardDefaults.cardColors(containerColor = llightBlue),
//////                shape = RoundedCornerShape(20.dp),
//////                elevation = CardDefaults.cardElevation(4.dp)
//////            ) {
//////                Column(modifier = Modifier.padding(16.dp)) {
//////                    Text(
//////                        text = newsItem.title,
//////                        fontSize = 24.sp,
//////                        fontWeight = FontWeight.Bold,
//////                        color = pprimaryBlack
//////                    )
//////                    Spacer(modifier = Modifier.height(16.dp))
//////                    Text(
//////                        text = newsItem.body,
//////                        fontSize = 16.sp,
//////                        color = secondaryBlackk,
//////                        lineHeight = 24.sp
//////                    )
//////                    Spacer(modifier = Modifier.height(16.dp))
//////                    Text(
//////                        text = "New Stock Prices:",
//////                        fontSize = 16.sp,
//////                        color = secondaryBlackk,
//////                        lineHeight = 24.sp
//////                    )
//////                    newsItem.newPrices.forEach { newPrice ->
//////                        Text(
//////                            text = newPrice.ticker + ": " + newPrice.newPrice.toString(),
//////                            fontSize = 16.sp,
//////                            color = secondaryBlackk,
//////                            lineHeight = 24.sp
//////                        )
//////                    }
//////                }
//////            }
////        }
////    }
////}
//
////data class NewsItem(val id: Int, val title: String, val content: String)
////
//////fun sampleNewsList() = listOf(
//////    NewsItem(1, "Something happened!", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut et massa mi..."),
//////    NewsItem(2, "Market update!", "Mauris aliquet ultricies ante, non malesuada odio malesuada at. Aenean sit..."),
//////    NewsItem(3, "New regulations", "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere...")
//////)