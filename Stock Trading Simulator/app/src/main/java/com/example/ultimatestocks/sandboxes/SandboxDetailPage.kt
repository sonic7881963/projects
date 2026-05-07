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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.LocalNavigator
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
import com.example.ultimatestocks.sharedPages.BuyPage
import com.example.ultimatestocks.MyCustomTextField
import com.example.ultimatestocks.StockList
import com.example.ultimatestocks.aaccentBlue
import com.example.ultimatestocks.deeepBlue
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.entities.StockInventory
import com.example.ultimatestocks.llightBlue
import com.example.ultimatestocks.ppureWhite
import com.example.ultimatestocks.sharedPages.NewsPage
import com.example.ultimatestocks.sharedPages.SellPage

fun filterStocks(query: String, allStocks: List<StockInventory>): List<StockInventory> {
    if (query.isEmpty()) {
        return allStocks
    }
    return allStocks.filter {
        it.ticker.contains(query, ignoreCase = true)
    }
}


class SandboxDetailPage(val originalSandbox: Sandbox): Screen {
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        val navigator = LocalNavigator.currentOrThrow

        val sandbox by remember { mutableStateOf<Sandbox>(originalSandbox) }
        var searchQuery by remember { mutableStateOf("") }
        var showBuyPage by remember { mutableStateOf(false) }
        var showSellPage by remember { mutableStateOf(false) }
        var showNewsPage by remember { mutableStateOf(false) }


        when {
            showBuyPage -> BuyPage(sandbox) { showBuyPage = false }
            showSellPage -> SellPage(sandbox) { showSellPage = false }
            showNewsPage -> NewsPage(sandbox) { showNewsPage = false }
            else -> Scaffold(
                topBar = {
                    TopAppBar(
                        title = { Text(sandbox.name, color = deeepBlue, fontWeight = FontWeight.Bold) },
                        navigationIcon = {
                            IconButton(onClick = { navigator.push(SandboxesHomePage()) }) {
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
                        .background(color = ppureWhite)
                        .padding(innerPadding)
                        .padding(16.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = sandbox.name,
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = deeepBlue
                        )
                        Button(
                            onClick = { showNewsPage = true },
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

                    Spacer(modifier = Modifier.height(16.dp))

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
                            val allStocks = sandbox.allStocks

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
                                        color = Color.Gray,
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

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Button(
                            onClick = { showBuyPage = true },
                            shape = RoundedCornerShape(15.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = aaccentBlue),
                            elevation = ButtonDefaults.buttonElevation(8.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = "Buy",
                                color = ppureWhite,
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Button(
                            onClick = { showSellPage = true },
                            shape = RoundedCornerShape(15.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = deeepBlue),
                            elevation = ButtonDefaults.buttonElevation(8.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = "Sell",
                                color = ppureWhite,
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    MyCustomTextField(searchQuery) { query ->
                        searchQuery = query
                    }

                    val filteredStocks = filterStocks(searchQuery, sandbox.ownedStocks)

                    Text(
                        text = "Current Portfolio",
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold,
                        color = deeepBlue,
                        modifier = Modifier.padding(vertical = 16.dp)
                    )

                    StockList(stocks = filteredStocks)
                }
            }
        }
    }
}