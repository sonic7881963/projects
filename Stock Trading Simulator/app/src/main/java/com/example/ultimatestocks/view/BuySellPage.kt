//package com.example.ultimatestocks.view
//
//import androidx.compose.foundation.Image
//import androidx.compose.foundation.ScrollState
//import androidx.compose.foundation.layout.Column
//import androidx.compose.foundation.layout.Row
//import androidx.compose.foundation.layout.Spacer
//import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.fillMaxWidth
//import androidx.compose.foundation.layout.height
//import androidx.compose.foundation.layout.padding
//import androidx.compose.foundation.shape.RoundedCornerShape
//import androidx.compose.foundation.verticalScroll
//import androidx.compose.material.icons.Icons
//import androidx.compose.material.icons.filled.Home
//import androidx.compose.material3.Button
//import androidx.compose.material3.ButtonDefaults.buttonColors
//import androidx.compose.material3.ButtonDefaults.buttonElevation
//import androidx.compose.material3.Divider
//import androidx.compose.material3.HorizontalDivider
//import androidx.compose.material3.OutlinedTextField
//import androidx.compose.material3.Text
//import androidx.compose.material3.TextField
//import androidx.compose.material3.TextFieldColors
//import androidx.compose.material3.TextFieldDefaults
//import androidx.compose.runtime.Composable
//import androidx.compose.runtime.getValue
//import androidx.compose.runtime.mutableStateOf
//import androidx.compose.runtime.remember
//import androidx.compose.runtime.setValue
//import androidx.compose.ui.Alignment
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Color
//import androidx.compose.ui.graphics.vector.rememberVectorPainter
//import androidx.compose.ui.layout.ContentScale
//import androidx.compose.ui.layout.VerticalAlignmentLine
//import androidx.compose.ui.res.painterResource
//import androidx.compose.ui.text.TextStyle
//import androidx.compose.ui.text.font.FontWeight
//import androidx.compose.ui.text.style.TextAlign
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//import cafe.adriel.voyager.core.screen.Screen
//import cafe.adriel.voyager.navigator.LocalNavigator
//import cafe.adriel.voyager.navigator.currentOrThrow
//import cafe.adriel.voyager.navigator.tab.Tab
//import cafe.adriel.voyager.navigator.tab.TabOptions
//import com.example.ultimatestocks.MainActivity
//import com.example.ultimatestocks.MainActivity.Companion.model
//import com.example.ultimatestocks.R
//import org.intellij.lang.annotations.JdkConstants.HorizontalAlignment
//
//private val buyBlue = Color(0xFF2196F3)
//private val sellRed = Color(0xFFFF5252)
//private val bgColor = Color(0xFFE3F2FD)
//private val darkText = Color(0xFF333333)
//
//object BuySellScreen : Screen {
//    @Composable
//    override fun Content() {
//        val SandCompState = model.SandCompState.ifEmpty { "Sandbox" }
//        val BuySellState = model.BuySellState.ifEmpty { "Sell" }
//        val CompSand = if (SandCompState == "Sandbox") "Open Sandbox" else "Ongoing Competition"
//        val currentCompSand = if (SandCompState == "Sandbox") "Sandbox 1" else "2024 USA Federal Elections"
//        val buttonState = if (BuySellState == "Buy") "Confirm Purchase" else "Confirm Sale"
//        val mainColor = if (BuySellState == "Buy") buyBlue else sellRed
//
//        val navigation = LocalNavigator.currentOrThrow
//
//        Column(
//            modifier = Modifier
//                .fillMaxSize()
//                .padding(top = 16.dp, start = 16.dp, end = 16.dp, bottom = 80.dp)
//                .verticalScroll(ScrollState(0), true)
//        ) {
//            Logo()
//            Spacer(modifier = Modifier.height(8.dp))
//            topLabel(CompSand, currentCompSand)
//            Spacer(modifier = Modifier.height(8.dp))
//            Text(
//                text = BuySellState,
//                color = mainColor,
//                fontSize = 30.sp,
//                fontWeight = FontWeight.Bold
//            )
//            Spacer(modifier = Modifier.height(8.dp))
//            Column(
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(10.dp)
//            ) {
//                Text(
//                    text = "Ticker Symbol",
//                    color = darkText,
//                    fontWeight = FontWeight.Bold
//                )
//                SearchFields("TCK, ABC, etc.")
//                Spacer(modifier = Modifier.height(8.dp))
//                Text(
//                    text = "Number of Shares",
//                    color = darkText,
//                    fontWeight = FontWeight.Bold
//                )
//                SearchFields("0")
//            }
//            Spacer(modifier = Modifier.height(12.dp))
//
//            Text(
//                text = "USD ($)",
//                textAlign = TextAlign.End,
//                color = darkText,
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(10.dp),
//                fontWeight = FontWeight.Bold
//            )
//
//            PriceLabel("Value per share", "34.2", 10)
//            PriceLabel("Available Cash", "344.2", 0)
//            PriceLabel("Gross Revenue", "342", 0)
//            HorizontalDivider(
//                color = darkText.copy(alpha = 0.2f),
//                thickness = 1.dp,
//                modifier = Modifier.padding(10.dp)
//            )
//            PriceLabel("Resulting Cash", "686.2", 0)
//
//            Spacer(modifier = Modifier.height(20.dp))
//
//            Button(
//                onClick = { navigation.pop() },
//                colors = buttonColors(mainColor),
//                modifier = Modifier.fillMaxWidth(),
//                shape = RoundedCornerShape(15.dp),
//                elevation = buttonElevation(8.dp)
//            ) {
//                Text(
//                    buttonState,
//                    color = Color.White,
//                    fontSize = 17.sp
//                )
//            }
//        }
//    }
//}
//
//@Composable
//fun Logo() {
//    Row(
//        modifier = Modifier.height(60.dp)
//    ) {
//        Text(
//            modifier = Modifier.padding(top = 16.dp),
//            text = "USX",
//            fontSize = 30.sp,
//            fontWeight = FontWeight.Bold,
//            color = darkText
//        )
//    }
//}
//
//@Composable
//fun topLabel(CompSand: String, currentCompSand: String) {
//    Text(
//        text = CompSand,
//        color = darkText,
//        fontSize = 15.sp,
//        fontWeight = FontWeight.SemiBold,
//    )
//    Text(
//        text = currentCompSand,
//        color = darkText,
//        fontSize = 24.sp,
//        fontWeight = FontWeight.Bold
//    )
//}
//
//@Composable
//fun SearchFields(placeholder: String) {
//    var text by remember { mutableStateOf("") }
//    OutlinedTextField(
//        value = text,
//        onValueChange = { text = it },
//        textStyle = TextStyle(color = darkText, fontSize = 18.sp),
//        placeholder = { Text(placeholder, color = Color.LightGray) },
//        shape = RoundedCornerShape(15.dp),
//        modifier = Modifier.fillMaxWidth()
//    )
//}
//
//@Composable
//fun PriceLabel(label: String, price: String, labelPadding: Int) {
//    Row(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(horizontal = 10.dp)
//    ) {
//        Text(
//            text = label,
//            textAlign = TextAlign.Start,
//            color = darkText,
//            fontWeight = FontWeight.Bold,
//            modifier = Modifier.padding(bottom = labelPadding.dp)
//        )
//        Text(
//            text = price,
//            textAlign = TextAlign.End,
//            modifier = Modifier.fillMaxWidth(),
//            color = darkText,
//            fontWeight = FontWeight.Bold
//        )
//    }
//}
//
//@Preview(showBackground = true)
//@Composable
//fun PreviewDashboardScreen() {
//    BuySellScreen
//}