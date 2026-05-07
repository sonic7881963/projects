//package com.example.ultimatestocks.view
//
//import androidx.compose.foundation.ScrollState
//import androidx.compose.foundation.clickable
//import androidx.compose.foundation.gestures.detectTapGestures
//import androidx.compose.foundation.layout.Column
//import androidx.compose.foundation.layout.Row
//import androidx.compose.foundation.layout.Spacer
//import androidx.compose.foundation.layout.fillMaxHeight
//import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.fillMaxWidth
//import androidx.compose.foundation.layout.height
//import androidx.compose.foundation.layout.padding
//import androidx.compose.foundation.verticalScroll
//import androidx.compose.material.icons.Icons
//import androidx.compose.material.icons.filled.Home
//import androidx.compose.material3.HorizontalDivider
//import androidx.compose.material3.Text
//import androidx.compose.runtime.Composable
//import androidx.compose.runtime.remember
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Color
//import androidx.compose.ui.graphics.vector.rememberVectorPainter
//import androidx.compose.ui.input.pointer.pointerInput
//import androidx.compose.ui.text.font.FontWeight
//import androidx.compose.ui.tooling.preview.Preview
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//import cafe.adriel.voyager.core.screen.Screen
//import cafe.adriel.voyager.navigator.LocalNavigator
//import cafe.adriel.voyager.navigator.currentOrThrow
//import cafe.adriel.voyager.navigator.tab.LocalTabNavigator
//import cafe.adriel.voyager.navigator.tab.Tab
//import cafe.adriel.voyager.navigator.tab.TabNavigator
//import cafe.adriel.voyager.navigator.tab.TabOptions
//import com.example.ultimatestocks.MainActivity.Companion.model
//import com.example.ultimatestocks.entities.NewsEvent
//
////object NewsPage : Tab {
////    private  fun readResolve(): Any = NewsPage
////    override val options: TabOptions
////        @Composable
////        get(){
////            val icon = rememberVectorPainter(Icons.Default.Home)
////            return remember {
////                TabOptions(
////                    index = 0u,
////                    title = "Home",
////                    icon = icon
////                )
////            }
////        }
////
////    @Composable
////    override fun Content(){
//////        BuySellScreen()
////        NewsScreen(
////            tabNavigation = LocalTabNavigator.current
////        )
////    }
////}
//
//
//object NewsScreen: Screen {
//    @Composable
//    override fun Content(){
//        val SandCompState = model.SandCompState.ifEmpty{"Sandbox"}
//        val CompSand = if (SandCompState == "Sandbox") "Open Sandbox" else "Ongoing Competition"
//        // current sandbox selected or current compeition
//        val currentCompSand = if (SandCompState == "Sandbox") "Sandbox 1" else "2024 USA Federal Elections"
//        val newsList = model.NewsList
//
//        if (newsList.isEmpty()) newsList.add(NewsEvent("Something happened!", "Sample", listOf()))
//        val navgiation = LocalNavigator.currentOrThrow
//
//        Column(
//            modifier = Modifier
//                .fillMaxSize()
//                .padding(top = 16.dp, start = 16.dp, end = 16.dp, bottom = 80.dp)
//                .verticalScroll(ScrollState(0), true)
//        ){
//            Logo()
//            Spacer(modifier = Modifier.height(8.dp))
//            topLabel(CompSand, currentCompSand)
//            Spacer(modifier = Modifier.height(4.dp))
//            Text(
//                text = "News",
//                fontWeight = FontWeight.Bold,
//                fontSize = 20.sp
//            )
//            Spacer(modifier = Modifier.height(10.dp))
//            Text(
//                text = "< Back",
//                fontWeight = FontWeight.Bold,
//                fontSize = 20.sp,
//                modifier = Modifier.clickable(onClick = {navgiation.push(NewsListScreen)})
//            )
//            news()
//        }
//    }
//
//}
//
//
//@Composable
//fun news(){
//    Column(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(10.dp)
////            .clickable(onClick = {nav(NewsEvent(newsTitle, news),tabNavigation)})
////            .pointerInput(tabNavigation.current = BuySellPage){
////                detectTapGestures {
////                    onTap = { tabNavigation.current = BuySellPage }
////                }
////            }
//    ){
//        Text(
//            text = model.selectedNews.title,
//            fontWeight = FontWeight.Bold,
//            fontSize = 30.sp,
//        )
//        Row(
//            modifier = Modifier.fillMaxHeight()
//        ){
//            Spacer(modifier = Modifier.height(10.dp))
//            Text(
//                text = model.selectedNews.body,
//                modifier =  Modifier.weight(1f),
//                fontSize = 22.sp
//            )
//        }
//    }
////    HorizontalDivider(
////        color = Color.Gray,
////        thickness = 3.dp,
////        modifier = Modifier.padding(10.dp)
////    )
//}