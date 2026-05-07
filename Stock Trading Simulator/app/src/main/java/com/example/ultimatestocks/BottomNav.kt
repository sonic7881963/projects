package com.example.ultimatestocks

import androidx.compose.material3.Text
import android.annotation.SuppressLint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import cafe.adriel.voyager.core.annotation.ExperimentalVoyagerApi
import cafe.adriel.voyager.core.lifecycle.LifecycleEffectOnce
import cafe.adriel.voyager.core.screen.Screen
import cafe.adriel.voyager.navigator.tab.CurrentTab
import cafe.adriel.voyager.navigator.tab.LocalTabNavigator
import cafe.adriel.voyager.navigator.tab.Tab
import cafe.adriel.voyager.navigator.tab.TabNavigator
import com.example.ultimatestocks.MainActivity.Companion.model
import com.example.ultimatestocks.sandboxes.LearnTab
import com.example.ultimatestocks.ui.theme.UltimateStocksTheme  // Added this import


class BottomNav : Screen {
    @OptIn(ExperimentalVoyagerApi::class)
    @SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
    @Composable
    override fun Content(){
        LifecycleEffectOnce {
            MainActivity.model.getSandboxes()
            MainActivity.model.getAllStocks()
            MainActivity.model.getCompetitions()
        }
            UltimateStocksTheme {
                TabNavigator(DashboardTab) {
                    Scaffold(
                        modifier = Modifier.fillMaxSize(),
                        bottomBar = {
                            if (MainActivity.model.auth.currentUser !== null) {

                                NavigationBar {
                                    TabNavigationItem(DashboardTab)
                                    TabNavigationItem(CompeteTab)
                                    TabNavigationItem(LearnTab)
                                    TabNavigationItem(ProfileTab)
                                }
                            }
                        }
                    ) {
                        CurrentTab()
                    }
                }
            }

    }
}

@Composable
private fun RowScope.TabNavigationItem(tab: Tab) {
    val tabNavigator = LocalTabNavigator.current

    NavigationBarItem(
        selected = tabNavigator.current == tab,
        onClick = { tabNavigator.current = tab },
        icon = { Icon(painter = tab.options.icon!!, contentDescription = tab.options.title) },
        label = { Text(text = tab.options.title) }
    )
}
