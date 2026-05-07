package com.example.ultimatestocks

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.tab.Tab
import cafe.adriel.voyager.navigator.tab.TabOptions
import com.example.ultimatestocks.compete.CompeteHomePage
import com.example.ultimatestocks.competeAdmin.CompeteAdminHomePage


object CompeteTab : Tab {
    override val options: TabOptions
        @Composable
        get() {
            val icon = rememberVectorPainter(Icons.Default.Star)
            return remember {
                TabOptions(
                    index = 1u,
                    title = "Compete",
                    icon = icon
                )
            }
        }

    @Composable
    override fun Content() {
        if (MainActivity.model.isAdmin) Navigator(CompeteAdminHomePage()) else Navigator(CompeteHomePage())
    }
}