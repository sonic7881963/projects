package com.example.ultimatestocks

import com.example.ultimatestocks.entities.Competition
import com.example.ultimatestocks.entities.NewsEvent
import com.example.ultimatestocks.entities.Player
import com.example.ultimatestocks.entities.StockDetails
import com.example.ultimatestocks.entities.StockInventory
import com.example.ultimatestocks.entities.StockNewPrice
import org.junit.Assert.*
import org.junit.Test

class CompetitionTests {
    @Test
    fun testCompetition() {
        val comp = Competition(
            title = "Test Competition",
            availableStocks = listOf(StockDetails("Tck", "Stock Description", mutableListOf(100.0f, 200.0f))),
            hasStarted = false,
            hasEnded = false,
            initialCash = 1000.0f,
            players = mutableMapOf("Test Player" to Player("player", 100.0f, mutableListOf(StockInventory("Tck", 1)))),
            maxNewsEvents = 2,
            newsEvents = mutableListOf(NewsEvent("News", "News Body", listOf(StockNewPrice("Tck", 100.0f, 20.0f, )))),
            latestNewsNotRevealed = ""
        )

        assertEquals("Test Competition", comp.title)
        assertEquals(false, comp.hasStarted)
        assertEquals(false, comp.hasEnded)
        assertEquals(200.0f, comp.getStockValue("Tck"), 0.001f)
        assertEquals(0.0f, comp.getStockValue("Invalid Stock"), 0.001f)

        val player = Player("Player", 100.0f, mutableListOf(StockInventory("Tck", 2)))
        assertEquals(500.0f, player.getPortfolioValue(comp), 0.001f)
    }
}